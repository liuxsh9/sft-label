"""Trajectory semantic clustering with pinned-prefix windowing + SemHash + ANN.

This module is intentionally CPU-first and dependency-light. It provides:
  - Long trajectory segmentation with pinned task-definition prefix
  - Bilingual role-aware text rendering for embeddings
  - Local lightweight embedding backend + API embedding fallback backend
  - Deterministic SemHash projection and band indexing
  - Candidate refinement with cosine-space ANN over coarse candidates
  - Cluster assembly, representative selection by SNR, and artifact export
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import heapq
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import httpx

from sft_label.config import PipelineConfig
from sft_label.preprocessing import estimate_tokens, normalize_sample
from sft_label.semantic_artifacts import (
    SEMANTIC_MANIFEST_FILE,
    SEMANTIC_STATS_FILE,
    resolve_semantic_output_dir,
)


SemanticProgressCb = Callable[[str, str, int | None, int | None], None]


ROLE_LABELS = {
    "human": "USER",
    "gpt": "ASSISTANT",
    "tool": "TOOL",
    "system": "SYSTEM",
}


@dataclass
class TrajectoryWindow:
    window_id: str
    source_id: str
    window_index: int
    total_windows: int
    turn_start: int
    turn_end: int
    pinned_prefix_turns: int
    conversations: list
    pinned_prefix: list
    body: list
    value_score: float | None = None
    source_file: str | None = None


@dataclass
class WindowEmbedding:
    window_id: str
    source_id: str
    dim: int
    provider: str
    model: str
    vector: list


@dataclass
class WindowSemHash:
    window_id: str
    source_id: str
    bits: int
    bits_hex: str
    band_values: list


@dataclass
class ClusterMember:
    cluster_id: str
    window_id: str
    source_id: str
    window_index: int
    snr: float
    action_tokens: int
    observation_tokens: int
    value_score: float | None
    representative: bool


@dataclass
class SemanticRunManifest:
    version: str
    created_at: str
    parameters: dict
    compatibility_key: str
    input_path: str
    output_dir: str
    total_samples: int
    total_windows: int
    total_clusters: int


class EmbeddingProvider:
    """Embedding provider interface."""

    name = "unknown"

    def __init__(self, model: str):
        self.model = model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic lightweight local embedding backend.

    Uses signed hashed character n-grams into a fixed-size vector.
    """

    name = "local"

    def __init__(self, model: str, dim: int = 384):
        super().__init__(model=model)
        self.dim = int(dim)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _iter_ngrams(text: str, n: int = 3):
        if not text:
            return
        txt = f" {text.lower()} "
        if len(txt) <= n:
            yield txt
            return
        for i in range(len(txt) - n + 1):
            yield txt[i:i + n]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for gram in self._iter_ngrams(self._normalize_text(text), n=3):
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=16).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            inv = 1.0 / norm
            vec = [v * inv for v in vec]
        return vec

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class APIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible API embedding backend."""

    name = "api"

    def __init__(self, model: str, base_url: str, api_key: str, timeout: int = 60):
        super().__init__(model=model)
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.base_url:
            raise ValueError("API embedding provider requires base_url")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "input": list(texts)}
        url = f"{self.base_url}/embeddings"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json().get("data") or []

        if not isinstance(data, list):
            raise ValueError("Invalid embedding response format: 'data' must be a list")
        if data and all(isinstance(d, dict) and isinstance(d.get("index"), int) for d in data):
            data = sorted(data, key=lambda d: d["index"])

        vectors = [d.get("embedding") for d in data]
        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        dims = {len(v) for v in vectors if isinstance(v, list)}
        if len(dims) > 1:
            raise ValueError(f"Inconsistent embedding dimensions from API: {sorted(dims)}")
        normalized = []
        for vec in vectors:
            if not isinstance(vec, list):
                raise ValueError("Invalid embedding vector type from API")
            norm = math.sqrt(sum(float(v) * float(v) for v in vec))
            if norm > 0:
                inv = 1.0 / norm
                vec = [float(v) * inv for v in vec]
            else:
                vec = [0.0 for _ in vec]
            normalized.append(vec)
        return normalized


def validate_semantic_config(config: PipelineConfig):
    """Validate semantic clustering config and raise ValueError on invalid values."""
    if config.semantic_long_turn_threshold < 1:
        raise ValueError("semantic_long_turn_threshold must be >= 1")
    if config.semantic_window_size < 2:
        raise ValueError("semantic_window_size must be >= 2")
    if config.semantic_window_stride < 1:
        raise ValueError("semantic_window_stride must be >= 1")
    if config.semantic_pinned_prefix_max_turns < 0:
        raise ValueError("semantic_pinned_prefix_max_turns must be >= 0")
    if config.semantic_embedding_provider not in ("local", "api"):
        raise ValueError("semantic_embedding_provider must be 'local' or 'api'")
    if config.semantic_embedding_dim < 32:
        raise ValueError("semantic_embedding_dim must be >= 32")
    if config.semantic_embedding_batch_size < 1:
        raise ValueError("semantic_embedding_batch_size must be >= 1")
    if config.semantic_embedding_max_workers < 1:
        raise ValueError("semantic_embedding_max_workers must be >= 1")
    if config.semantic_semhash_bits not in (64, 128, 256, 512):
        raise ValueError("semantic_semhash_bits must be one of 64/128/256/512")
    if config.semantic_semhash_bands < 1:
        raise ValueError("semantic_semhash_bands must be >= 1")
    if config.semantic_semhash_bits % config.semantic_semhash_bands != 0:
        raise ValueError("semantic_semhash_bits must be divisible by semantic_semhash_bands")
    if config.semantic_hamming_radius < 0:
        raise ValueError("semantic_hamming_radius must be >= 0")
    if config.semantic_ann_top_k < 1:
        raise ValueError("semantic_ann_top_k must be >= 1")
    if not (0.0 <= config.semantic_ann_sim_threshold <= 1.0):
        raise ValueError("semantic_ann_sim_threshold must be in [0, 1]")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "samples" in payload and isinstance(payload["samples"], list):
            return payload["samples"]
        # Some datasets may be a single sample dict.
        return [payload]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _dedupe_by_stem_prefer_jsonl(files: Sequence[Path]) -> list[Path]:
    grouped: dict[str, Path] = {}
    for file_path in (f.resolve() for f in files if f.is_file()):
        key = str(file_path.with_suffix(""))
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = file_path
            continue
        # Prefer jsonl over json when both variants exist for the same stem.
        if file_path.suffix == ".jsonl" and existing.suffix != ".jsonl":
            grouped[key] = file_path
    return sorted(grouped.values())


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]

    # Prefer scored files when present; fallback to labeled files.
    scored_patterns = (
        "scored*.jsonl", "scored*.json",
        "**/scored*.jsonl", "**/scored*.json",
    )
    scored_files = []
    for ptn in scored_patterns:
        scored_files.extend(input_path.glob(ptn))
    dedup_scored = _dedupe_by_stem_prefer_jsonl(scored_files)
    if dedup_scored:
        return dedup_scored

    labeled_patterns = (
        "labeled*.jsonl", "labeled*.json",
        "**/labeled*.jsonl", "**/labeled*.json",
    )
    labeled_files = []
    for ptn in labeled_patterns:
        labeled_files.extend(input_path.glob(ptn))
    dedup_labeled = _dedupe_by_stem_prefer_jsonl(labeled_files)
    if dedup_labeled:
        return dedup_labeled

    # Final fallback: generic JSON/JSONL
    generic = [f.resolve() for f in input_path.rglob("*") if f.is_file() and f.suffix in (".json", ".jsonl")]
    return _dedupe_by_stem_prefer_jsonl(generic)


def load_samples(input_path: str | Path, limit: int = 0) -> list[dict]:
    path = Path(input_path)
    files = discover_input_files(path)
    if not files:
        raise FileNotFoundError(f"No input files found for semantic clustering: {input_path}")

    samples = []
    sample_seq = 0
    for f in files:
        if f.suffix == ".jsonl":
            iterator = _iter_jsonl(f)
        else:
            iterator = iter(_load_json(f))

        for item in iterator:
            if not isinstance(item, dict):
                continue
            sample_seq += 1
            item = dict(item)
            meta = dict(item.get("metadata") or {})
            meta.setdefault("source_file", str(f))
            meta.setdefault("semantic_input_seq", sample_seq)
            item["metadata"] = meta
            samples.append(item)
            if limit and len(samples) >= limit:
                return samples
    return samples


def _first_index(conversations: Sequence[dict], role: str, start: int = 0) -> int | None:
    for i in range(start, len(conversations)):
        if conversations[i].get("from") == role:
            return i
    return None


def extract_pinned_prefix(conversations: Sequence[dict], max_turns: int = 3) -> tuple[list, int]:
    """Extract task-definition pinned prefix and return (prefix_turns, body_start_idx)."""
    if not conversations or max_turns <= 0:
        return [], 0

    n = len(conversations)
    picked_idx = set()

    # 1) Keep leading system turns.
    i = 0
    while i < n and conversations[i].get("from") == "system":
        if len(picked_idx) < max_turns:
            picked_idx.add(i)
        i += 1

    # 2) Include first task-defining user turn.
    first_h = _first_index(conversations, "human", start=i)
    if first_h is not None and len(picked_idx) < max_turns:
        picked_idx.add(first_h)

        # 3) Include first assistant reply after first user, when available.
        first_g = _first_index(conversations, "gpt", start=first_h + 1)
        if first_g is not None and len(picked_idx) < max_turns:
            picked_idx.add(first_g)

    if not picked_idx:
        # Fallback: deterministic head prefix.
        picked_idx.update(range(min(max_turns, n)))

    # Keep prefix contiguous to avoid dropping intermediate turns
    # (for example tool observations between first human/gpt).
    max_idx = max(picked_idx)
    prefix = list(conversations[:max_idx + 1])
    body_start = min(max_idx + 1, n)
    return prefix, body_start


def _align_window_start(conversations: Sequence[dict], start: int, end: int) -> int:
    s = start
    while s <= end and conversations[s].get("from") not in ("human", "system"):
        s += 1
    return s if s <= end else start


def _align_window_end(conversations: Sequence[dict], start: int, end: int) -> int:
    e = end
    while e >= start and conversations[e].get("from") not in ("gpt", "tool"):
        e -= 1
    return e if e >= start else end


def _uniq_windows(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen = set()
    out = []
    for r in ranges:
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def build_windows_from_sample(sample: dict, config: PipelineConfig) -> list[TrajectoryWindow]:
    normalized = normalize_sample(sample)
    conversations = normalized.get("conversations") or []
    if not conversations:
        return []

    meta = normalized.get("metadata") or {}
    source_id = meta.get("source_id") or normalized.get("id")
    if not source_id:
        stable_payload = {
            "source_file": meta.get("source_file"),
            "semantic_input_seq": meta.get("semantic_input_seq"),
            "turn_count": len(conversations),
            "head": [
                {
                    "from": t.get("from"),
                    "value": (t.get("value") or "")[:96],
                }
                for t in conversations[:3]
            ],
        }
        stable_json = json.dumps(
            stable_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        source_id = f"sample_{hashlib.sha256(stable_json.encode('utf-8')).hexdigest()[:12]}"
    value_score = (normalized.get("value") or {}).get("value_score")
    source_file = meta.get("source_file")

    total_turns = len(conversations)
    threshold = config.semantic_long_turn_threshold
    if total_turns <= threshold:
        return [
            TrajectoryWindow(
                window_id=f"{source_id}_w1",
                source_id=source_id,
                window_index=1,
                total_windows=1,
                turn_start=0,
                turn_end=total_turns - 1,
                pinned_prefix_turns=0,
                conversations=conversations,
                pinned_prefix=[],
                body=conversations,
                value_score=value_score,
                source_file=source_file,
            )
        ]

    prefix, body_start = extract_pinned_prefix(
        conversations,
        max_turns=config.semantic_pinned_prefix_max_turns,
    )
    n = len(conversations)
    if body_start >= n:
        return [
            TrajectoryWindow(
                window_id=f"{source_id}_w1",
                source_id=source_id,
                window_index=1,
                total_windows=1,
                turn_start=0,
                turn_end=n - 1,
                pinned_prefix_turns=len(prefix),
                conversations=conversations,
                pinned_prefix=list(prefix),
                body=conversations,
                value_score=value_score,
                source_file=source_file,
            )
        ]
    window_size = config.semantic_window_size
    stride = config.semantic_window_stride

    ranges = []
    start = body_start
    while start < n:
        end = min(start + window_size - 1, n - 1)
        s = _align_window_start(conversations, start, end)
        e = _align_window_end(conversations, s, end)
        if s > e:
            break
        ranges.append((s, e))
        if e >= n - 1:
            break
        start = start + stride

    ranges = _uniq_windows(ranges)
    if not ranges:
        ranges = [(body_start, n - 1)]

    windows = []
    total_windows = len(ranges)
    for idx, (s, e) in enumerate(ranges, start=1):
        body = conversations[s:e + 1]
        merged = list(prefix) + list(body)
        windows.append(
            TrajectoryWindow(
                window_id=f"{source_id}_w{idx}",
                source_id=source_id,
                window_index=idx,
                total_windows=total_windows,
                turn_start=s,
                turn_end=e,
                pinned_prefix_turns=len(prefix),
                conversations=merged,
                pinned_prefix=list(prefix),
                body=body,
                value_score=value_score,
                source_file=source_file,
            )
        )
    return windows


def build_windows(samples: Sequence[dict], config: PipelineConfig) -> list[TrajectoryWindow]:
    windows = []
    for sample in samples:
        windows.extend(build_windows_from_sample(sample, config))
    return windows


def detect_language_mix(text: str) -> dict:
    zh = 0
    en = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            zh += 1
        elif ("a" <= ch.lower() <= "z"):
            en += 1
    total = zh + en
    if total == 0:
        return {"dominant": "unknown", "zh_ratio": 0.0, "en_ratio": 0.0}
    zh_ratio = zh / total
    en_ratio = en / total
    dominant = "zh" if zh_ratio > en_ratio else "en"
    if min(zh_ratio, en_ratio) > 0.2:
        dominant = "mixed"
    return {"dominant": dominant, "zh_ratio": round(zh_ratio, 4), "en_ratio": round(en_ratio, 4)}


def render_window_for_embedding(window: TrajectoryWindow) -> str:
    """Role-aware bilingual rendering for embedding."""
    lines = []
    for turn in window.conversations:
        role = turn.get("from", "unknown")
        role_label = ROLE_LABELS.get(role, role.upper())
        val = (turn.get("value") or "").strip()
        if not val:
            continue
        lines.append(f"[{role_label}] {val}")
    rendered = "\n".join(lines)
    mix = detect_language_mix(rendered)
    header = (
        f"[LANG={mix['dominant']} zh={mix['zh_ratio']:.2f} en={mix['en_ratio']:.2f}] "
        f"[WINDOW={window.window_index}/{window.total_windows}]"
    )
    return header + "\n" + rendered


def create_embedding_provider(config: PipelineConfig) -> EmbeddingProvider:
    provider = config.semantic_embedding_provider
    if provider == "api":
        return APIEmbeddingProvider(
            model=config.semantic_embedding_model,
            base_url=config.litellm_base,
            api_key=config.litellm_key,
            timeout=config.request_timeout,
        )
    return LocalHashEmbeddingProvider(
        model=config.semantic_embedding_model,
        dim=config.semantic_embedding_dim,
    )


def _chunked(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _embed_batch_local(provider: LocalHashEmbeddingProvider, texts: list[str], workers: int):
    if workers <= 1 or len(texts) < workers:
        return provider.embed_texts(texts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(provider._embed_one, t) for t in texts]
        return [f.result() for f in futures]


def embed_windows(
    windows: Sequence[TrajectoryWindow],
    config: PipelineConfig,
    progress_cb: SemanticProgressCb | None = None,
) -> list[WindowEmbedding]:
    provider = create_embedding_provider(config)
    batch_size = config.semantic_embedding_batch_size
    workers = config.semantic_embedding_max_workers

    out = []
    total = len(windows)
    processed = 0
    for window_batch in _chunked(windows, batch_size):
        rendered_batch = [render_window_for_embedding(w) for w in window_batch]
        if isinstance(provider, LocalHashEmbeddingProvider):
            vectors = _embed_batch_local(provider, rendered_batch, workers)
        else:
            vectors = provider.embed_texts(rendered_batch)
        if len(vectors) != len(window_batch):
            raise ValueError("Embedding vector count mismatch")
        for w, vec in zip(window_batch, vectors):
            out.append(
                WindowEmbedding(
                    window_id=w.window_id,
                    source_id=w.source_id,
                    dim=len(vec),
                    provider=provider.name,
                    model=provider.model,
                    vector=vec,
                )
            )
        processed += len(window_batch)
        if progress_cb:
            progress_cb("embed", "Embedding windows", processed, total)
    if len(out) != len(windows):
        raise ValueError("Embedding output count mismatch")
    return out


def build_hyperplanes(dim: int, bits: int, seed: int) -> list[list[float]]:
    rnd = random.Random(seed)
    planes = []
    for _ in range(bits):
        plane = [rnd.uniform(-1.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(v * v for v in plane))
        inv = 1.0 / norm if norm > 0 else 1.0
        planes.append([v * inv for v in plane])
    return planes


def project_semhash(vector: Sequence[float], planes: Sequence[Sequence[float]]) -> int:
    bits = 0
    for i, plane in enumerate(planes):
        dot = 0.0
        for a, b in zip(vector, plane):
            dot += a * b
        if dot >= 0:
            bits |= (1 << i)
    return bits


def _split_bands(hash_value: int, bits: int, bands: int) -> list[int]:
    width = bits // bands
    mask = (1 << width) - 1
    vals = []
    for i in range(bands):
        vals.append((hash_value >> (i * width)) & mask)
    return vals


def compute_semhash_records(
    embeddings: Sequence[WindowEmbedding],
    config: PipelineConfig,
) -> list[WindowSemHash]:
    if not embeddings:
        return []
    dim = embeddings[0].dim
    planes = build_hyperplanes(dim, config.semantic_semhash_bits, config.semantic_semhash_seed)
    width_hex = config.semantic_semhash_bits // 4
    recs = []
    for emb in embeddings:
        bits_val = project_semhash(emb.vector, planes)
        bits_hex = f"{bits_val:0{width_hex}x}"
        band_values = _split_bands(
            bits_val, config.semantic_semhash_bits, config.semantic_semhash_bands
        )
        recs.append(
            WindowSemHash(
                window_id=emb.window_id,
                source_id=emb.source_id,
                bits=bits_val,
                bits_hex=bits_hex,
                band_values=band_values,
            )
        )
    return recs


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _candidate_index(semhash_records: Sequence[WindowSemHash]) -> dict:
    index = defaultdict(list)
    for i, rec in enumerate(semhash_records):
        for band_i, val in enumerate(rec.band_values):
            index[(band_i, val)].append(i)
    return index


def build_neighbor_graph(
    embeddings: Sequence[WindowEmbedding],
    semhash_records: Sequence[WindowSemHash],
    config: PipelineConfig,
    progress_cb: SemanticProgressCb | None = None,
) -> tuple[dict, dict]:
    """Build ANN-refined neighbor graph.

    Returns:
        neighbors: {idx: [neighbor_idx...]}
        metrics: coarse/refined graph stats
    """
    band_index = _candidate_index(semhash_records)
    hash_vals = [r.bits for r in semhash_records]
    vectors = [e.vector for e in embeddings]
    neighbors = defaultdict(list)

    total_coarse_pairs = 0
    total_refined_links = 0
    radius = config.semantic_hamming_radius
    top_k = config.semantic_ann_top_k
    sim_threshold = config.semantic_ann_sim_threshold

    total = len(semhash_records)
    progress_step = max(total // 100, 1)

    for i, rec in enumerate(semhash_records):
        coarse = set()
        for band_i, val in enumerate(rec.band_values):
            coarse.update(band_index[(band_i, val)])
        coarse.discard(i)

        # Hamming-radius filter from coarse candidates.
        filtered = [j for j in coarse if _hamming(hash_vals[i], hash_vals[j]) <= radius]
        total_coarse_pairs += len(filtered)

        scored = []
        for j in filtered:
            sim = _dot(vectors[i], vectors[j])  # vectors are normalized
            if sim >= sim_threshold:
                scored.append((sim, j))
        refined = [j for _, j in heapq.nlargest(top_k, scored)]
        neighbors[i] = refined
        total_refined_links += len(refined)
        done = i + 1
        if progress_cb and (done == total or done % progress_step == 0):
            progress_cb("graph", "Refining ANN neighbor graph", done, total)

    metrics = {
        "coarse_pairs": total_coarse_pairs,
        "refined_links": total_refined_links,
        "avg_coarse_per_window": round(total_coarse_pairs / max(len(semhash_records), 1), 4),
        "avg_refined_per_window": round(total_refined_links / max(len(semhash_records), 1), 4),
    }
    return neighbors, metrics


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_clusters(neighbors: dict, windows: Sequence[TrajectoryWindow]) -> dict[str, list[int]]:
    uf = UnionFind(len(windows))
    for i, nbs in neighbors.items():
        for j in nbs:
            uf.union(i, j)

    roots = defaultdict(list)
    for i in range(len(windows)):
        roots[uf.find(i)].append(i)

    clusters = {}
    for idx, (_, members) in enumerate(sorted(roots.items(), key=lambda kv: min(kv[1])), start=1):
        cid = f"c{idx:07d}"
        clusters[cid] = sorted(members, key=lambda m: windows[m].window_id)
    return clusters


def compute_window_snr(window: TrajectoryWindow) -> tuple[float, int, int]:
    action = 0
    observation = 0
    for t in window.body:
        tokens = estimate_tokens(t.get("value", ""))
        role = t.get("from")
        if role == "gpt":
            action += tokens
        else:
            observation += tokens
    snr = action / max(observation, 1)
    return snr, action, observation


def select_representatives(
    clusters: dict[str, list[int]],
    windows: Sequence[TrajectoryWindow],
) -> list[ClusterMember]:
    members = []
    for cid, idxs in clusters.items():
        ranked = []
        for i in idxs:
            w = windows[i]
            snr, action, obs = compute_window_snr(w)
            ranked.append((snr, w.value_score if w.value_score is not None else -1e9, w.source_id, w.window_index, i, action, obs))
        ranked.sort(reverse=True)
        rep_idx = ranked[0][4] if ranked else None

        for _, _, _, _, i, action, obs in ranked:
            w = windows[i]
            snr = action / max(obs, 1)
            members.append(
                ClusterMember(
                    cluster_id=cid,
                    window_id=w.window_id,
                    source_id=w.source_id,
                    window_index=w.window_index,
                    snr=round(snr, 6),
                    action_tokens=action,
                    observation_tokens=obs,
                    value_score=w.value_score,
                    representative=(i == rep_idx),
                )
            )
    members.sort(key=lambda m: (m.cluster_id, m.window_index, m.window_id))
    return members


def compute_cluster_stats(
    windows: Sequence[TrajectoryWindow],
    clusters: dict[str, list[int]],
    semhash_records: Sequence[WindowSemHash],
    graph_metrics: dict,
) -> dict:
    sizes = [len(v) for v in clusters.values()]
    singleton = sum(1 for s in sizes if s == 1)
    bits = [r.bits for r in semhash_records]

    bit_density = 0.0
    if bits:
        total_one = sum(b.bit_count() for b in bits)
        bit_width = len(semhash_records[0].bits_hex) * 4 if semhash_records[0].bits_hex else 0
        if bit_width <= 0:
            bit_width = max(max(b.bit_length() for b in bits), 1)
        bit_density = total_one / (len(bits) * bit_width)
        bit_density = min(max(bit_density, 0.0), 1.0)

    sizes_sorted = sorted(sizes)
    p50 = sizes_sorted[len(sizes_sorted) // 2] if sizes_sorted else 0
    p90 = sizes_sorted[min(int(len(sizes_sorted) * 0.9), len(sizes_sorted) - 1)] if sizes_sorted else 0
    return {
        "total_windows": len(windows),
        "total_clusters": len(clusters),
        "singleton_clusters": singleton,
        "singleton_ratio": round(singleton / max(len(clusters), 1), 6),
        "cluster_size_mean": round(sum(sizes) / max(len(sizes), 1), 6),
        "cluster_size_p50": p50,
        "cluster_size_p90": p90,
        "semhash_bit_density": round(bit_density, 6),
        "graph": graph_metrics,
    }


def _json_default(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _write_jsonl(path: Path, rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def _compatibility_key(config: PipelineConfig) -> str:
    payload = {
        "version": config.semantic_manifest_version,
        "bits": config.semantic_semhash_bits,
        "seed": config.semantic_semhash_seed,
        "bands": config.semantic_semhash_bands,
        "hamming_radius": config.semantic_hamming_radius,
        "ann_top_k": config.semantic_ann_top_k,
        "ann_sim_threshold": config.semantic_ann_sim_threshold,
        "embedding_provider": config.semantic_embedding_provider,
        "embedding_model": config.semantic_embedding_model,
        "embedding_dim": config.semantic_embedding_dim,
        "window_size": config.semantic_window_size,
        "window_stride": config.semantic_window_stride,
        "long_turn_threshold": config.semantic_long_turn_threshold,
        "pinned_prefix_max_turns": config.semantic_pinned_prefix_max_turns,
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def save_manifest(path: Path, manifest: SemanticRunManifest):
    _write_json(path, asdict(manifest))


def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_manifest_compatible(existing_manifest: dict | None, config: PipelineConfig):
    if not existing_manifest:
        return
    existing_key = existing_manifest.get("compatibility_key")
    current_key = _compatibility_key(config)
    if existing_key and existing_key != current_key:
        raise ValueError(
            "Incompatible semantic clustering manifest: "
            f"existing={existing_key[:12]} current={current_key[:12]}"
        )


def export_representative_windows(
    windows: Sequence[TrajectoryWindow],
    members: Sequence[ClusterMember],
    output_path: str | Path,
):
    rep_ids = {m.window_id for m in members if m.representative}
    rows = []
    for w in windows:
        if w.window_id not in rep_ids:
            continue
        rows.append({
            "window_id": w.window_id,
            "source_id": w.source_id,
            "window_index": w.window_index,
            "total_windows": w.total_windows,
            "turn_start": w.turn_start,
            "turn_end": w.turn_end,
            "pinned_prefix_turns": w.pinned_prefix_turns,
            "value_score": w.value_score,
            "conversations": w.conversations,
            "source_file": w.source_file,
        })
    _write_jsonl(Path(output_path), rows)
    return len(rows)


def run_semantic_clustering(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    limit: int = 0,
    config: PipelineConfig | None = None,
    resume: bool = False,
    export_representatives: bool = True,
    progress_cb: SemanticProgressCb | None = None,
):
    """Run full-batch semantic clustering pipeline."""
    t0 = time.time()
    if config is None:
        config = PipelineConfig()
    validate_semantic_config(config)

    in_path = Path(input_path)
    out_dir = resolve_semantic_output_dir(
        in_path,
        output_dir,
        prefer_existing=resume,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / SEMANTIC_MANIFEST_FILE
    if resume:
        ensure_manifest_compatible(load_manifest(manifest_path), config)

    stage_seconds: dict[str, float] = {}
    if progress_cb:
        progress_cb("start", "Semantic clustering started", None, None)

    t_stage = time.time()
    samples = load_samples(in_path, limit=limit)
    stage_seconds["load"] = round(time.time() - t_stage, 3)
    if progress_cb:
        progress_cb("load", f"Loaded samples from {in_path}", len(samples), len(samples))
    if not samples:
        raise ValueError("No valid samples found for semantic clustering")

    t_stage = time.time()
    windows = build_windows(samples, config)
    stage_seconds["window"] = round(time.time() - t_stage, 3)
    if progress_cb:
        progress_cb("window", "Built trajectory windows", len(windows), len(windows))
    if not windows:
        raise ValueError("No semantic windows produced from input samples")

    t_stage = time.time()
    embeddings = embed_windows(windows, config, progress_cb=progress_cb)
    stage_seconds["embed"] = round(time.time() - t_stage, 3)

    t_stage = time.time()
    semhash_records = compute_semhash_records(embeddings, config)
    stage_seconds["hash"] = round(time.time() - t_stage, 3)
    if progress_cb:
        progress_cb("hash", "Projected SemHash signatures", len(semhash_records), len(semhash_records))

    t_stage = time.time()
    neighbors, graph_metrics = build_neighbor_graph(
        embeddings,
        semhash_records,
        config,
        progress_cb=progress_cb,
    )
    stage_seconds["graph"] = round(time.time() - t_stage, 3)

    t_stage = time.time()
    clusters = build_clusters(neighbors, windows)
    members = select_representatives(clusters, windows)
    stage_seconds["cluster"] = round(time.time() - t_stage, 3)
    if progress_cb:
        progress_cb("cluster", "Built clusters and selected representatives", len(clusters), len(clusters))

    stats = compute_cluster_stats(windows, clusters, semhash_records, graph_metrics)
    stats["elapsed_seconds"] = round(time.time() - t0, 3)
    stats["total_samples"] = len(samples)
    stats["stage_seconds"] = stage_seconds

    # Artifacts
    prefix = config.semantic_output_prefix
    windows_file = out_dir / f"{prefix}_windows.jsonl"
    emb_file = out_dir / f"{prefix}_embeddings.jsonl"
    hash_file = out_dir / f"{prefix}_semhash.jsonl"
    members_file = out_dir / f"{prefix}_cluster_membership.jsonl"
    clusters_file = out_dir / f"{prefix}_clusters.json"
    reps_file = out_dir / f"{prefix}_representatives.jsonl"
    stats_file = out_dir / SEMANTIC_STATS_FILE

    t_stage = time.time()
    _write_jsonl(windows_file, (asdict(w) for w in windows))
    if progress_cb:
        progress_cb("export", f"Wrote {windows_file.name}", 1, 5)
    _write_jsonl(emb_file, (asdict(e) for e in embeddings))
    if progress_cb:
        progress_cb("export", f"Wrote {emb_file.name}", 2, 5)
    _write_jsonl(hash_file, (asdict(h) for h in semhash_records))
    if progress_cb:
        progress_cb("export", f"Wrote {hash_file.name}", 3, 5)
    _write_jsonl(members_file, (asdict(m) for m in members))
    if progress_cb:
        progress_cb("export", f"Wrote {members_file.name}", 4, 5)
    _write_json(
        clusters_file,
        {"clusters": {cid: [windows[i].window_id for i in idxs] for cid, idxs in clusters.items()}},
    )
    if progress_cb:
        progress_cb("export", f"Wrote {clusters_file.name}", 5, 5)

    if export_representatives:
        rep_count = export_representative_windows(windows, members, reps_file)
        if progress_cb:
            progress_cb("export", f"Wrote {reps_file.name}", rep_count, rep_count)
    else:
        rep_count = 0
    stage_seconds["export"] = round(time.time() - t_stage, 3)

    manifest = SemanticRunManifest(
        version=config.semantic_manifest_version,
        created_at=_utc_now_iso(),
        parameters={
            "window": {
                "long_turn_threshold": config.semantic_long_turn_threshold,
                "window_size": config.semantic_window_size,
                "stride": config.semantic_window_stride,
                "pinned_prefix_max_turns": config.semantic_pinned_prefix_max_turns,
            },
            "embedding": {
                "provider": config.semantic_embedding_provider,
                "model": config.semantic_embedding_model,
                "dim": config.semantic_embedding_dim,
                "batch_size": config.semantic_embedding_batch_size,
                "max_workers": config.semantic_embedding_max_workers,
            },
            "semhash": {
                "bits": config.semantic_semhash_bits,
                "seed": config.semantic_semhash_seed,
                "bands": config.semantic_semhash_bands,
                "hamming_radius": config.semantic_hamming_radius,
            },
            "ann": {
                "top_k": config.semantic_ann_top_k,
                "sim_threshold": config.semantic_ann_sim_threshold,
            },
            "output_prefix": config.semantic_output_prefix,
        },
        compatibility_key=_compatibility_key(config),
        input_path=str(in_path),
        output_dir=str(out_dir),
        total_samples=len(samples),
        total_windows=len(windows),
        total_clusters=len(clusters),
    )
    save_manifest(manifest_path, manifest)

    artifacts = {
            "windows": str(windows_file),
            "embeddings": str(emb_file),
            "semhash": str(hash_file),
            "cluster_membership": str(members_file),
            "clusters": str(clusters_file),
            "manifest": str(manifest_path),
            "stats": str(stats_file),
        }
    if export_representatives:
        artifacts["representatives"] = str(reps_file)

    stats.update({
        "output_dir": str(out_dir),
        "artifacts": artifacts,
        "representative_count": rep_count,
    })
    _write_json(stats_file, stats)
    if progress_cb:
        progress_cb("done", "Semantic clustering complete", None, None)
    return stats


def format_semantic_summary(stats: dict) -> str:
    summary = (
        "Semantic clustering complete\n"
        f"Samples: {stats.get('total_samples', 0)}\n"
        f"Windows: {stats.get('total_windows', 0)}\n"
        f"Clusters: {stats.get('total_clusters', 0)}\n"
        f"Representatives: {stats.get('representative_count', 0)}\n"
        f"Elapsed: {stats.get('elapsed_seconds', 0)}s"
    )
    stage_seconds = stats.get("stage_seconds") or {}
    if stage_seconds:
        ordered = ["load", "window", "embed", "hash", "graph", "cluster", "export"]
        parts = [f"{k}={stage_seconds[k]}s" for k in ordered if k in stage_seconds]
        if parts:
            summary += "\nStages: " + ", ".join(parts)
    output_dir = stats.get("output_dir")
    if output_dir:
        summary += f"\nArtifacts: {output_dir}"
    return summary
