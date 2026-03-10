import json
from pathlib import Path

import pytest

from sft_label.config import PipelineConfig
from sft_label.semantic_clustering import (
    APIEmbeddingProvider,
    TrajectoryWindow,
    WindowEmbedding,
    WindowSemHash,
    _compatibility_key,
    build_windows_from_sample,
    compute_cluster_stats,
    compute_semhash_records,
    discover_input_files,
    extract_pinned_prefix,
    load_samples,
    run_semantic_clustering,
    select_representatives,
    validate_semantic_config,
)
from sft_label.tools.export_semantic_clusters import run_export_semantic_clusters


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_conversations(n_pairs=30):
    convs = [
        {"from": "system", "value": "You are a coding assistant."},
        {"from": "human", "value": "Build a helper script and explain it."},
        {"from": "gpt", "value": "I will implement step by step."},
    ]
    for i in range(n_pairs):
        convs.append({"from": "human", "value": f"Question {i}: optimize this flow"})
        convs.append({"from": "gpt", "value": f"Answer {i}: use chunking and caching"})
    return convs


def test_extract_pinned_prefix_basic():
    convs = [
        {"from": "system", "value": "S"},
        {"from": "human", "value": "H1"},
        {"from": "gpt", "value": "A1"},
        {"from": "human", "value": "H2"},
    ]
    prefix, body_start = extract_pinned_prefix(convs, max_turns=3)
    assert len(prefix) == 3
    assert prefix[0]["from"] == "system"
    assert prefix[1]["from"] == "human"
    assert prefix[2]["from"] == "gpt"
    assert body_start == 3


def test_extract_pinned_prefix_keeps_intermediate_turns():
    convs = [
        {"from": "system", "value": "S"},
        {"from": "human", "value": "H1"},
        {"from": "tool", "value": "OBS"},
        {"from": "gpt", "value": "A1"},
        {"from": "human", "value": "H2"},
    ]
    prefix, body_start = extract_pinned_prefix(convs, max_turns=3)
    assert [t["from"] for t in prefix] == ["system", "human", "tool", "gpt"]
    assert body_start == 4


def test_long_trajectory_windowing_with_metadata():
    sample = {
        "id": "sample_long",
        "conversations": _make_conversations(n_pairs=35),  # > 50 turns
        "metadata": {"source_file": "unit_test.json"},
    }
    cfg = PipelineConfig(
        semantic_long_turn_threshold=50,
        semantic_window_size=20,
        semantic_window_stride=12,
        semantic_pinned_prefix_max_turns=3,
    )
    windows = build_windows_from_sample(sample, cfg)
    assert len(windows) > 1
    for w in windows:
        assert w.source_id == "sample_long"
        assert w.total_windows == len(windows)
        assert w.turn_start <= w.turn_end
        assert w.pinned_prefix_turns <= 3
        # Pair alignment guarantees start at human/system and end at gpt/tool.
        assert w.body[0]["from"] in ("human", "system")
        assert w.body[-1]["from"] in ("gpt", "tool")


def test_source_id_prefers_metadata_source_id():
    sample = {
        "id": "slice_t2",
        "metadata": {"source_id": "source_conv"},
        "conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
        ],
    }
    windows = build_windows_from_sample(sample, PipelineConfig())
    assert windows[0].source_id == "source_conv"


def test_source_id_fallback_is_stable():
    sample = {
        "metadata": {"source_file": "x.json", "semantic_input_seq": 42},
        "conversations": [
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
        ],
    }
    w1 = build_windows_from_sample(sample, PipelineConfig())[0]
    w2 = build_windows_from_sample(sample, PipelineConfig())[0]
    assert w1.source_id == w2.source_id
    assert w1.source_id.startswith("sample_")


def test_discover_input_files_prefers_scored_and_jsonl(tmp_path: Path):
    row = {"id": "x", "conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}
    (tmp_path / "scored.json").write_text(json.dumps([row]), encoding="utf-8")
    (tmp_path / "scored.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "labeled.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    files = discover_input_files(tmp_path)
    assert [p.name for p in files] == ["scored.jsonl"]


def test_load_samples_avoids_scored_json_jsonl_double_count(tmp_path: Path):
    row = {"id": "x", "conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]}
    (tmp_path / "scored.json").write_text(json.dumps([row]), encoding="utf-8")
    (tmp_path / "scored.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_samples(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["metadata"]["semantic_input_seq"] == 1


def test_semhash_is_deterministic():
    cfg = PipelineConfig(semantic_semhash_bits=256, semantic_semhash_seed=123, semantic_semhash_bands=8)
    emb = [
        WindowEmbedding(
            window_id="w1",
            source_id="s1",
            dim=4,
            provider="local",
            model="hash",
            vector=[0.5, 0.5, 0.5, 0.5],
        )
    ]
    a = compute_semhash_records(emb, cfg)[0]
    b = compute_semhash_records(emb, cfg)[0]
    assert a.bits_hex == b.bits_hex
    assert a.band_values == b.band_values


def test_api_embedding_provider_respects_index_order(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 0.0]},
                ]
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("httpx.Client", _Client)
    provider = APIEmbeddingProvider(
        model="m",
        base_url="http://localhost/v1",
        api_key="k",
        timeout=10,
    )
    vectors = provider.embed_texts(["a", "b"])
    assert vectors[0] == [1.0, 0.0]
    assert vectors[1] == [0.0, 1.0]


def test_api_embedding_provider_rejects_inconsistent_dims(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                ]
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return _Resp()

    monkeypatch.setattr("httpx.Client", _Client)
    provider = APIEmbeddingProvider(
        model="m",
        base_url="http://localhost/v1",
        api_key="k",
        timeout=10,
    )
    with pytest.raises(ValueError, match="Inconsistent embedding dimensions"):
        provider.embed_texts(["a", "b"])


def test_representative_selection_snr_then_value_tiebreak():
    windows = [
        TrajectoryWindow(
            window_id="s1_w1",
            source_id="s1",
            window_index=1,
            total_windows=2,
            turn_start=0,
            turn_end=1,
            pinned_prefix_turns=0,
            conversations=[],
            pinned_prefix=[],
            body=[
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A" * 400},  # higher SNR
            ],
            value_score=5.0,
        ),
        TrajectoryWindow(
            window_id="s1_w2",
            source_id="s1",
            window_index=2,
            total_windows=2,
            turn_start=2,
            turn_end=3,
            pinned_prefix_turns=0,
            conversations=[],
            pinned_prefix=[],
            body=[
                {"from": "human", "value": "Q" * 200},
                {"from": "gpt", "value": "A" * 100},
            ],
            value_score=9.0,
        ),
    ]
    clusters = {"c0000001": [0, 1]}
    members = select_representatives(clusters, windows)
    reps = [m for m in members if m.representative]
    assert len(reps) == 1
    assert reps[0].window_id == "s1_w1"


def test_representative_selection_tie_break_by_value_then_source():
    windows = [
        TrajectoryWindow(
            window_id="b_w1",
            source_id="b",
            window_index=1,
            total_windows=2,
            turn_start=0,
            turn_end=1,
            pinned_prefix_turns=0,
            conversations=[],
            pinned_prefix=[],
            body=[{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}],
            value_score=8.0,
        ),
        TrajectoryWindow(
            window_id="a_w1",
            source_id="a",
            window_index=1,
            total_windows=2,
            turn_start=0,
            turn_end=1,
            pinned_prefix_turns=0,
            conversations=[],
            pinned_prefix=[],
            body=[{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}],
            value_score=9.0,
        ),
    ]
    members = select_representatives({"c0000001": [0, 1]}, windows)
    reps = [m for m in members if m.representative]
    assert len(reps) == 1
    assert reps[0].window_id == "a_w1"


def test_semhash_bit_density_is_bounded():
    windows = [
        TrajectoryWindow(
            window_id="w1",
            source_id="s",
            window_index=1,
            total_windows=1,
            turn_start=0,
            turn_end=0,
            pinned_prefix_turns=0,
            conversations=[],
            pinned_prefix=[],
            body=[],
        )
    ]
    semhash_records = [
        WindowSemHash(window_id="w1", source_id="s", bits=1, bits_hex="0" * 63 + "1", band_values=[]),
        WindowSemHash(window_id="w2", source_id="s", bits=(1 << 256) - 1, bits_hex="f" * 64, band_values=[]),
    ]
    stats = compute_cluster_stats(windows, {"c0000001": [0]}, semhash_records, {})
    assert 0.0 <= stats["semhash_bit_density"] <= 1.0


def test_validate_semantic_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="semantic_window_stride must be >= 1"):
        validate_semantic_config(PipelineConfig(semantic_window_stride=0))
    with pytest.raises(ValueError, match="semantic_semhash_bits must be divisible"):
        validate_semantic_config(PipelineConfig(semantic_semhash_bits=128, semantic_semhash_bands=7))


def test_compatibility_key_includes_pinned_prefix_turns():
    k1 = _compatibility_key(PipelineConfig(semantic_pinned_prefix_max_turns=1))
    k2 = _compatibility_key(PipelineConfig(semantic_pinned_prefix_max_turns=5))
    assert k1 != k2


def test_semantic_clustering_end_to_end(tmp_path: Path):
    fixture_path = FIXTURES_DIR / "smoke_semantic_clustering.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Add one long sample to exercise sliding window code path.
    samples.append(
        {
            "id": "s_long",
            "conversations": _make_conversations(n_pairs=8),
            "value": {"value_score": 7.1},
        }
    )
    in_file = tmp_path / "scored_semantic.json"
    with open(in_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    out_dir = tmp_path / "out"
    cfg = PipelineConfig(
        semantic_long_turn_threshold=10,
        semantic_window_size=8,
        semantic_window_stride=4,
        semantic_embedding_provider="local",
        semantic_embedding_dim=128,
        semantic_embedding_batch_size=16,
        semantic_embedding_max_workers=2,
        semantic_semhash_bits=128,
        semantic_semhash_bands=8,
        semantic_hamming_radius=32,
        semantic_ann_top_k=8,
        semantic_ann_sim_threshold=0.3,
    )

    stats = run_semantic_clustering(
        input_path=in_file,
        output_dir=out_dir,
        config=cfg,
        resume=False,
    )
    assert stats["total_samples"] == 4
    assert stats["total_windows"] >= 4
    assert (out_dir / "trajectory_windows.jsonl").exists()
    assert (out_dir / "trajectory_cluster_membership.jsonl").exists()
    assert (out_dir / "trajectory_representatives.jsonl").exists()
    assert (out_dir / "semantic_cluster_manifest.json").exists()

    # Resume with same config should pass compatibility checks.
    stats_resume = run_semantic_clustering(
        input_path=in_file,
        output_dir=out_dir,
        config=cfg,
        resume=True,
    )
    assert stats_resume["total_samples"] == 4

    # Resume with incompatible config should fail.
    bad_cfg = PipelineConfig(
        semantic_semhash_bits=256,  # changed from 128
        semantic_semhash_bands=8,
    )
    with pytest.raises(ValueError):
        run_semantic_clustering(
            input_path=in_file,
            output_dir=out_dir,
            config=bad_cfg,
            resume=True,
        )


def test_stats_file_matches_return_payload_when_no_representative_export(tmp_path: Path):
    samples = [
        {
            "id": "s1",
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ],
        }
    ]
    in_file = tmp_path / "scored_semantic.json"
    in_file.write_text(json.dumps(samples), encoding="utf-8")
    out_dir = tmp_path / "out"

    cfg = PipelineConfig(
        semantic_embedding_provider="local",
        semantic_embedding_dim=64,
        semantic_semhash_bits=64,
        semantic_semhash_bands=8,
        semantic_hamming_radius=4,
        semantic_ann_top_k=4,
        semantic_ann_sim_threshold=0.1,
    )
    stats = run_semantic_clustering(
        input_path=in_file,
        output_dir=out_dir,
        config=cfg,
        export_representatives=False,
    )
    on_disk = json.loads((out_dir / "semantic_cluster_stats.json").read_text(encoding="utf-8"))
    assert on_disk["representative_count"] == 0
    assert "artifacts" in on_disk
    assert "representatives" not in on_disk["artifacts"]
    assert on_disk["artifacts"] == stats["artifacts"]


def test_export_semantic_prefers_manifest_prefix_and_detects_mismatch(tmp_path: Path):
    manifest = {
        "parameters": {"output_prefix": "foo"},
    }
    (tmp_path / "semantic_cluster_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    # Correct prefix artifacts
    (tmp_path / "foo_windows.jsonl").write_text(
        json.dumps({"window_id": "w1"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "foo_cluster_membership.jsonl").write_text(
        json.dumps({"window_id": "w1", "cluster_id": "c1", "representative": True}) + "\n",
        encoding="utf-8",
    )
    # Distractor artifacts should be ignored
    (tmp_path / "bar_windows.jsonl").write_text(
        json.dumps({"window_id": "w2"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "bar_cluster_membership.jsonl").write_text(
        json.dumps({"window_id": "missing", "cluster_id": "c2", "representative": True}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "export.jsonl"
    count = run_export_semantic_clusters(tmp_path, out)
    assert count == 1

    # Corrupt membership references should fail loudly.
    (tmp_path / "foo_cluster_membership.jsonl").write_text(
        json.dumps({"window_id": "missing", "cluster_id": "c1", "representative": True}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing windows"):
        run_export_semantic_clusters(tmp_path, out)


def test_semantic_progress_callback_receives_stage_updates(tmp_path: Path):
    samples = [
        {
            "id": "s1",
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ],
        }
    ]
    in_file = tmp_path / "scored_semantic.json"
    in_file.write_text(json.dumps(samples), encoding="utf-8")
    out_dir = tmp_path / "out"

    events = []
    cfg = PipelineConfig(
        semantic_embedding_provider="local",
        semantic_embedding_dim=64,
        semantic_semhash_bits=64,
        semantic_semhash_bands=8,
        semantic_hamming_radius=4,
        semantic_ann_top_k=4,
        semantic_ann_sim_threshold=0.1,
    )
    run_semantic_clustering(
        input_path=in_file,
        output_dir=out_dir,
        config=cfg,
        export_representatives=False,
        progress_cb=lambda stage, message, current, total: events.append((stage, current, total)),
    )
    assert events
    assert events[0][0] == "start"
    assert events[-1][0] == "done"
    assert any(stage == "embed" for stage, _, _ in events)
    assert any(stage == "graph" and current == total for stage, current, total in events if current is not None)
