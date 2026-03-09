import json
from pathlib import Path

import pytest

from sft_label.config import PipelineConfig
from sft_label.semantic_clustering import (
    TrajectoryWindow,
    WindowEmbedding,
    build_windows_from_sample,
    compute_semhash_records,
    extract_pinned_prefix,
    run_semantic_clustering,
    select_representatives,
)


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
