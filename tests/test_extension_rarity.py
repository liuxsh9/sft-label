import math

from sft_label.scoring import (
    compute_extension_field_idf,
    compute_extension_field_idf_map,
    compute_extension_field_rarity,
    compute_sample_extension_rarity,
    compute_extension_spec_rarity,
    normalize_extension_rarity_score,
)


def test_compute_extension_field_idf_clamps_non_negative():
    idf_map = compute_extension_field_idf({"common": 5, "rare": 1}, baseline_total=4)

    assert idf_map["common"] == 0.0
    assert idf_map["rare"] == math.log2(4 / 2)


def test_compute_extension_field_rarity_applies_confidence_shrinkage_to_prior():
    field_idfs = {"dashboard": 2.0, "form": 0.5}

    rarity = compute_extension_field_rarity("dashboard", field_idfs, confidence=0.25)

    expected_prior = (2.0 + 0.5) / 2
    expected = 0.25 * 2.0 + 0.75 * expected_prior
    assert rarity == expected


def test_compute_extension_field_rarity_multi_enum_uses_max_mean_blend():
    field_idfs = {"layout": 4.0, "chart": 1.0, "table": 1.0}

    rarity = compute_extension_field_rarity(
        ["layout", "chart", "table"],
        field_idfs,
        confidence=1.0,
    )

    assert rarity == 0.6 * 4.0 + 0.4 * 2.0


def test_normalize_extension_rarity_score_uses_spec_local_baseline_total():
    normalized = normalize_extension_rarity_score(raw_score=1.0, baseline_total=8, mode="absolute")

    assert normalized == 4.0


def test_compute_extension_spec_rarity_uses_baseline_distributions_and_uniform_fields():
    baseline = {
        "spec_version": "v1",
        "spec_hash": "sha256:ui-v1",
        "baseline_total": 8,
        "field_value_distributions": {
            "surface": {"dashboard": 1, "form": 4},
            "goal": {"mix-optimization": 2, "debugging": 3},
        },
        "field_presence_counts": {"surface": 8, "goal": 8},
        "config": {
            "schema": {
                "surface": {"type": "enum"},
                "goal": {"type": "enum"},
            }
        },
    }
    payload = {
        "status": "success",
        "matched": True,
        "spec_version": "v1",
        "spec_hash": "sha256:ui-v1",
        "labels": {
            "surface": "dashboard",
            "goal": "mix-optimization",
        },
        "confidence": {
            "surface": 1.0,
            "goal": 0.5,
        },
    }

    result = compute_extension_spec_rarity(payload, baseline, normalization_mode="absolute")

    field_idfs = compute_extension_field_idf_map(baseline["field_value_distributions"], baseline_total=8)
    surface_rarity = field_idfs["surface"]["dashboard"]
    goal_prior = sum(field_idfs["goal"].values()) / len(field_idfs["goal"])
    goal_rarity = 0.5 * field_idfs["goal"]["mix-optimization"] + 0.5 * goal_prior
    expected_raw = (surface_rarity + goal_rarity) / 2

    assert result["matched"] is True
    assert result["status"] == "success"
    assert result["baseline_total"] == 8
    assert result["raw_score"] == round(expected_raw, 4)
    assert result["score"] == normalize_extension_rarity_score(expected_raw, baseline_total=8, mode="absolute")
    assert result["confidence"] == 0.75


def test_compute_extension_spec_rarity_returns_none_for_non_success_payload():
    baseline = {
        "spec_version": "v1",
        "spec_hash": "sha256:ui-v1",
        "baseline_total": 8,
        "field_value_distributions": {"surface": {"dashboard": 1}},
        "field_presence_counts": {"surface": 8},
    }
    payload = {
        "status": "skipped",
        "matched": False,
        "spec_version": "v1",
        "spec_hash": "sha256:ui-v1",
        "labels": {"surface": "dashboard"},
    }

    result = compute_extension_spec_rarity(payload, baseline, normalization_mode="absolute")

    assert result["score"] is None
    assert result["raw_score"] is None


def test_compute_sample_extension_rarity_ignores_insufficient_support_specs():
    class Config:
        extension_rarity_mode = "preview"
        rarity_score_mode = "absolute"
        min_extension_baseline_total = 10

    sample = {
        "label_extensions": {
            "ui_small": {
                "status": "success",
                "matched": True,
                "spec_version": "v1",
                "spec_hash": "sha256:small",
                "labels": {"surface": "dashboard"},
                "confidence": {"surface": 0.95},
            },
            "ui_large": {
                "status": "success",
                "matched": True,
                "spec_version": "v1",
                "spec_hash": "sha256:large",
                "labels": {"surface": "form"},
                "confidence": {"surface": 0.8},
            },
        }
    }
    extension_stats = {
        "specs": {
            "ui_small": {
                "baselines": {
                    "sha256:small": {
                        "spec_version": "v1",
                        "spec_hash": "sha256:small",
                        "baseline_total": 2,
                        "field_value_distributions": {"surface": {"dashboard": 1, "form": 1}},
                        "field_presence_counts": {"surface": 2},
                    }
                }
            },
            "ui_large": {
                "baselines": {
                    "sha256:large": {
                        "spec_version": "v1",
                        "spec_hash": "sha256:large",
                        "baseline_total": 20,
                        "field_value_distributions": {"surface": {"dashboard": 19, "form": 1}},
                        "field_presence_counts": {"surface": 20},
                    }
                }
            },
        }
    }

    result = compute_sample_extension_rarity(sample, extension_stats, config=Config())

    assert result["specs"]["ui_small"]["support_sufficient"] is False
    assert result["specs"]["ui_large"]["support_sufficient"] is True
    assert result["support_sufficient"] is True
    assert result["score"] == result["specs"]["ui_large"]["score"]
