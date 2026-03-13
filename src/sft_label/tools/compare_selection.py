from __future__ import annotations

import copy
import json
from pathlib import Path

from sft_label.config import PipelineConfig
from sft_label.scoring import compute_selection_scores


def load_selection_regression_pack(path: str | Path):
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'Regression pack must be a JSON list: {path}')
    return data


def compare_selection_configs(samples, *, legacy_config=None, current_config=None):
    """Compare legacy vs current selection ordering on a fixture pack."""
    legacy_config = legacy_config or PipelineConfig(
        enable_value_stability=False,
        enable_selection_stability=False,
        enable_domain_backfill=False,
        selection_intra_weight=0.55,
        selection_quality_weight=0.20,
        selection_rarity_weight=0.25,
    )
    current_config = current_config or PipelineConfig()

    legacy_samples = copy.deepcopy(samples)
    current_samples = copy.deepcopy(samples)
    compute_selection_scores(legacy_samples, min_group_size=1, config=legacy_config)
    compute_selection_scores(current_samples, min_group_size=1, config=current_config)

    legacy_by_id = {sample['id']: sample['value']['selection_score'] for sample in legacy_samples}
    current_by_id = {sample['id']: sample['value']['selection_score'] for sample in current_samples}

    deltas = []
    for sample in current_samples:
        sample_id = sample['id']
        before = legacy_by_id.get(sample_id)
        after = current_by_id.get(sample_id)
        deltas.append({
            'id': sample_id,
            'label': sample.get('audit_label') or sample_id,
            'legacy_selection': before,
            'current_selection': after,
            'delta': round((after or 0.0) - (before or 0.0), 2),
            'value_score': (sample.get('value') or {}).get('value_score'),
        })

    def _ordered(rows, key):
        return [row['id'] for row in sorted(rows, key=lambda item: item[key], reverse=True)]

    legacy_order = _ordered(deltas, 'legacy_selection')
    current_order = _ordered(deltas, 'current_selection')
    top_n = min(3, len(deltas))
    bottom_n = min(3, len(deltas))
    return {
        'count': len(deltas),
        'deltas': sorted(deltas, key=lambda item: item['delta']),
        'legacy_order': legacy_order,
        'current_order': current_order,
        'top_bucket_shift': {
            'legacy_top': legacy_order[:top_n],
            'current_top': current_order[:top_n],
        },
        'bottom_bucket_shift': {
            'legacy_bottom': legacy_order[-bottom_n:],
            'current_bottom': current_order[-bottom_n:],
        },
        'threshold_counts': {
            'legacy_ge_8': sum(1 for row in deltas if (row['legacy_selection'] or 0) >= 8.0),
            'current_ge_8': sum(1 for row in deltas if (row['current_selection'] or 0) >= 8.0),
            'legacy_le_3': sum(1 for row in deltas if (row['legacy_selection'] or 10) <= 3.0),
            'current_le_3': sum(1 for row in deltas if (row['current_selection'] or 10) <= 3.0),
        },
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare legacy/new selection scoring on a regression pack.')
    parser.add_argument('fixture', help='Path to JSON regression pack')
    args = parser.parse_args()

    report = compare_selection_configs(load_selection_regression_pack(args.fixture))
    print(json.dumps(report, ensure_ascii=False, indent=2))
