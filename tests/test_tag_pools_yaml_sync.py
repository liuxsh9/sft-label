"""Guard test: TAG_POOLS (runtime) and taxonomy YAML must stay in sync.

If this test fails, it means someone added a tag to prompts.py TAG_POOLS
but forgot to add the corresponding entry to taxonomy/tags/*.yaml (or vice versa).
"""
from __future__ import annotations


def test_tag_pools_and_yaml_are_in_sync():
    """Every tag in TAG_POOLS must exist in taxonomy YAML, and vice versa."""
    from sft_label.prompts import TAG_POOLS
    from sft_label._resources import load_all_tag_yamls

    yaml_tags = load_all_tag_yamls()
    yaml_by_dim: dict[str, set[str]] = {}
    for t in yaml_tags:
        yaml_by_dim.setdefault(t["category"].lower(), set()).add(t["id"])

    pool_by_dim = {dim: set(tags) for dim, tags in TAG_POOLS.items()}

    errors: list[str] = []
    all_dims = sorted(set(list(yaml_by_dim.keys()) + list(pool_by_dim.keys())))

    for dim in all_dims:
        yaml_ids = yaml_by_dim.get(dim, set())
        pool_ids = pool_by_dim.get(dim, set())
        only_yaml = sorted(yaml_ids - pool_ids)
        only_pool = sorted(pool_ids - yaml_ids)
        if only_yaml:
            errors.append(f"{dim}: in YAML but not TAG_POOLS: {only_yaml}")
        if only_pool:
            errors.append(f"{dim}: in TAG_POOLS but not YAML: {only_pool}")

    assert not errors, (
        "TAG_POOLS and taxonomy YAML are out of sync:\n  " + "\n  ".join(errors)
    )


def test_yaml_dimensions_match_tag_pools_dimensions():
    """The set of dimensions in YAML must match TAG_POOLS keys."""
    from sft_label.prompts import TAG_POOLS
    from sft_label._resources import load_all_tag_yamls

    yaml_dims = {t["category"].lower() for t in load_all_tag_yamls()}
    pool_dims = set(TAG_POOLS.keys())

    assert yaml_dims == pool_dims, (
        f"Dimension mismatch — YAML: {sorted(yaml_dims)}, TAG_POOLS: {sorted(pool_dims)}"
    )
