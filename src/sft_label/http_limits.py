"""Helpers for deriving safe HTTP connection limits from process FD budgets."""

from __future__ import annotations


def _soft_nofile_limit() -> int | None:
    try:
        import resource
    except Exception:
        return None

    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return None

    if soft is None or soft <= 0:
        return None
    return int(soft)


def resolve_httpx_connection_limits(
    *,
    requested_concurrency: int,
    extra_connections: int = 10,
    reserve_fds: int = 64,
) -> tuple[int, int, bool]:
    """Return (max_connections, max_keepalive_connections, capped)."""
    concurrency = max(int(requested_concurrency or 0), 1)
    requested_max_connections = max(concurrency + max(int(extra_connections or 0), 0), 1)
    requested_keepalive = min(concurrency, requested_max_connections)

    soft_limit = _soft_nofile_limit()
    if soft_limit is None or soft_limit >= 1_000_000:
        return requested_max_connections, requested_keepalive, False

    available = max(soft_limit - max(int(reserve_fds or 0), 0), 1)
    max_connections = min(requested_max_connections, available)
    max_keepalive = min(requested_keepalive, max_connections)
    capped = max_connections < requested_max_connections or max_keepalive < requested_keepalive
    return max_connections, max_keepalive, capped
