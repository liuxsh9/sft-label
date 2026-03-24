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


def estimate_pass1_extra_connections(
    *,
    chunked_output_files: int = 0,
    base_extra_connections: int = 10,
    per_chunked_file_output_handles: int = 4,
) -> int:
    """Estimate non-LLM FD overhead for Pass 1 chunked pipelines."""
    base = max(int(base_extra_connections or 0), 0)
    chunked_files = max(int(chunked_output_files or 0), 0)
    handles_per_file = max(int(per_chunked_file_output_handles or 0), 0)
    return base + chunked_files * handles_per_file


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
