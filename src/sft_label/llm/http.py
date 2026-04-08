from __future__ import annotations

from pathlib import Path

import httpx

from sft_label.http_limits import estimate_pass1_extra_connections, resolve_httpx_connection_limits
from sft_label.config import DIR_PIPELINE_MAX_FILES
from sft_label.config import PipelineConfig


def _build_http_client_limits(
    concurrency: int,
    *,
    chunked_output_files: int = 0,
    pprint=print,
) -> httpx.Limits:
    extra_connections = estimate_pass1_extra_connections(
        chunked_output_files=chunked_output_files,
    )
    max_connections, max_keepalive, capped = resolve_httpx_connection_limits(
        requested_concurrency=concurrency,
        extra_connections=extra_connections,
    )
    if capped:
        pprint(
            "  [warn] lowering HTTP connection pool to "
            f"max_connections={max_connections}, keepalive={max_keepalive} "
            f"for requested concurrency={concurrency} to avoid FD exhaustion"
        )
    return httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive,
    )


def _chunked_output_fd_budget(*, is_directory: bool, input_path: Path | None, config: PipelineConfig | None) -> int:
    if is_directory:
        max_files = config.dir_pipeline_max_files if config else DIR_PIPELINE_MAX_FILES
        return max(int(max_files or 0), 0)
    if input_path is not None and str(input_path).endswith(".jsonl"):
        return 1
    return 0
