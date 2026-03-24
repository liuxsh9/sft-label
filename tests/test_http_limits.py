import importlib


def test_resolve_httpx_connection_limits_caps_when_fd_limit_is_low(monkeypatch):
    http_limits = importlib.import_module("sft_label.http_limits")

    monkeypatch.setattr(http_limits, "_soft_nofile_limit", lambda: 256)

    max_connections, max_keepalive, capped = http_limits.resolve_httpx_connection_limits(
        requested_concurrency=300,
        extra_connections=10,
    )

    assert capped is True
    assert max_connections < 310
    assert max_keepalive <= max_connections


def test_resolve_httpx_connection_limits_keeps_requested_values_when_fd_limit_is_high(monkeypatch):
    http_limits = importlib.import_module("sft_label.http_limits")

    monkeypatch.setattr(http_limits, "_soft_nofile_limit", lambda: 1_000_000)

    max_connections, max_keepalive, capped = http_limits.resolve_httpx_connection_limits(
        requested_concurrency=300,
        extra_connections=10,
    )

    assert capped is False
    assert max_connections == 310
    assert max_keepalive == 300
