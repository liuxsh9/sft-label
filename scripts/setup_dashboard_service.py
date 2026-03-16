#!/usr/bin/env python3
"""Bootstrap a shared dashboard static service for sft-label."""

from __future__ import annotations

import argparse
from pathlib import Path

from sft_label.dashboard_service import (
    dashboard_service_status,
    init_dashboard_service,
    restart_dashboard_service,
    start_dashboard_service,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize and optionally start a shared sft-label dashboard static service.",
    )
    parser.add_argument("--name", default="default", help="Service name (default: default)")
    parser.add_argument(
        "--web-root",
        default=str((Path.home() / "sft-label-dashboard").resolve()),
        help="Shared web root for published dashboards",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument("--service-type", choices=["builtin", "pm2"], default="pm2",
                        help="Service backend type (default: pm2)")
    parser.add_argument("--public-base-url", default=None,
                        help="Public share URL base, e.g. https://dash.example.com")
    parser.add_argument("--pm2-name", default=None,
                        help="PM2 process name override")
    parser.add_argument("--start", action="store_true", help="Start the static service after init")
    parser.add_argument("--restart", action="store_true", help="Restart the static service after init")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    service = init_dashboard_service(
        name=args.name,
        web_root=args.web_root,
        host=args.host,
        port=args.port,
        service_type=args.service_type,
        public_base_url=args.public_base_url,
        pm2_name=args.pm2_name,
    )
    print(f"Initialized dashboard service '{service.name}'")
    print(f"  root: {service.web_root}")
    print(f"  url:  {service.base_url()}")
    print(f"  public: {service.share_base_url()}")
    print(f"  type: {service.service_type}")

    status = dashboard_service_status(service)
    if args.restart:
        status = restart_dashboard_service(service)
    elif args.start:
        status = start_dashboard_service(service)

    print(f"  state: {status['state']}")
    if service.service_type == "pm2":
        print("  note: run `pm2 startup` once on this machine if you need reboot persistence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
