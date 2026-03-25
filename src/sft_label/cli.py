"""
CLI entry point for sft-label.

Subcommands:
  sft-label start                — Interactive launcher for grouped workflows
  sft-label run                  — Run Pass 1 tag labeling (optionally + Pass 2)
  sft-label score                — Run Pass 2 value scoring on pre-labeled data
  sft-label refresh-rarity       — Refresh Pass 2 rarity/value fields (no LLM)
  sft-label semantic-cluster     — Run trajectory SemHash + ANN clustering
  sft-label export-semantic      — Export representative windows from clustering artifacts
  sft-label optimize-layout      — One-time optimize historical output layout
  sft-label filter               — Filter scored or inline-labeled data by value threshold
  sft-label analyze-unmapped     — Inspect out-of-pool tags from labeling output or stats
  sft-label validate             — Validate taxonomy definitions
  sft-label export-review        — Export labeled or inline-labeled data to review CSV
  sft-label recompute-stats      — Recompute stats from labeled/scored output (no LLM)
  sft-label regenerate-dashboard — Regenerate HTML dashboards from stats/data
  sft-label complete-postprocess — Materialize deferred Pass 2 postprocess offline

Most defaults come from config.py (PipelineConfig).
Selected runtime knobs (concurrency/rps/timeout/retries) can be overridden via CLI flags.
"""

import argparse
import asyncio
import json
import os
import shlex
import socket
import sys
import time
import random
from pathlib import Path
from dataclasses import dataclass

from sft_label.config import (
    DEFAULT_ROLLOUT_PRESET,
    DEFAULT_CONCURRENCY,
    DEFAULT_RPS_LIMIT,
    DEFAULT_RPS_WARMUP,
    ENABLE_ADAPTIVE_RUNTIME,
    ENABLE_STAGE_RECOVERY_SWEEP,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    ROLLOUT_PRESETS,
    apply_rollout_preset,
)
from sft_label.dashboard_service import (
    DashboardPortConflictError,
    dashboard_service_status,
    init_dashboard_service,
    load_dashboard_service_store,
    list_published_runs,
    publish_run_dashboards,
    restart_dashboard_service,
    set_default_dashboard_service,
    start_dashboard_service,
    stop_dashboard_service,
    update_dashboard_service_port,
)
from sft_label.label_extensions_guidance import (
    COMPACT_EXTENSION_RECOMMENDED_CHARS,
    summarize_extension_specs,
)
from sft_label.label_extensions_schema import load_extension_specs
from sft_label.progress_heartbeat import run_with_heartbeat


def _start_msg(lang: str, zh: str, en: str) -> str:
    return en if lang == "en" else zh


def _is_sensitive_env_key(key: str) -> bool:
    upper = key.upper()
    return any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD"))


def _mask_secret(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:2]}***{value[-2:]}"


def _format_start_env_lines(env_overrides: dict[str, str]) -> list[str]:
    lines: list[str] = []
    for key in sorted(env_overrides):
        value = env_overrides[key]
        shown = _mask_secret(value) if _is_sensitive_env_key(key) else value
        lines.append(f"  {key}={shown}")
    return lines


def _print_extension_preflight(specs, *, prompt_mode: str) -> None:
    if not specs:
        return
    summaries = summarize_extension_specs(specs)
    print("\nExtension preflight")
    print("-" * 56)
    print(
        "Current prompt mode: "
        f"{prompt_mode}. Compact-friendly guidance: keep prompt+schema <= {COMPACT_EXTENSION_RECOMMENDED_CHARS} chars per extension."
    )
    print("Recommended first run: start with one extension and validate on a small sample (limit=10/50).")
    for summary in summaries:
        status = "OK" if summary["compact_within_recommendation"] else "WARN"
        trigger = "yes" if summary["has_trigger"] else "no"
        name = summary["display_name"] or summary["id"]
        print(
            f"  - {name} ({summary['id']}) | fields={summary['field_count']} | options={summary['option_count']} "
            f"| trigger={trigger} | prompt={summary['prompt_chars']} chars | schema≈{summary['schema_chars']} chars "
            f"| prompt+schema≈{summary['prompt_schema_chars']} chars | compact<={COMPACT_EXTENSION_RECOMMENDED_CHARS}: {status}"
        )
        if "triggerless" in summary["warnings"]:
            print("    advisory: no trigger configured; this extension may run on a broader set of samples.")
        if "compact_over_budget" in summary["warnings"]:
            print("    advisory: consider shortening the prompt, reducing schema size, or splitting this spec.")
    print("-" * 56)


def _print_extension_followup(stats: dict | None, run_dir: str | None) -> None:
    extension_stats = (stats or {}).get("extension_stats") or {}
    specs = (extension_stats.get("specs") or {}) if isinstance(extension_stats, dict) else {}
    if not specs:
        return
    print("\nExtension follow-up")
    print("-" * 56)
    for ext_id, spec in sorted(specs.items()):
        total = int((spec or {}).get("total", 0) or 0)
        matched = int((spec or {}).get("matched", 0) or 0)
        status_counts = (spec or {}).get("status_counts") or {}
        statuses = ", ".join(
            f"{key} ({int(value or 0)})"
            for key, value in sorted(status_counts.items())
            if int(value or 0) > 0
        ) or "none"
        unmapped_total = sum(int(value or 0) for value in (((spec or {}).get("unmapped_counts") or {}).values()))
        print(f"  - {ext_id}: matched {matched} / {total} | statuses: {statuses} | unmapped={unmapped_total}")
        if total >= 5 and matched == 0:
            print("    advisory: trigger scope may be too narrow, or the core-label routing may not match the subset you intended.")
        elif total >= 5 and matched == total:
            print("    advisory: this extension matched every processed sample; verify the trigger is not too broad for this dataset.")
        if int(status_counts.get("failed", 0) or 0) > 0 or int(status_counts.get("invalid", 0) or 0) > 0:
            print("    advisory: inspect failed/invalid samples in the dashboard drawer before scaling up.")
        if unmapped_total > 0:
            print("    advisory: review unmapped values in the dashboard or export-review --include-extensions before widening the rollout.")
    if run_dir:
        print(f"Open the dashboard and inspect extension panels under: {run_dir}")
        print(f"Review export with extensions: uv run sft-label export-review --input {run_dir} --include-extensions")
    else:
        print("Open the generated dashboard and inspect extension panels, then export review with --include-extensions if needed.")
    print("First-run checklist: verify trigger scope, inspect 10-20 matched samples, and check low-confidence / unmapped values before scaling up.")
    print("-" * 56)


def _dashboard_start_msg(lang: str, zh: str, en: str) -> str:
    return en if lang == "en" else zh


def _resolve_dashboard_service(store, name: str | None = None):
    service_name = name or getattr(store, "default_service", None)
    if not service_name and getattr(store, "services", None):
        service_name = next(iter(store.services))
    if not service_name or service_name not in store.services:
        raise ValueError("No dashboard service configured. Run `sft-label dashboard-service init` first.")
    return store.services[service_name]


def _dashboard_service_prompt(
    lang: str,
    input_fn,
    prompt_zh: str,
    prompt_en: str,
) -> str:
    return input_fn(_dashboard_start_msg(lang, prompt_zh, prompt_en))


def _default_input(prompt: str) -> str:
    return input(prompt)


def _print_dashboard_port_conflict(error: DashboardPortConflictError, lang: str):
    owner = []
    if error.owner_pid:
        owner.append(f"PID {error.owner_pid}")
    if error.owner_command:
        owner.append(error.owner_command)
    owner_text = " | ".join(owner) if owner else _dashboard_start_msg(lang, "未知进程", "unknown process")
    print(
        _dashboard_start_msg(
            lang,
            f"Dashboard 端口冲突：{error.host}:{error.port} 已被占用（{owner_text}）",
            f"Dashboard port conflict: {error.host}:{error.port} is already in use ({owner_text})",
        )
    )


def _retry_dashboard_service_with_new_port(
    service,
    runner,
    *,
    lang: str,
    input_fn=None,
):
    prompt_fn = input_fn
    while True:
        try:
            return runner(service)
        except DashboardPortConflictError as error:
            _print_dashboard_port_conflict(error, lang)
            if prompt_fn is None:
                raise
            try:
                raw = _dashboard_service_prompt(
                    lang,
                    prompt_fn,
                    "请输入新的端口继续（Ctrl+C 取消）：",
                    "Enter a new port to continue (Ctrl+C to cancel): ",
                ).strip()
            except KeyboardInterrupt:
                print(_dashboard_start_msg(lang, "已取消 dashboard 服务启动。", "Cancelled dashboard service start."))
                raise SystemExit(130) from None
            if not raw:
                print(_dashboard_start_msg(lang, "端口不能为空，请重试。", "Port cannot be empty. Please try again."))
                continue
            try:
                new_port = int(raw)
            except ValueError:
                print(_dashboard_start_msg(lang, "端口必须是数字。", "Port must be a number."))
                continue
            if new_port <= 0 or new_port > 65535:
                print(_dashboard_start_msg(lang, "端口必须在 1-65535 之间。", "Port must be between 1 and 65535."))
                continue
            if new_port == service.port:
                print(_dashboard_start_msg(lang, "新端口不能与当前端口相同。", "New port must differ from the current port."))
                continue
            service = update_dashboard_service_port(service, new_port)
            print(
                _dashboard_start_msg(
                    lang,
                    f"已更新 dashboard 服务端口为 {service.port}，正在重试。",
                    f"Updated dashboard service port to {service.port}; retrying.",
                )
            )


def _guess_local_network_host() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = str(sock.getsockname()[0] or "").strip()
            if host and not host.startswith("127."):
                return host
    except OSError:
        pass

    try:
        host = str(socket.gethostbyname(socket.gethostname()) or "").strip()
    except OSError:
        return None
    if host and not host.startswith("127."):
        return host
    return None


def _suggest_dashboard_share_base_url(exposure: str, port: int) -> str | None:
    if exposure not in {"lan", "public"}:
        return None
    host = _guess_local_network_host()
    if not host:
        return None
    return f"http://{host}:{int(port)}"

@dataclass
class DashboardPublishPlan:
    enabled: bool
    service_name: str | None = None
    service_type: str | None = None
    state: str | None = None
    public_url: str | None = None
    start_now: bool = False
    bootstrap_kwargs: dict | None = None


def _collect_dashboard_publish_plan(launched_args, lang: str, input_fn) -> DashboardPublishPlan:
    if launched_args.command not in {"run", "score", "regenerate-dashboard"}:
        return DashboardPublishPlan(enabled=False)

    raw = _dashboard_service_prompt(
        lang,
        input_fn,
        "任务完成后自动发布 dashboard 到静态服务？ [Y/n]: ",
        "Auto-publish dashboards to the static service after completion? [Y/n]: ",
    ).strip().lower()
    if raw in ("n", "no"):
        return DashboardPublishPlan(enabled=False)

    store = load_dashboard_service_store()
    if not store.services:
        default_root = str((Path.home() / "sft-label-dashboard").resolve())
        name = (
            _dashboard_service_prompt(
                lang,
                input_fn,
                "服务名称（默认 default）：",
                "Service name [default default]: ",
            ).strip()
            or "default"
        )
        web_root = (
            _dashboard_service_prompt(
                lang,
                input_fn,
                f"静态目录（默认 {default_root}）：",
                f"Web root [{default_root}]: ",
            ).strip()
            or default_root
        )
        exposure_raw = (
            _dashboard_service_prompt(
                lang,
                input_fn,
                "访问方式 [1=本机, 2=局域网, 3=公网/反代]（默认 2）：",
                "Dashboard exposure [1=local, 2=LAN, 3=public/reverse-proxy] [default 2]: ",
            ).strip()
            or "2"
        )
        if exposure_raw in {"1", "local"}:
            exposure = "local"
            host = "127.0.0.1"
        elif exposure_raw in {"3", "public"}:
            exposure = "public"
            host = "0.0.0.0"
        else:
            exposure = "lan"
            host = "0.0.0.0"
        port_raw = (
            _dashboard_service_prompt(
                lang,
                input_fn,
                "端口（默认 8765）：",
                "Port [8765]: ",
            ).strip()
            or "8765"
        )
        suggested_share_url = _suggest_dashboard_share_base_url(exposure, int(port_raw))
        public_base_url = None
        if exposure != "local":
            prompt_suffix = (
                f" {suggested_share_url}"
                if suggested_share_url
                else _dashboard_start_msg(lang, " 自动探测失败，可手填", " auto-detect unavailable; enter manually")
            )
            public_base_url_raw = _dashboard_service_prompt(
                lang,
                input_fn,
                f"访问链接（回车使用默认；输入 - 表示不设置）。建议值：{prompt_suffix.strip()}：",
                f"Share/Public URL (Enter to use default; '-' to skip). Suggested:{prompt_suffix}: ",
            ).strip()
            if public_base_url_raw == "-":
                public_base_url = None
            elif public_base_url_raw:
                public_base_url = public_base_url_raw
            else:
                public_base_url = suggested_share_url
        start_now = _dashboard_service_prompt(
            lang,
            input_fn,
            "服务初始化后是否立即启动？ [Y/n]: ",
            "Start the service after initialization? [Y/n]: ",
        ).strip().lower()
        return DashboardPublishPlan(
            enabled=True,
            service_name=name,
            service_type="pm2",
            state="not-configured",
            public_url=public_base_url,
            start_now=start_now not in ("n", "no"),
            bootstrap_kwargs={
                "name": name,
                "web_root": web_root,
                "host": host,
                "port": int(port_raw),
                "service_type": "pm2",
                "public_base_url": public_base_url,
            },
        )

    service = _resolve_dashboard_service(store)
    status = dashboard_service_status(service)
    public_url = status.get("public_url") or service.share_base_url()
    start_now = False
    if status["state"] not in {"running", "starting"}:
        start_now = _dashboard_service_prompt(
            lang,
            input_fn,
            "服务未运行，是否现在启动？ [Y/n]: ",
            "Service is not running. Start it now? [Y/n]: ",
        ).strip().lower()
    return DashboardPublishPlan(
        enabled=True,
        service_name=service.name,
        service_type=service.service_type,
        state=status["state"],
        public_url=public_url,
        start_now=start_now not in ("n", "no"),
        bootstrap_kwargs=None,
    )


def _materialize_dashboard_publish_plan(dashboard_plan: DashboardPublishPlan | None, lang: str, input_fn):
    if dashboard_plan is None or not dashboard_plan.enabled:
        return None

    if dashboard_plan.bootstrap_kwargs:
        print(_dashboard_start_msg(
            lang,
            "当前没有已配置的 dashboard 静态服务，开始初始化默认服务。",
            "No dashboard service is configured yet. Initializing a default service.",
        ))
        service = init_dashboard_service(**dashboard_plan.bootstrap_kwargs)
    else:
        store = load_dashboard_service_store()
        service = _resolve_dashboard_service(store, dashboard_plan.service_name)

    status = dashboard_service_status(service)
    public_url = status.get("public_url") or service.share_base_url()
    print(
        _dashboard_start_msg(
            lang,
            f"Dashboard 服务：{service.name} | {service.service_type} | {status['state']} | {public_url}",
            f"Dashboard service: {service.name} | {service.service_type} | {status['state']} | {public_url}",
        )
    )

    if dashboard_plan.start_now and status["state"] not in {"running", "starting"}:
        status = _retry_dashboard_service_with_new_port(
            service,
            start_dashboard_service,
            lang=lang,
            input_fn=input_fn,
        )
    public_url = status.get("public_url") or service.share_base_url()
    print(
        _dashboard_start_msg(
            lang,
            f"Dashboard 服务状态：{status['state']} | {public_url}",
            f"Dashboard service status: {status['state']} | {public_url}",
        )
    )
    return service


def _extract_dashboard_publish_run_dir(launched_args, result):
    if isinstance(result, dict):
        run_dir = result.get("run_dir")
        if run_dir:
            return run_dir
    if launched_args.command == "regenerate-dashboard":
        return launched_args.input
    return None


def _score_postprocess_payload_for_publish_guard(launched_args, result):
    if launched_args.command == "run":
        if not getattr(launched_args, "score", False):
            return None
        summary = (result or {}).get("score_summary") if isinstance(result, dict) else None
    elif launched_args.command == "score":
        summary = result if isinstance(result, dict) else None
    else:
        return None

    if not isinstance(summary, dict):
        return {}
    postprocess = summary.get("postprocess")
    return postprocess if isinstance(postprocess, dict) else {}


def _load_regenerate_postprocess_payload(run_dir: str | os.PathLike[str] | None):
    if not run_dir:
        return None

    from sft_label.artifacts import PASS2_SUMMARY_STATS_FILE, PASS2_SUMMARY_STATS_FILE_LEGACY

    root = Path(run_dir)
    candidates = (
        root / PASS2_SUMMARY_STATS_FILE,
        root / PASS2_SUMMARY_STATS_FILE_LEGACY,
        root / "meta_label_data" / PASS2_SUMMARY_STATS_FILE,
        root / "meta_label_data" / PASS2_SUMMARY_STATS_FILE_LEGACY,
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        postprocess = payload.get("postprocess")
        return postprocess if isinstance(postprocess, dict) else {}
    return None


def _auto_publish_pass2_guard(
    launched_args,
    result,
    lang: str,
    run_dir: str | None = None,
) -> tuple[bool, str | None]:
    if launched_args.command == "regenerate-dashboard" and str(getattr(launched_args, "pass_num", "both")) == "1":
        return True, None

    postprocess = _score_postprocess_payload_for_publish_guard(launched_args, result)
    if postprocess is None and launched_args.command == "regenerate-dashboard":
        postprocess = _load_regenerate_postprocess_payload(run_dir or getattr(launched_args, "input", None))
    if postprocess is None:
        return True, None

    blocked_statuses = {"deferred", "pending", "failed"}
    safe_statuses = {"completed", "disabled"}
    required_keys = ("conversation_scores", "dashboard")
    observed: list[str] = []
    failures: list[str] = []
    has_scoring_dashboard = _run_dir_has_scoring_dashboard(run_dir)

    for key in required_keys:
        node = postprocess.get(key)
        status = node.get("status") if isinstance(node, dict) else None
        if not status:
            failures.append(f"{key}=missing")
            continue
        observed.append(f"{key}={status}")
        if status in blocked_statuses or status not in safe_statuses:
            failures.append(f"{key}={status}")
            continue
        if key == "dashboard" and status == "disabled" and has_scoring_dashboard:
            failures.append("dashboard=disabled(stale-scoring-dashboard)")

    if failures:
        observed_text = ", ".join(observed) if observed else "none"
        blocked_text = ", ".join(failures)
        run_dir_arg = str(run_dir) if run_dir else "<run_dir>"
        command_run_dir_arg = shlex.quote(run_dir_arg) if run_dir else run_dir_arg
        return (
            False,
            _start_msg(
                lang,
                f"Dashboard 自动发布已跳过（安全兜底）：Pass 2 后处理状态未完成（{blocked_text}）。当前状态：{observed_text}。请先运行 `sft-label complete-postprocess --input {command_run_dir_arg}`。",
                f"Dashboard auto-publish skipped (fail-closed): Pass 2 postprocess is incomplete ({blocked_text}). Current status: {observed_text}. Run `sft-label complete-postprocess --input {command_run_dir_arg}` first.",
            ),
        )

    return True, None


def _run_dir_has_scoring_dashboard(run_dir: str | None) -> bool:
    if not run_dir:
        return False
    root = Path(run_dir).expanduser().resolve()
    candidates = (
        root / "meta_label_data" / "dashboards",
        root / "dashboards",
        root,
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        html_files = sorted(candidate.glob("*.html"))
        if not html_files:
            continue
        return any(("scoring" in path.name.lower() or "value" in path.name.lower()) for path in html_files)
    return False


def _print_published_dashboard_urls(published, lang: str):
    dashboards = (published or {}).get("dashboards") or {}
    if not dashboards:
        return
    print(_dashboard_start_msg(lang, "\nDashboard 访问地址：", "\nDashboard URLs:"))
    for key in sorted(dashboards):
        url = dashboards[key].get("url")
        if url:
            print(f"  [{key}] {url}")


def _runtime_summary_values(launched_args) -> dict[str, object]:
    return {
        "concurrency": getattr(launched_args, "concurrency", None) or DEFAULT_CONCURRENCY,
        "rps_limit": getattr(launched_args, "rps_limit", None),
        "rps_warmup": getattr(launched_args, "rps_warmup", None) or DEFAULT_RPS_WARMUP,
        "request_timeout": getattr(launched_args, "request_timeout", None) or REQUEST_TIMEOUT,
        "max_retries": getattr(launched_args, "max_retries", None) or MAX_RETRIES,
        "adaptive_runtime": ENABLE_ADAPTIVE_RUNTIME if getattr(launched_args, "adaptive_runtime", None) is None else bool(getattr(launched_args, "adaptive_runtime")),
        "recovery_sweep": ENABLE_STAGE_RECOVERY_SWEEP if getattr(launched_args, "recovery_sweep", None) is None else bool(getattr(launched_args, "recovery_sweep")),
        "rollout_preset": getattr(launched_args, "rollout_preset", None) or DEFAULT_ROLLOUT_PRESET,
    }


def _print_start_plan_summary(plan, lang: str, *, launched_args=None, dashboard_plan: DashboardPublishPlan | None = None):
    from sft_label.launcher import format_command

    cmd_preview = format_command(plan.argv)
    print(f"\n{_start_msg(lang, '执行确认', 'Launch summary')}")
    print("-" * 56)
    if launched_args is not None:
        runtime = _runtime_summary_values(launched_args)
        print(_start_msg(lang, "执行总览：", "Execution overview:"))
        print(f"  {_start_msg(lang, '工作流', 'Workflow')}: {launched_args.command}")
        print(f"  {_start_msg(lang, '并发上限', 'Concurrency cap')}: {runtime['concurrency']}")
        rps_limit = runtime["rps_limit"]
        rps_shown = _start_msg(lang, "默认", "default") if rps_limit is None else rps_limit
        if rps_limit is None:
            rps_shown = DEFAULT_RPS_LIMIT
        print(f"  {_start_msg(lang, 'RPS 上限', 'RPS cap')}: {rps_shown}")
        print(f"  {_start_msg(lang, 'RPS 预热秒数', 'RPS warmup seconds')}: {runtime['rps_warmup']}")
        print(f"  {_start_msg(lang, '请求超时秒数', 'Request timeout seconds')}: {runtime['request_timeout']}")
        print(f"  {_start_msg(lang, '最大重试次数', 'Max retries')}: {runtime['max_retries']}")
        print(f"  {_start_msg(lang, '自适应运行时', 'Adaptive runtime')}: {'on' if runtime['adaptive_runtime'] else 'off'}")
        print(f"  {_start_msg(lang, '恢复补跑', 'Recovery sweep')}: {'on' if runtime['recovery_sweep'] else 'off'}")
        rollout_preset = runtime["rollout_preset"]
        preset_desc = (ROLLOUT_PRESETS.get(rollout_preset, {}) or {}).get("description", "")
        print(f"  {_start_msg(lang, '多轮优化预设', 'Multi-turn rollout preset')}: {rollout_preset}")
        if preset_desc:
            print(f"    {_start_msg(lang, '说明', 'Note')}: {preset_desc}")
        if dashboard_plan is not None:
            print(_start_msg(lang, "Dashboard 计划：", "Dashboard plan:"))
            if not dashboard_plan.enabled:
                print(f"  {_start_msg(lang, '自动发布', 'Auto-publish')}: no")
            else:
                print(f"  {_start_msg(lang, '自动发布', 'Auto-publish')}: yes")
                if dashboard_plan.service_name:
                    print(f"  {_start_msg(lang, '服务', 'Service')}: {dashboard_plan.service_name}")
                if dashboard_plan.service_type:
                    print(f"  {_start_msg(lang, '类型', 'Type')}: {dashboard_plan.service_type}")
                if dashboard_plan.state:
                    print(f"  {_start_msg(lang, '状态', 'State')}: {dashboard_plan.state}")
                if dashboard_plan.public_url:
                    print(f"  {_start_msg(lang, '访问地址', 'URL')}: {dashboard_plan.public_url}")
                print(f"  {_start_msg(lang, '确认后立即启动服务', 'Start service after confirm')}: {'yes' if dashboard_plan.start_now else 'no'}")
    print(_start_msg(lang, "命令：", "Command:"))
    print(f"  {cmd_preview}")
    if plan.env_overrides:
        print(_start_msg(lang, "环境变量覆盖（敏感值已脱敏）：", "Environment overrides (sensitive values masked):"))
        for line in _format_start_env_lines(plan.env_overrides):
            print(line)
    else:
        print(_start_msg(lang, "环境变量覆盖：无", "Environment overrides: none"))
    print("-" * 56)


def _parse_launch_plan_args(plan, parser, lang: str):
    try:
        launched_args = parser.parse_args(plan.argv)
    except SystemExit:
        print(_start_msg(
            lang,
            "生成参数解析失败，请重试 `sft-label start`。",
            "Generated arguments failed to parse. Re-run `sft-label start`.",
        ))
        return None

    if launched_args.command == "start":
        print(_start_msg(lang, "交互启动器不能再次启动自身。", "Interactive launcher cannot launch itself."))
        return None
    return launched_args


def _run_dashboard_service_maintenance_session(
    parser,
    lang: str,
    input_fn,
    build_dashboard_service_plan_fn,
    initial_plan,
):
    session_plan = initial_plan
    while True:
        for key, value in session_plan.env_overrides.items():
            os.environ[key] = value
        launched_args = _parse_launch_plan_args(session_plan, parser, lang)
        if launched_args is None:
            return None
        result = dispatch_command(launched_args, parser=parser)
        keep_going = _dashboard_service_prompt(
            lang,
            input_fn,
            "继续看板服务维护？ [Y/n]: ",
            "Continue dashboard service maintenance? [Y/n]: ",
        ).strip().lower()
        if keep_going in ("n", "no"):
            return result
        session_plan = build_dashboard_service_plan_fn(input_fn=input_fn, output_fn=print)
        _print_start_plan_summary(session_plan, lang)


def _apply_runtime_overrides(config, args):
    """Apply optional runtime overrides from CLI args onto PipelineConfig."""
    apply_rollout_preset(config, getattr(args, "rollout_preset", None))

    shared_concurrency = getattr(args, "concurrency", None)
    legacy_scoring_concurrency = getattr(args, "scoring_concurrency", None)
    if shared_concurrency is None and legacy_scoring_concurrency is not None:
        shared_concurrency = legacy_scoring_concurrency
    if shared_concurrency is not None:
        # One shared LLM concurrency setting for Pass 1 and Pass 2.
        config.concurrency = shared_concurrency
        config.scoring_concurrency = shared_concurrency

    for attr, arg_name in (
        ("rps_limit", "rps_limit"),
        ("rps_warmup", "rps_warmup"),
        ("request_timeout", "request_timeout"),
        ("max_retries", "max_retries"),
        ("rarity_score_mode", "rarity_mode"),
        ("extension_rarity_mode", "extension_rarity_mode"),
    ):
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, attr, val)
    if getattr(args, "adaptive_runtime", None) is not None:
        config.enable_adaptive_runtime = bool(args.adaptive_runtime)
    if getattr(args, "recovery_sweep", None) is not None:
        config.enable_stage_recovery_sweep = bool(args.recovery_sweep)


def _format_eta(seconds) -> str:
    if seconds is None:
        return "--:--"
    sec = max(int(seconds), 0)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class _CombinedLLMProgressTracker:
    """Track global LLM call progress across Pass 1 and Pass 2."""

    def __init__(self, planned_total_calls: int):
        self.planned_total_calls = max(int(planned_total_calls), 1)
        self.calls_done = 0
        self.pass1_calls = 0
        self.pass2_calls = 0
        self._start_time = time.time()
        self._smoothed_cps = 0.0

    def update(self, delta_calls: int, stage: str) -> str:
        delta = max(int(delta_calls or 0), 0)
        self.calls_done += delta
        if stage == "pass1":
            self.pass1_calls += delta
        elif stage == "pass2":
            self.pass2_calls += delta

        elapsed = max(time.time() - self._start_time, 1e-6)
        instant_cps = self.calls_done / elapsed
        if self._smoothed_cps <= 0:
            self._smoothed_cps = instant_cps
        else:
            self._smoothed_cps = 0.2 * instant_cps + 0.8 * self._smoothed_cps

        return self.eta_line()

    def eta_seconds(self):
        if self._smoothed_cps <= 0:
            return None
        remaining = max(self.planned_total_calls - self.calls_done, 0)
        return remaining / self._smoothed_cps

    def eta_line(self) -> str:
        cps = f"{self._smoothed_cps:.1f}/s" if self._smoothed_cps > 0 else "warming"
        eta = _format_eta(self.eta_seconds())
        return f"{self.progress_line()} eta {eta} rate {cps}"

    def summary_line(self) -> str:
        return f"{self.eta_line()} p1={self.pass1_calls} p2={self.pass2_calls}"

    def progress_line(self) -> str:
        pct = min(self.calls_done / max(self.planned_total_calls, 1) * 100, 100.0)
        return f"run {self.calls_done}/{self.planned_total_calls} ({pct:.1f}%)"


class _SemanticProgressPrinter:
    """Print semantic clustering progress in CLI-friendly format."""

    def __init__(self):
        self._last_percent: dict[str, int] = {}
        self._last_message: dict[str, str] = {}

    def __call__(self, stage: str, message: str, current=None, total=None):
        prefix = f"[semantic:{stage}]"
        if current is not None and total is not None and total > 0:
            percent = min(int((current / total) * 100), 100)
            last = self._last_percent.get(stage, -1)
            if percent <= last and current != total:
                return
            self._last_percent[stage] = percent
            print(f"{prefix} {message} {current}/{total} ({percent}%)")
            return

        last_msg = self._last_message.get(stage)
        if last_msg == message:
            return
        self._last_message[stage] = message
        print(f"{prefix} {message}")


def _estimate_pass2_calls(total_samples: int, sample_max_retries: int) -> int:
    retries = max(int(sample_max_retries), 1)
    retry_factor = 1.0 + min(0.2, 0.05 * (retries - 1))
    est = int(round(total_samples * retry_factor))
    return max(est, total_samples)


def _estimate_end_to_end_llm_calls(args, config):
    """Estimate Pass1+Pass2 total LLM calls for run --score mode."""
    if args.resume or not args.input or getattr(args, "mode", "refresh") == "recompute":
        return None

    input_path = Path(args.input)
    if not input_path.exists():
        return None

    from sft_label.pipeline import (
        discover_input_files,
        estimate_directory_workload,
        iter_samples_from_file,
        apply_sparse_sampling,
    )
    from sft_label.inline_pass1 import prepare_inline_pass1_batch
    from sft_label.inline_rows import iter_row_sample_bundles_from_jsonl
    from sft_label.inline_migration import load_inline_migration_index
    from sft_label.inline_scoring import infer_inline_scoring_target
    from sft_label.scoring import ScoringDirectoryWorkloadEstimate

    spec_count = len(config.extension_spec_paths or [])

    inline_target = infer_inline_scoring_target(input_path)
    if inline_target is not None and input_path.is_dir() and input_path == inline_target.layout.run_root:
        input_path = inline_target.layout.dataset_root

    migration_index = None
    if getattr(args, "mode", None) == "migrate" and getattr(args, "migrate_from", None):
        try:
            migration_index, _stats = load_inline_migration_index(args.migrate_from)
        except (FileNotFoundError, ValueError):
            return None

    pass1_labeled = 0
    pass1_calls = 0
    pass2_samples = 0
    pass1_workload_estimate = None
    pass2_workload_estimate = None

    if input_path.is_dir():
        files = discover_input_files(input_path)
        dir_files = [(a, r) for a, r in files if r is not None]
        if not dir_files:
            return None
        est = estimate_directory_workload(
            dir_files,
            limit=args.limit,
            shuffle=args.shuffle,
            config=config,
            enable_arbitration=not args.no_arbitration,
            mode=getattr(args, "mode", "refresh"),
            migration_index=migration_index,
        )
        pass1_calls = est.initial_estimated_llm_calls
        pass1_labeled = est.total_labeled_samples
        pass2_samples = est.total_samples
        pass1_workload_estimate = est
        pass2_workload_estimate = ScoringDirectoryWorkloadEstimate(
            files_planned=est.files_planned,
            total_samples=est.total_samples,
            baseline_total_llm_calls=est.total_samples,
            initial_estimated_llm_calls=_estimate_pass2_calls(est.total_samples, config.sample_max_retries),
            scan_elapsed_seconds=est.scan_elapsed_seconds,
        )
    else:
        sparse_kw = dict(
            full_label_count=config.sparse_full_label_count,
            gap_multiplier=config.sparse_gap_multiplier,
            min_gap=config.sparse_min_gap,
            max_gap=config.sparse_max_gap,
            threshold=config.sparse_threshold,
        )
        if str(input_path).endswith(".jsonl"):
            row_bundles = list(iter_row_sample_bundles_from_jsonl(input_path, limit=args.limit))
            prepared = prepare_inline_pass1_batch(
                row_bundles,
                mode=getattr(args, "mode", "refresh"),
                sparse_kwargs=sparse_kw,
                migration_index=migration_index,
            )
            pass1_labeled = len(prepared.label_indices)
            pass2_samples = len(prepared.samples)
        else:
            rng_state = random.getstate()
            try:
                samples, _ = iter_samples_from_file(
                    input_path,
                    limit=args.limit,
                    shuffle=args.shuffle,
                )
            finally:
                random.setstate(rng_state)
            label_indices, _ = apply_sparse_sampling(samples, **sparse_kw)
            pass1_labeled = len(label_indices)
            pass2_samples = len(samples)
        baseline = pass1_labeled * 2
        calls_per_sample = 2.2 if not args.no_arbitration else 2.0
        pass1_calls = max(int(round(pass1_labeled * calls_per_sample)), baseline)

    pass1_extension_calls = pass1_labeled * spec_count
    pass2_calls = _estimate_pass2_calls(pass2_samples, config.sample_max_retries)
    total_est_calls = pass1_calls + pass1_extension_calls + pass2_calls

    return {
        "pass1_labeled_samples": pass1_labeled,
        "pass2_samples": pass2_samples,
        "pass1_est_calls": pass1_calls,
        "pass2_est_calls": pass2_calls,
        "pass1_extension_calls": pass1_extension_calls,
        "total_est_calls": total_est_calls,
        "pass1_workload_estimate": pass1_workload_estimate,
        "pass2_workload_estimate": pass2_workload_estimate,
    }


def cmd_run(args):
    """Run the labeling pipeline."""
    if not args.input and not args.resume:
        print("Error: --input is required (unless using --resume)")
        sys.exit(1)
    if args.mode == "migrate" and not args.migrate_from and not args.resume:
        print("Error: --migrate-from is required when --mode migrate")
        sys.exit(1)
    if args.mode != "migrate" and args.migrate_from:
        print("Error: --migrate-from only applies to --mode migrate")
        sys.exit(1)
    if args.mode == "recompute" and args.resume:
        print("Error: --resume is not supported when --mode recompute")
        sys.exit(1)
    if args.mode == "recompute" and args.score:
        print("Error: --score cannot be combined with --mode recompute")
        sys.exit(1)
    if args.mode == "recompute" and args.semantic_cluster:
        print("Error: --semantic-cluster cannot be combined with --mode recompute")
        sys.exit(1)

    from sft_label.config import PipelineConfig
    from sft_label.pipeline import run

    config = PipelineConfig(prompt_mode=args.prompt_mode)
    _apply_runtime_overrides(config, args)
    if args.model:
        config.labeling_model = args.model
        config.scoring_model = args.model
    config.extension_spec_paths = args.label_extension or []
    if config.extension_spec_paths:
        config.extension_specs = load_extension_specs(config.extension_spec_paths)
    else:
        config.extension_specs = []
    setattr(config, "label_extension_specs", config.extension_specs)
    _print_extension_preflight(config.extension_specs, prompt_mode=args.prompt_mode)

    llm_progress_cb = None
    global_tracker = None
    precomputed_workload_estimate = None
    precomputed_scoring_workload_estimate = None
    if getattr(args, "score", False):
        plan = run_with_heartbeat(
            "Estimating workload",
            lambda: _estimate_end_to_end_llm_calls(args, config),
        )
        if plan:
            global_tracker = _CombinedLLMProgressTracker(plan["total_est_calls"])
            precomputed_workload_estimate = plan.get("pass1_workload_estimate")
            precomputed_scoring_workload_estimate = plan.get("pass2_workload_estimate")
            ext_calls = plan.get("pass1_extension_calls", 0)
            ext_segment = f" + ext_llm~{ext_calls}" if ext_calls else ""
            print(
                "Run plan | "
                f"total_llm~{plan['total_est_calls']} = "
                f"pass1_llm~{plan['pass1_est_calls']} ({plan['pass1_labeled_samples']} labeled){ext_segment} + "
                f"pass2_llm~{plan['pass2_est_calls']} ({plan['pass2_samples']} samples)"
            )
            llm_progress_cb = global_tracker.update
    try:
        stats = asyncio.run(run(
            input_path=args.input,
            output=args.output,
            resume=args.resume,
            limit=args.limit,
            shuffle=args.shuffle,
            enable_arbitration=not args.no_arbitration,
            config=config,
            mode=args.mode,
            migrate_from=args.migrate_from,
            llm_progress_cb=llm_progress_cb,
            precomputed_workload_estimate=precomputed_workload_estimate,
        ))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Continuous mode: run Pass 2 scoring after Pass 1
    score_summary = None
    if getattr(args, "score", False) and stats:
        run_dir = stats.get("run_dir")
        if run_dir:
            print("\n── Pass 2: Value Scoring ──\n")
            from sft_label.scoring import run_scoring
            score_summary = asyncio.run(run_scoring(
                input_path=run_dir,
                output_dir=run_dir,
                tag_stats_path=getattr(args, "tag_stats", None),
                config=config,
                resume=bool(args.resume) or args.mode in {"incremental", "migrate"},
                llm_progress_cb=llm_progress_cb,
                precomputed_workload_estimate=precomputed_scoring_workload_estimate,
            ))

    # Optional semantic clustering pass (trajectory-level)
    semantic_summary = None
    if getattr(args, "semantic_cluster", False) and stats:
        run_dir = stats.get("run_dir")
        if run_dir:
            print("\n── Trajectory Semantic Clustering ──\n")
            from sft_label.semantic_clustering import (
                run_semantic_clustering,
                format_semantic_summary,
            )
            semantic_progress_cb = _SemanticProgressPrinter()
            try:
                sc_stats = run_semantic_clustering(
                    input_path=run_dir,
                    output_dir=run_dir,
                    config=config,
                    export_representatives=True,
                    progress_cb=semantic_progress_cb,
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)
            print(format_semantic_summary(sc_stats))
            semantic_summary = sc_stats

    run_dir = None
    if isinstance(stats, dict):
        run_dir = stats.get("run_dir")
    if not run_dir and isinstance(score_summary, dict):
        run_dir = score_summary.get("run_dir")
    _print_extension_followup(stats if isinstance(stats, dict) else None, run_dir)
    return {
        "command": "run",
        "run_dir": run_dir,
        "stats": stats,
        "score_summary": score_summary,
        "semantic_summary": semantic_summary,
    }


def cmd_validate(args):
    """Validate taxonomy definitions."""
    from sft_label.validate import run_validation

    report = run_validation()
    report.print_report()

    if report.has_errors():
        print("\n❌ Validation failed with errors")
        sys.exit(1)
    else:
        print("\n✅ Validation passed")
        sys.exit(0)


def cmd_export_review(args):
    """Export labeled or inline-labeled data to review CSV."""
    from sft_label.tools.export_review import main as export_main
    # Patch sys.argv for the tool's own argparse
    patched = [
        "sft-label export-review",
        "--input", args.input,
        "--output", args.output,
    ]
    if args.monitor:
        patched.extend(["--monitor", args.monitor])
    if args.output_format:
        patched.extend(["--format", args.output_format])
    if getattr(args, "include_extensions", False):
        patched.append("--include-extensions")
    sys.argv = patched
    try:
        export_main()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_score(args):
    """Run value scoring (Pass 2) on pre-labeled data."""
    from sft_label.config import PipelineConfig
    from sft_label.scoring import run_scoring

    config = PipelineConfig(prompt_mode=args.prompt_mode)
    _apply_runtime_overrides(config, args)
    if args.model:
        config.scoring_model = args.model
    try:
        summary = asyncio.run(run_scoring(
            input_path=args.input,
            tag_stats_path=getattr(args, "tag_stats", None),
            limit=args.limit,
            config=config,
            resume=getattr(args, "resume", False),
        ))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    return summary


def cmd_filter(args):
    """Filter scored data by value threshold and other criteria."""
    from sft_label.tools.filter_value import run_filter, FilterConfig

    # Build FilterConfig from CLI args
    config = FilterConfig(
        value_min=args.value_min if args.value_min is not None else args.threshold,
        selection_min=args.selection_min,
        include_tags=[t.strip() for t in args.include_tags.split(",") if t.strip()] if args.include_tags else [],
        exclude_tags=[t.strip() for t in args.exclude_tags.split(",") if t.strip()] if args.exclude_tags else [],
        difficulty=[d.strip() for d in args.difficulty.split(",") if d.strip()] if args.difficulty else [],
        thinking_mode=args.thinking_mode,
        exclude_inherited=args.exclude_inherited,
        include_unscored=args.include_unscored,
        output_format=args.output_format,
        verify_source=args.verify_source,
        conv_value_min=args.conv_value_min,
        conv_selection_min=args.conv_selection_min,
        conv_value_v2_min=args.conv_value_v2_min,
        conv_selection_v2_min=args.conv_selection_v2_min,
        peak_complexity_min=args.peak_complexity_min,
        trajectory_structure_min=args.trajectory_structure_min,
        rarity_confidence_min=args.rarity_confidence_min,
        observed_turn_ratio_min=args.observed_turn_ratio_min,
        turn_count_min=args.turn_count_min,
        turn_count_max=args.turn_count_max,
        correctness_min=args.correctness_min,
        turn_value_min=args.turn_value_min,
        turn_quality_min=args.turn_quality_min,
        max_pruned_ratio=args.max_pruned_ratio,
        keep_first_last=not args.no_keep_first_last,
        preserve_structure=args.preserve_structure,
    )

    # Validate: at least one criterion must be set
    has_criterion = any([
        config.value_min is not None,
        config.selection_min is not None,
        config.include_tags,
        config.exclude_tags,
        config.difficulty,
        config.thinking_mode,
        config.exclude_inherited,
        config.verify_source,
        config.conv_value_min is not None,
        config.conv_selection_min is not None,
        config.conv_value_v2_min is not None,
        config.conv_selection_v2_min is not None,
        config.peak_complexity_min is not None,
        config.trajectory_structure_min is not None,
        config.rarity_confidence_min is not None,
        config.observed_turn_ratio_min is not None,
        config.turn_count_min is not None,
        config.turn_count_max is not None,
        config.correctness_min is not None,
        config.turn_value_min is not None,
        config.turn_quality_min is not None,
    ])
    if not has_criterion and not config.include_unscored:
        print("Error: At least one filter criterion is required "
              "(e.g. --value-min, --selection-min, --difficulty, etc.)")
        sys.exit(1)

    try:
        run_filter(
            input_path=args.input,
            output_path=args.output,
            config=config,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_analyze_unmapped(args):
    """Inspect unmapped tags from labeling outputs or labeling stats."""
    from sft_label.tools.analyze_unmapped import run_unmapped_analysis

    try:
        run_unmapped_analysis(
            input_path=args.input,
            top=args.top,
            examples=args.examples,
            dimension=args.dimension,
            stats_only=args.stats_only,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


def _apply_semantic_overrides(config, args):
    """Apply CLI semantic options onto PipelineConfig."""
    for key in (
        "semantic_long_turn_threshold",
        "semantic_window_size",
        "semantic_window_stride",
        "semantic_pinned_prefix_max_turns",
        "semantic_embedding_provider",
        "semantic_embedding_model",
        "semantic_embedding_dim",
        "semantic_embedding_batch_size",
        "semantic_embedding_max_workers",
        "semantic_semhash_bits",
        "semantic_semhash_seed",
        "semantic_semhash_bands",
        "semantic_hamming_radius",
        "semantic_ann_top_k",
        "semantic_ann_sim_threshold",
        "semantic_output_prefix",
    ):
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)


def cmd_semantic_cluster(args):
    """Run trajectory-level semantic clustering."""
    from sft_label.config import PipelineConfig
    from sft_label.semantic_clustering import (
        run_semantic_clustering,
        format_semantic_summary,
        validate_semantic_config,
    )

    config = PipelineConfig()
    _apply_semantic_overrides(config, args)

    try:
        validate_semantic_config(config)
        semantic_progress_cb = _SemanticProgressPrinter()
        stats = run_semantic_clustering(
            input_path=args.input,
            output_dir=args.output,
            limit=args.limit,
            config=config,
            resume=args.resume,
            export_representatives=not args.no_export_representatives,
            progress_cb=semantic_progress_cb,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(format_semantic_summary(stats))


def cmd_recompute_stats(args):
    """Recompute stats from labeled/scored pipeline output."""
    from sft_label.tools.recompute import run_recompute

    try:
        written = run_recompute(
            input_path=args.input,
            pass_num=args.pass_num,
            output_dir=args.output,
            workers=getattr(args, "workers", 8),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not written:
        print("No stats files generated. Check that input contains "
              "labeled/scored output.")
        sys.exit(1)
    print(f"\nDone. Wrote {len(written)} file(s).")


def cmd_refresh_rarity(args):
    """Refresh Pass 2 rarity/value fields from existing scored output."""
    from sft_label.config import PipelineConfig
    from sft_label.tools.recompute import run_refresh_rarity

    config = PipelineConfig()
    if getattr(args, "mode", None):
        config.rarity_score_mode = args.mode
    if getattr(args, "extension_rarity_mode", None):
        config.extension_rarity_mode = args.extension_rarity_mode

    try:
        written = run_refresh_rarity(
            input_path=args.input,
            tag_stats_path=getattr(args, "tag_stats", None),
            output_dir=args.output,
            config=config,
            workers=getattr(args, "workers", 8),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not written:
        print("No rarity updates generated. Check that input contains scored output.")
        sys.exit(1)
    print(f"\nDone. Refreshed rarity for {written.get('rarity_refreshed_samples', '0')} sample(s).")


def cmd_regenerate_dashboard(args):
    """Regenerate HTML dashboards from existing stats and data."""
    from sft_label.tools.recompute import run_regenerate_dashboard

    try:
        generated = run_regenerate_dashboard(
            input_path=args.input,
            pass_num=args.pass_num,
            open_browser=getattr(args, "open", False),
            workers=getattr(args, "workers", 8),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not generated:
        print("No dashboards generated. Check that input directory "
              "contains stats files.")
        sys.exit(1)
    print(f"\nDone. Generated {len(generated)} dashboard(s).")
    return {
        "command": "regenerate-dashboard",
        "run_dir": args.input,
        "generated": generated,
    }


def cmd_complete_postprocess(args):
    """Materialize deferred Pass 2 postprocess artifacts offline."""
    from sft_label.tools.recompute import run_complete_postprocess

    try:
        result = run_complete_postprocess(
            input_path=args.input,
            scope=getattr(args, "scope", "global"),
            open_browser=getattr(args, "open", False),
            workers=getattr(args, "workers", 8),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    dashboards = result.get("generated_dashboards") or []
    print(
        f"\nDone. Completed deferred postprocess for {result.get('files_processed', 0)} "
        f"file(s); generated {len(dashboards)} dashboard(s)."
    )
    return result


def cmd_export_semantic(args):
    """Export representative windows from semantic clustering artifacts."""
    from sft_label.tools.export_semantic_clusters import run_export_semantic_clusters

    try:
        count = run_export_semantic_clusters(
            input_dir=args.input,
            output_path=args.output,
            include_non_representative=args.include_all,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Exported {count} semantic window rows to {args.output}")


def cmd_optimize_layout(args):
    """One-time optimize historical output file layout."""
    from sft_label.tools.optimize_layout import (
        run_optimize_layout,
        format_optimize_layout_summary,
    )

    try:
        summary = run_optimize_layout(
            input_path=args.input,
            apply=getattr(args, "apply", False),
            prune_legacy=getattr(args, "prune_legacy", False),
            manifest_path=getattr(args, "manifest", None),
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(format_optimize_layout_summary(summary))
    if not getattr(args, "apply", False):
        print("\nDry-run only. Re-run with --apply to execute changes.")


def cmd_dashboard_service(args):
    """Manage dashboard static services and published runs."""
    action = getattr(args, "dashboard_service_action", None)
    if not action:
        print("Error: dashboard-service action is required")
        sys.exit(1)

    if action == "init":
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
        return {"service": service}

    store = load_dashboard_service_store()
    if action == "list":
        if not store.services:
            print("No dashboard services configured.")
            return []
        rows = []
        for name, service in sorted(store.services.items()):
            marker = "*" if name == store.default_service else " "
            status = dashboard_service_status(service)
            run_count = len(list_published_runs(service))
            print(
                f"{marker} {name} | {service.service_type} | {status['state']} | {status['public_url']} | "
                f"runs={run_count} | root={service.web_root}"
            )
            rows.append({
                "name": name,
                "service_type": service.service_type,
                "state": status["state"],
                "url": status["public_url"],
                "run_count": run_count,
                "web_root": service.web_root,
            })
        return rows

    if action == "set-default":
        updated = set_default_dashboard_service(args.name)
        default_name = updated.default_service if updated is not None else args.name
        print(f"Default dashboard service: {default_name}")
        return {"default_service": default_name}

    try:
        service = _resolve_dashboard_service(store, getattr(args, "name", None))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if action == "status":
        status = dashboard_service_status(service)
        print(
            f"{service.name} | {service.service_type} | {status['state']} | {status['public_url']} | root={service.web_root}"
        )
        return status
    if action == "start":
        try:
            status = _retry_dashboard_service_with_new_port(
                service,
                start_dashboard_service,
                lang="zh",
                input_fn=_default_input if sys.stdin.isatty() else None,
            )
        except DashboardPortConflictError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"{service.name} | {service.service_type} | {status['state']} | {status['public_url']}")
        return status
    if action == "restart":
        try:
            status = _retry_dashboard_service_with_new_port(
                service,
                restart_dashboard_service,
                lang="zh",
                input_fn=_default_input if sys.stdin.isatty() else None,
            )
        except DashboardPortConflictError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"{service.name} | {service.service_type} | {status['state']} | {status['public_url']}")
        return status
    if action == "stop":
        status = stop_dashboard_service(service)
        print(f"{service.name} | {service.service_type} | {status['state']} | {status['public_url']}")
        return status
    if action == "runs":
        runs = list_published_runs(service)
        if not runs:
            print(f"No published runs for service '{service.name}'.")
            return []
        for item in runs:
            dash = item.get("dashboards") or {}
            primary = (
                dash.get("scoring", {}).get("url")
                or dash.get("labeling", {}).get("url")
                or "-"
            )
            print(f"{item.get('run_id')} | {item.get('published_at', '-')} | {primary}")
        return runs
    if action == "register-run":
        published = publish_run_dashboards(service, args.run_dir)
        print(f"Published run: {published['run_id']}")
        for key in sorted((published.get("dashboards") or {}).keys()):
            url = published["dashboards"][key].get("url")
            if url:
                print(f"  [{key}] {url}")
        return published

    print(f"Error: unsupported dashboard-service action: {action}")
    sys.exit(1)


def cmd_start(args, parser):
    """Interactive launcher for grouped workflows."""
    from sft_label.launcher import (
        build_dashboard_service_plan,
        build_launch_plan,
        interactive_input,
        is_back_token,
        set_language,
        sanitize_prompt_input,
    )

    lang = getattr(args, "lang", "zh")
    if getattr(args, "en", False):
        lang = "en"
    set_language(lang)

    plan = build_launch_plan(input_fn=interactive_input, language=lang)
    if plan is None:
        print(_start_msg(lang, "已取消启动。", "Launch cancelled."))
        return

    if getattr(plan, "workflow_key", None) == "dashboard-service":
        _print_start_plan_summary(plan, lang)
        if args.dry_run:
            print(f"\n{_start_msg(lang, '仅预览模式，未执行命令。', 'Dry run mode: command not executed.')}")
            return
        confirm_raw = interactive_input(
            _start_msg(lang, "立即执行？ [Y/b/n]: ", "Execute now? [Y/b/n]: ")
        )
        confirm, had_control = sanitize_prompt_input(confirm_raw)
        if had_control:
            print(_start_msg(lang, "检测到方向键/控制字符输入，已忽略。", "Detected arrow/control key input and ignored."))
        if is_back_token(confirm):
            print(_start_msg(lang, "已取消执行。", "Launch aborted."))
            return
        confirm = confirm.lower()
        if confirm in ("n", "no"):
            print(_start_msg(lang, "已取消执行。", "Launch aborted."))
            return
        for key, value in plan.env_overrides.items():
            os.environ[key] = value
        return _run_dashboard_service_maintenance_session(
            parser=parser,
            lang=lang,
            input_fn=interactive_input,
            build_dashboard_service_plan_fn=build_dashboard_service_plan,
            initial_plan=plan,
        )

    launched_args = _parse_launch_plan_args(plan, parser, lang)
    if launched_args is None:
        return

    dashboard_plan = None
    if not args.dry_run:
        dashboard_plan = _collect_dashboard_publish_plan(
            launched_args,
            lang,
            interactive_input,
        )

    _print_start_plan_summary(plan, lang, launched_args=launched_args, dashboard_plan=dashboard_plan)

    if args.dry_run:
        print(f"\n{_start_msg(lang, '仅预览模式，未执行命令。', 'Dry run mode: command not executed.')}")
        return

    confirm_raw = interactive_input(
        _start_msg(lang, "立即执行？ [Y/b/n]: ", "Execute now? [Y/b/n]: ")
    )
    confirm, had_control = sanitize_prompt_input(confirm_raw)
    if had_control:
        print(_start_msg(lang, "检测到方向键/控制字符输入，已忽略。", "Detected arrow/control key input and ignored."))
    if is_back_token(confirm):
        print(_start_msg(lang, "已取消执行。", "Launch aborted."))
        return
    confirm = confirm.lower()
    if confirm in ("n", "no"):
        print(_start_msg(lang, "已取消执行。", "Launch aborted."))
        return

    for key, value in plan.env_overrides.items():
        os.environ[key] = value

    dashboard_service = _materialize_dashboard_publish_plan(
        dashboard_plan,
        lang,
        interactive_input,
    )
    result = dispatch_command(launched_args, parser=parser)
    if dashboard_service is not None:
        run_dir = _extract_dashboard_publish_run_dir(launched_args, result)
        if run_dir:
            can_publish, guard_message = _auto_publish_pass2_guard(
                launched_args,
                result,
                lang,
                run_dir=str(run_dir),
            )
            if not can_publish:
                print(guard_message)
            else:
                try:
                    published = run_with_heartbeat(
                        _start_msg(lang, "正在发布 dashboards", "Publishing dashboards"),
                        lambda: publish_run_dashboards(dashboard_service, run_dir),
                    )
                except (FileNotFoundError, ValueError) as e:
                    print(_start_msg(
                        lang,
                        f"Dashboard 自动发布失败：{e}",
                        f"Dashboard auto-publish failed: {e}",
                    ))
                else:
                    _print_published_dashboard_urls(published, lang)
        else:
            print(_start_msg(
                lang,
                "本次任务未返回可发布的 run 目录，跳过 dashboard 发布。",
                "No publishable run directory was returned; skipped dashboard publishing.",
                ))
    return result


def build_parser():
    """Build CLI parser for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="sft-label",
        description="SFT Capability Taxonomy & Auto-Labeling Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- start ---
    start_parser = subparsers.add_parser(
        "start",
        help="Interactive launcher for grouped workflows",
        description="Interactive task launcher that groups pipeline features "
                    "and builds a runnable command for you.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    start_parser.add_argument("--dry-run", action="store_true",
                              help="Only print generated command; do not execute it")
    start_parser.add_argument("--lang", choices=["zh", "en"], default="zh",
                              help="Language for interactive prompts (default: zh)")
    start_parser.add_argument("--en", action="store_true",
                              help=argparse.SUPPRESS)

    # --- dashboard-service ---
    dashboard_parser = subparsers.add_parser(
        "dashboard-service",
        help="Manage shared dashboard static services and published runs",
        description="Initialize/list/start/restart/stop shared dashboard static services, "
                    "and publish run dashboards into stable URLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dashboard_subparsers = dashboard_parser.add_subparsers(
        dest="dashboard_service_action",
        help="Dashboard service actions",
    )

    dashboard_init = dashboard_subparsers.add_parser("init", help="Initialize or update a dashboard service")
    dashboard_init.add_argument("--name", type=str, default="default",
                                help="Service name (default: default)")
    dashboard_init.add_argument("--web-root", type=str, required=True,
                                help="Shared web root for published dashboards")
    dashboard_init.add_argument("--host", type=str, default="127.0.0.1",
                                help="Host to bind (default: 127.0.0.1)")
    dashboard_init.add_argument("--port", type=int, default=8765,
                                help="Port to bind (default: 8765)")
    dashboard_init.add_argument("--service-type", choices=["builtin", "pm2"], default="pm2",
                                help="Service backend type (default: pm2)")
    dashboard_init.add_argument("--public-base-url", type=str, default=None,
                                help="Public share URL base, e.g. https://dash.example.com")
    dashboard_init.add_argument("--pm2-name", type=str, default=None,
                                help="PM2 process name override")

    dashboard_subparsers.add_parser("list", help="List configured dashboard services")

    dashboard_status = dashboard_subparsers.add_parser("status", help="Show dashboard service status")
    dashboard_status.add_argument("--name", type=str, default=None,
                                  help="Service name (default: configured default)")

    dashboard_start = dashboard_subparsers.add_parser("start", help="Start dashboard static service")
    dashboard_start.add_argument("--name", type=str, default=None,
                                 help="Service name (default: configured default)")

    dashboard_restart = dashboard_subparsers.add_parser("restart", help="Restart dashboard static service")
    dashboard_restart.add_argument("--name", type=str, default=None,
                                   help="Service name (default: configured default)")

    dashboard_stop = dashboard_subparsers.add_parser("stop", help="Stop dashboard static service")
    dashboard_stop.add_argument("--name", type=str, default=None,
                                help="Service name (default: configured default)")

    dashboard_runs = dashboard_subparsers.add_parser("runs", help="List published runs for a service")
    dashboard_runs.add_argument("--name", type=str, default=None,
                                help="Service name (default: configured default)")

    dashboard_default = dashboard_subparsers.add_parser("set-default", help="Set the default dashboard service")
    dashboard_default.add_argument("--name", type=str, required=True,
                                   help="Service name to mark as default")

    dashboard_register = dashboard_subparsers.add_parser("register-run", help="Publish an existing run directory")
    dashboard_register.add_argument("--name", type=str, default=None,
                                    help="Service name (default: configured default)")
    dashboard_register.add_argument("--run-dir", type=str, required=True,
                                    help="Absolute or relative run directory to publish")

    # --- run ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run Pass 1 tag labeling pipeline",
        description="Run the tag labeling pipeline (Pass 1). "
                    "Optionally chain Pass 2 value scoring with --score.",
        epilog="Model and concurrency are configured in config.py "
               "(DEFAULT_LABELING_MODEL, DEFAULT_CONCURRENCY). "
               "Override via PipelineConfig in library mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument("--input", type=str, default=None,
                            help="Input file (.json/.jsonl) or directory")
    run_parser.add_argument("--output", type=str, default=None,
                            help="Output directory (default: sibling of input)")
    run_parser.add_argument("--resume", type=str, default=None,
                            help="Resume from an existing run directory")
    run_parser.add_argument("--mode", type=str, default="refresh",
                            choices=["incremental", "refresh", "migrate", "recompute"],
                            help="Inline dataset mode: incremental, refresh, migrate, or recompute")
    run_parser.add_argument("--migrate-from", type=str, default=None,
                            help="Inline-labeled source dataset/run to copy data_label from (used with --mode migrate)")
    run_parser.add_argument("--limit", type=int, default=0,
                            help="Max samples per file, 0 = all (default: 0)")
    run_parser.add_argument("--shuffle", action="store_true",
                            help="Randomly shuffle samples before processing")
    run_parser.add_argument("--no-arbitration", action="store_true",
                            help="Disable low-confidence arbitration pass")
    run_parser.add_argument("--score", action="store_true",
                            help="Chain Pass 2 value scoring after labeling")
    run_parser.add_argument("--semantic-cluster", action="store_true",
                            help="Chain trajectory semantic clustering after labeling/scoring")
    run_parser.add_argument("--tag-stats", type=str, default=None,
                            help="Path to stats.json for Pass 2 rarity (used with --score)")
    run_parser.add_argument("--rarity-mode", type=str, default=None,
                            choices=["absolute", "percentile"],
                            help="Rarity normalization mode for Pass 2 (used with --score)")
    run_parser.add_argument("--extension-rarity-mode", type=str, default=None,
                            choices=["off", "preview", "bonus_only"],
                            help="Extension Rarity V2 mode for Pass 2: off (default, legacy unchanged), preview (compute extension rarity diagnostics only), or bonus_only (experimental V2 bonus fields only).")
    run_parser.add_argument("--prompt-mode", type=str, choices=["full", "compact"],
                            default="compact",
                            help="Prompt mode: 'full' or 'compact' (default: compact). Compact uses smaller payloads; for extension labeling, keep each prompt+schema compact-friendly when possible.")
    run_parser.add_argument("--rollout-preset", type=str,
                            choices=sorted(ROLLOUT_PRESETS.keys()),
                            default=DEFAULT_ROLLOUT_PRESET,
                            help="Multi-turn rollout preset: compact_safe (default recommended), planner_hybrid (experimental), or baseline_control.")
    run_parser.add_argument("--model", type=str, default=None,
                            help="LLM model name (default: gpt-4o-mini)")
    run_parser.add_argument("--concurrency", type=int, default=None,
                            help="Override pass1 labeling concurrency")
    run_parser.add_argument("--scoring-concurrency", type=int, default=None,
                            help=argparse.SUPPRESS)
    run_parser.add_argument("--rps-limit", type=float, default=None,
                            help="Override global LLM requests/sec limit")
    run_parser.add_argument("--rps-warmup", type=float, default=None,
                            help="Override RPS warmup seconds")
    run_parser.add_argument("--request-timeout", type=int, default=None,
                            help="Override per-request timeout seconds")
    run_parser.add_argument("--max-retries", type=int, default=None,
                            help="Override max retries per request")
    run_parser.add_argument("--adaptive-runtime", action=argparse.BooleanOptionalAction,
                            default=None,
                            help="Enable adaptive LLM pressure control (default: on)")
    run_parser.add_argument("--recovery-sweep", action=argparse.BooleanOptionalAction,
                            default=None,
                            help="Retry infra-failed samples once before finalizing each stage (default: on)")
    run_parser.add_argument("--label-extension", action="append", default=None,
                            help="Path to a label extension spec (YAML/JSON). Repeatable; each spec should use a unique id and multiple specs run independently.")

    # --- validate ---
    subparsers.add_parser("validate", help="Validate taxonomy definitions")

    # --- score ---
    score_parser = subparsers.add_parser(
        "score",
        help="Run Pass 2 value scoring on pre-labeled data",
        description="Run value scoring (Pass 2) on pre-labeled data. "
                    "Computes complexity, quality, reasoning, and rarity scores.",
        epilog="Model and concurrency are configured in config.py "
               "(DEFAULT_SCORING_MODEL, DEFAULT_SCORING_CONCURRENCY). "
               "Override via PipelineConfig in library mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    score_parser.add_argument("--input", type=str, required=True,
                               help="Path to labeled.json or directory of labeled files")
    score_parser.add_argument("--tag-stats", type=str, default=None,
                               help="Path to stats.json for rarity (default: auto-discover)")
    score_parser.add_argument("--rarity-mode", type=str, default=None,
                               choices=["absolute", "percentile"],
                               help="Rarity normalization mode override")
    score_parser.add_argument("--extension-rarity-mode", type=str, default=None,
                               choices=["off", "preview", "bonus_only"],
                               help="Extension Rarity V2 mode: off (default, legacy unchanged), preview (compute extension rarity diagnostics only), or bonus_only (experimental V2 bonus fields only).")
    score_parser.add_argument("--limit", type=int, default=0,
                               help="Max samples to score, 0 = all (default: 0)")
    score_parser.add_argument("--resume", action="store_true",
                               help="Resume scoring: skip samples already in scored.jsonl")
    score_parser.add_argument("--prompt-mode", type=str, choices=["full", "compact"],
                               default="compact",
                               help="Prompt mode: 'full' or 'compact' (default: compact). Compact uses smaller payloads for size-limited endpoints.")
    score_parser.add_argument("--rollout-preset", type=str,
                               choices=sorted(ROLLOUT_PRESETS.keys()),
                               default=DEFAULT_ROLLOUT_PRESET,
                               help="Multi-turn rollout preset: compact_safe (default recommended), planner_hybrid (experimental), or baseline_control.")
    score_parser.add_argument("--model", type=str, default=None,
                               help="LLM model name (default: gpt-4o-mini)")
    score_parser.add_argument("--concurrency", type=int, default=None,
                               help="Override shared LLM concurrency (used by pass1/pass2)")
    score_parser.add_argument("--scoring-concurrency", type=int, default=None,
                               help=argparse.SUPPRESS)
    score_parser.add_argument("--rps-limit", type=float, default=None,
                               help="Override global LLM requests/sec limit")
    score_parser.add_argument("--rps-warmup", type=float, default=None,
                               help="Override RPS warmup seconds")
    score_parser.add_argument("--request-timeout", type=int, default=None,
                               help="Override per-request timeout seconds")
    score_parser.add_argument("--max-retries", type=int, default=None,
                               help="Override max retries per request")
    score_parser.add_argument("--adaptive-runtime", action=argparse.BooleanOptionalAction,
                               default=None,
                               help="Enable adaptive LLM pressure control (default: on)")
    score_parser.add_argument("--recovery-sweep", action=argparse.BooleanOptionalAction,
                               default=None,
                               help="Retry infra-failed samples once before finalizing scoring (default: on)")

    # --- semantic-cluster ---
    semantic_parser = subparsers.add_parser(
        "semantic-cluster",
        help="Run trajectory SemHash + ANN clustering",
        description="Segment long trajectories with pinned prefix, compute semantic embeddings, "
                    "build SemHash + ANN clusters, and select representative windows by SNR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    semantic_parser.add_argument("--input", type=str, required=True,
                                 help="Input file/dir (prefers scored/labeled outputs)")
    semantic_parser.add_argument("--output", type=str, default=None,
                                 help="Output directory (default: input dir)")
    semantic_parser.add_argument("--limit", type=int, default=0,
                                 help="Max samples to process, 0 = all")
    semantic_parser.add_argument("--resume", action="store_true",
                                 help="Check compatibility against existing manifest before run")
    semantic_parser.add_argument("--no-export-representatives", action="store_true",
                                 help="Do not write representative windows output file")
    semantic_parser.add_argument("--semantic-long-turn-threshold", type=int, default=None,
                                 help="Only trajectories above this turn count use sliding windows")
    semantic_parser.add_argument("--semantic-window-size", type=int, default=None,
                                 help="Window body size in turns")
    semantic_parser.add_argument("--semantic-window-stride", type=int, default=None,
                                 help="Sliding stride in turns")
    semantic_parser.add_argument("--semantic-pinned-prefix-max-turns", type=int, default=None,
                                 help="Max turns kept in pinned task-definition prefix")
    semantic_parser.add_argument("--semantic-embedding-provider", choices=["local", "api"], default=None,
                                 help="Embedding backend: local or api")
    semantic_parser.add_argument("--semantic-embedding-model", type=str, default=None,
                                 help="Embedding model id")
    semantic_parser.add_argument("--semantic-embedding-dim", type=int, default=None,
                                 help="Embedding vector dimension (local backend)")
    semantic_parser.add_argument("--semantic-embedding-batch-size", type=int, default=None,
                                 help="Embedding batch size")
    semantic_parser.add_argument("--semantic-embedding-max-workers", type=int, default=None,
                                 help="Embedding worker count for local backend")
    semantic_parser.add_argument("--semantic-semhash-bits", type=int, default=None,
                                 help="SemHash bit width (64/128/256/512)")
    semantic_parser.add_argument("--semantic-semhash-seed", type=int, default=None,
                                 help="SemHash random seed")
    semantic_parser.add_argument("--semantic-semhash-bands", type=int, default=None,
                                 help="SemHash band count")
    semantic_parser.add_argument("--semantic-hamming-radius", type=int, default=None,
                                 help="Hamming radius for coarse candidate filtering")
    semantic_parser.add_argument("--semantic-ann-top-k", type=int, default=None,
                                 help="Max refined neighbors per window")
    semantic_parser.add_argument("--semantic-ann-sim-threshold", type=float, default=None,
                                 help="Cosine similarity threshold for refined links")
    semantic_parser.add_argument("--semantic-output-prefix", type=str, default=None,
                                 help="Artifact filename prefix")

    # --- export-semantic ---
    export_sem_parser = subparsers.add_parser(
        "export-semantic",
        help="Export semantic clustering rows for review/training",
    )
    export_sem_parser.add_argument("--input", type=str, required=True,
                                   help="Directory containing semantic clustering artifacts")
    export_sem_parser.add_argument("--output", type=str, required=True,
                                   help="Output JSONL path")
    export_sem_parser.add_argument("--include-all", action="store_true",
                                   help="Include non-representative rows (default: representatives only)")

    # --- optimize-layout ---
    optimize_parser = subparsers.add_parser(
        "optimize-layout",
        help="One-time optimize historical output layout",
        description="Normalize legacy output naming/layout in an existing run "
                    "directory. Default is dry-run; use --apply to execute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    optimize_parser.add_argument("--input", type=str, required=True,
                                 help="Run directory root to optimize")
    optimize_parser.add_argument("--apply", action="store_true",
                                 help="Apply planned operations (default: dry-run)")
    optimize_parser.add_argument("--prune-legacy", action="store_true",
                                 help="Delete legacy alias files when canonical equivalents exist")
    optimize_parser.add_argument("--manifest", type=str, default=None,
                                 help="Manifest output path (default: <input>/layout_optimization_manifest.json)")

    # --- recompute-stats ---
    recompute_parser = subparsers.add_parser(
        "recompute-stats",
        help="Recompute stats from labeled/scored output (no LLM)",
        description="Rebuild stats.json / stats_value.json from existing pipeline output. "
                    "Useful after manually editing labels or merging runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    recompute_parser.add_argument("--input", type=str, required=True,
                                   help="Labeled/scored file or run directory")
    recompute_parser.add_argument("--pass", type=str, default="both",
                                   choices=["1", "2", "both"], dest="pass_num",
                                   help="Which pass stats to recompute (default: both)")
    recompute_parser.add_argument("--output", type=str, default=None,
                                   help="Output directory (default: same as input)")
    recompute_parser.add_argument("--workers", type=int, default=8,
                                   help="File-level parallel workers (default: 8)")

    # --- refresh-rarity ---
    refresh_parser = subparsers.add_parser(
        "refresh-rarity",
        help="Refresh Pass 2 rarity/value fields from existing scored output (no LLM)",
        description="Recompute value.rarity, value_score, selection_score and Pass 2 stats "
                    "from existing scored output without LLM calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    refresh_parser.add_argument("--input", type=str, required=True,
                                help="Scored file (.json/.jsonl) or run directory")
    refresh_parser.add_argument("--tag-stats", type=str, default=None,
                                help="Path to Pass 1 stats for cross-dataset rarity baseline")
    refresh_parser.add_argument("--output", type=str, default=None,
                                help="Output directory (default: in-place)")
    refresh_parser.add_argument("--mode", type=str, default=None,
                                choices=["absolute", "percentile"],
                                help="Rarity score normalization mode override")
    refresh_parser.add_argument("--extension-rarity-mode", type=str, default=None,
                                choices=["off", "preview", "bonus_only"],
                                help="Extension Rarity V2 mode: off (default, legacy unchanged), preview (compute extension rarity diagnostics only), or bonus_only (experimental V2 bonus fields only).")
    refresh_parser.add_argument("--workers", type=int, default=8,
                                help="File-level parallel workers in directory mode (default: 8)")

    # --- regenerate-dashboard ---
    regen_parser = subparsers.add_parser(
        "regenerate-dashboard",
        help="Regenerate HTML dashboards from existing stats/data",
        description="Re-generate HTML dashboards from existing stats and data files. "
                    "Requires stats files to exist (run recompute-stats first if needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    regen_parser.add_argument("--input", type=str, required=True,
                               help="Run directory containing stats/data files")
    regen_parser.add_argument("--pass", type=str, default="both",
                               choices=["1", "2", "both"], dest="pass_num",
                               help="Which dashboards to regenerate (default: both)")
    regen_parser.add_argument("--open", action="store_true",
                               help="Open dashboards in browser after generation")
    regen_parser.add_argument("--workers", type=int, default=8,
                               help="Directory-level parallel workers in batch mode (default: 8)")

    # --- complete-postprocess ---
    complete_parser = subparsers.add_parser(
        "complete-postprocess",
        help="Materialize deferred Pass 2 postprocess offline",
        description="Finish deferred Pass 2 conversation aggregation and dashboard generation "
                    "after a large run completes. Default scope=global is large-run-safe; "
                    "scope=all also builds standalone per-file dashboards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    complete_parser.add_argument("--input", type=str, required=True,
                                 help="Run directory containing scored outputs and summary stats")
    complete_parser.add_argument("--scope", type=str, default="global",
                                 choices=["global", "all"],
                                 help="global: per-file conversation_scores + global dashboard "
                                      "(default, safer for huge runs); "
                                      "all: also generate standalone per-file dashboards")
    complete_parser.add_argument("--open", action="store_true",
                                 help="Open the last generated dashboard in browser")
    complete_parser.add_argument("--workers", type=int, default=8,
                                 help="File-level workers for conversation aggregation (default: 8)")

    # --- export-review ---
    review_parser = subparsers.add_parser(
        "export-review",
        help="Export labeled or inline-labeled data to review CSV",
    )
    review_parser.add_argument("--input", type=str, required=True,
                               help="Input labeled JSON/JSONL, mirrored inline JSONL, or run directory")
    review_parser.add_argument("--monitor", type=str, default="",
                               help="Monitor JSONL file (optional)")
    review_parser.add_argument("--output", type=str, required=True,
                               help="Output CSV/TSV path")
    review_parser.add_argument("--format", choices=["csv", "tsv"], default=None,
                               dest="output_format",
                               help="Output format override (default: infer from extension)")
    review_parser.add_argument("--include-extensions", action="store_true",
                               help="Include flattened label extension columns in the review export")

    # --- filter ---
    filter_parser = subparsers.add_parser(
        "filter",
        help="Filter scored or inline-labeled data by value and other criteria",
        description="Select high-value samples from scored artifacts or mirrored inline datasets. "
                    "Supports multi-condition filtering and training format output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    filter_parser.add_argument("--input", type=str, required=True,
                                help="Input scored file, mirrored inline JSONL, or directory/run root")
    filter_parser.add_argument("--value-min", type=float, default=None,
                                help="Minimum value_score to retain")
    filter_parser.add_argument("--threshold", type=float, default=None,
                                help="Alias for --value-min (backward compat)")
    filter_parser.add_argument("--selection-min", type=float, default=None,
                                help="Minimum selection_score to retain")
    filter_parser.add_argument("--include-tags", type=str, default=None,
                                help="Require at least one of these tags (comma-separated, 'dim:tag' format)")
    filter_parser.add_argument("--exclude-tags", type=str, default=None,
                                help="Exclude samples with any of these tags (comma-separated, 'dim:tag' format)")
    filter_parser.add_argument("--difficulty", type=str, default=None,
                                help="Allowed difficulty levels (comma-separated, e.g. 'advanced,expert')")
    filter_parser.add_argument("--thinking-mode", type=str, choices=["slow", "fast"], default=None,
                                help="Filter by thinking mode (slow or fast)")
    filter_parser.add_argument("--exclude-inherited", action="store_true",
                                help="Exclude samples with inherited labels")
    filter_parser.add_argument("--output", type=str, default=None,
                                help="Output file path (default: auto-generated alongside input)")
    filter_parser.add_argument("--include-unscored", action="store_true",
                                help="Keep samples without value scores (default: exclude)")
    filter_parser.add_argument("--format", type=str, choices=["scored", "training"],
                                default="scored", dest="output_format",
                                help="Output format: 'scored' (with labels) or 'training' (training-ready)")
    filter_parser.add_argument("--verify-source", type=str, default=None,
                                help="Only retain samples from this source file path")
    filter_parser.add_argument("--conv-value-min", type=float, default=None,
                                help="Min conversation-level value score (multi-turn)")
    filter_parser.add_argument("--conv-selection-min", type=float, default=None,
                                help="Min conversation-level selection score (multi-turn)")
    filter_parser.add_argument("--conv-value-v2-min", type=float, default=None,
                                help="Min conversation-level value_v2 score (multi-turn trajectory calibration)")
    filter_parser.add_argument("--conv-selection-v2-min", type=float, default=None,
                                help="Min conversation-level selection_v2 score (multi-turn trajectory calibration)")
    filter_parser.add_argument("--peak-complexity-min", type=float, default=None,
                                help="Min peak complexity across turns (multi-turn)")
    filter_parser.add_argument("--trajectory-structure-min", type=float, default=None,
                                help="Min deterministic trajectory structure score (multi-turn)")
    filter_parser.add_argument("--rarity-confidence-min", type=float, default=None,
                                help="Min conversation rarity confidence (multi-turn)")
    filter_parser.add_argument("--observed-turn-ratio-min", type=float, default=None,
                                help="Min directly labeled turn ratio in conversation (multi-turn)")
    filter_parser.add_argument("--turn-count-min", type=int, default=None,
                                help="Min total turns in conversation (multi-turn)")
    filter_parser.add_argument("--turn-count-max", type=int, default=None,
                                help="Max total turns in conversation (multi-turn)")
    filter_parser.add_argument("--correctness-min", type=float, default=None,
                                help="Min quality.correctness score (hard floor)")
    filter_parser.add_argument("--turn-value-min", type=float, default=None,
                                help="Min per-turn value_score (turn-level pruning)")
    filter_parser.add_argument("--turn-quality-min", type=float, default=None,
                                help="Min per-turn quality.overall (turn-level pruning)")
    filter_parser.add_argument("--max-pruned-ratio", type=float, default=0.5,
                                help="Max fraction of turns to prune per conversation (default: 0.5)")
    filter_parser.add_argument("--no-keep-first-last", action="store_true",
                                help="Allow pruning first/last turns in conversations")
    filter_parser.add_argument("--preserve-structure", action="store_true",
                                help="Directory input only: mirror input folder structure and per-file format (skip empty output files)")

    # --- analyze-unmapped ---
    unmapped_parser = subparsers.add_parser(
        "analyze-unmapped",
        help="Inspect unmapped tags from labeled/scored output or stats",
        description="Inspect out-of-pool tags from labeled/scored artifacts, inline runs, "
                    "or Pass 1 labeling stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    unmapped_parser.add_argument("--input", type=str, required=True,
                                 help="Labeled/scored file, inline run root/file, run directory, or stats_labeling.json")
    unmapped_parser.add_argument("--dimension", type=str, default=None,
                                 help="Only show one dimension (e.g. task, concept, unknown)")
    unmapped_parser.add_argument("--top", type=int, default=20,
                                 help="Max unmapped tags to show per dimension (default: 20)")
    unmapped_parser.add_argument("--examples", type=int, default=2,
                                 help="Max sample examples to show per unmapped tag (default: 2)")
    unmapped_parser.add_argument("--stats-only", action="store_true",
                                 help="Only read unmapped summary from stats_labeling.json/stats.json")

    return parser


def dispatch_command(args, parser):
    """Dispatch parsed args to command handlers."""
    if args.command == "start":
        return cmd_start(args, parser)
    elif args.command == "dashboard-service":
        return cmd_dashboard_service(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "score":
        return cmd_score(args)
    elif args.command == "semantic-cluster":
        return cmd_semantic_cluster(args)
    elif args.command == "export-semantic":
        return cmd_export_semantic(args)
    elif args.command == "optimize-layout":
        return cmd_optimize_layout(args)
    elif args.command == "filter":
        return cmd_filter(args)
    elif args.command == "analyze-unmapped":
        return cmd_analyze_unmapped(args)
    elif args.command == "export-review":
        return cmd_export_review(args)
    elif args.command == "recompute-stats":
        return cmd_recompute_stats(args)
    elif args.command == "refresh-rarity":
        return cmd_refresh_rarity(args)
    elif args.command == "regenerate-dashboard":
        return cmd_regenerate_dashboard(args)
    elif args.command == "complete-postprocess":
        return cmd_complete_postprocess(args)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch_command(args, parser=parser)


if __name__ == "__main__":
    main()
