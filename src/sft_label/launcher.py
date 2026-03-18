"""Interactive launcher for sft-label CLI workflows."""

from __future__ import annotations

import codecs
import json
import os
import re
import select
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sft_label.config import (
    DEFAULT_CONCURRENCY,
    DEFAULT_RPS_LIMIT,
    DEFAULT_RPS_WARMUP,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
)

DEFAULT_LITELLM_BASE = "http://localhost:4000/v1"
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
DEFAULT_LANGUAGE = "zh"
SUPPORTED_LANGUAGES = {"zh", "en"}
_LANG = DEFAULT_LANGUAGE
SECTION_DIVIDER = "-" * 56
INTERACTIVE_DEFAULT_PROMPT_MODE = "compact"
DEFAULT_MAINTENANCE_WORKERS = 8
MAINTENANCE_WORKER_CHOICES = (1, 4, 8, 16, 32)
BACK_TOKENS = {
    "b", "back", "/b", ":b",
    "返回", "上一步", "上一级",
}


class BackRequested(Exception):
    """Raised when user asks to return to previous level."""


def set_language(language: str):
    """Set interactive display language."""
    global _LANG
    _LANG = "en" if str(language).lower() == "en" else "zh"


def get_language() -> str:
    """Get current interactive display language."""
    return _LANG


def localize_text(text: str) -> str:
    """Select localized text from a 'zh / en' pair."""
    if text is None:
        return ""
    raw = str(text)
    if " / " not in raw:
        return raw
    left, right = raw.split(" / ", 1)
    return right if _LANG == "en" else left


def _msg(zh: str, en: str) -> str:
    return en if _LANG == "en" else zh


@dataclass
class LaunchPlan:
    """Executable launch plan built from interactive prompts."""

    argv: list[str]
    env_overrides: dict[str, str] = field(default_factory=dict)
    workflow_key: str | None = None


@dataclass(frozen=True)
class Workflow:
    key: str
    label: str
    description: str
    group: str


@dataclass(frozen=True)
class SwitchOption:
    value: str
    label: str


@dataclass(frozen=True)
class SwitchField:
    key: str
    label: str
    options: list[SwitchOption]
    default_value: str


def _format_default_value(value: int | float | str) -> str:
    if isinstance(value, float) and value.is_integer():
        v = str(int(value))
    else:
        v = str(value)
    return f"{v}（默认） / {v} (default)"


WORKFLOWS = [
    Workflow(
        key="run-pass1-pass2",
        label="一阶段 + 二阶段 / Pass 1 + Pass 2",
        description="默认推荐：先标注后打分 / Recommended default: labeling followed by value scoring",
        group="流水线 / Pipeline",
    ),
    Workflow(
        key="run-pass1",
        label="仅一阶段标注 / Pass 1 labeling only",
        description="只产出 taxonomy 标签与一阶段统计 / Produce taxonomy labels and Pass 1 stats only",
        group="流水线 / Pipeline",
    ),
    Workflow(
        key="score",
        label="仅二阶段打分 / Pass 2 scoring only",
        description="对已标注数据评分 / Score pre-labeled data",
        group="流水线 / Pipeline",
    ),
    Workflow(
        key="run-pass1-pass2-semantic",
        label="一阶段 + 二阶段 + 三阶段 / Pass 1 + Pass 2 + Pass 3",
        description="标注、打分、语义聚类 / Labeling, scoring, then semantic clustering",
        group="流水线 / Pipeline",
    ),
    Workflow(
        key="semantic",
        label="仅三阶段语义聚类 / Pass 3 semantic clustering only",
        description="运行 SemHash + ANN 聚类 / Run SemHash + ANN clustering",
        group="流水线 / Pipeline",
    ),
    Workflow(
        key="filter",
        label="四阶段过滤与选样 / Pass 4 filtering & selection",
        description="在最终阶段筛选训练子集 / Final-stage training subset selection",
        group="数据整理 / Data Curation",
    ),
    Workflow(
        key="dashboard-service",
        label="看板服务维护 / Dashboard service maintenance",
        description="管理 dashboard 静态服务、默认服务与已发布 runs / Manage dashboard services, defaults, and published runs",
        group="服务 / Service",
    ),
    Workflow(
        key="recompute-stats",
        label="重算统计 / Recompute stats",
        description="不调用 LLM 重新计算统计 / Rebuild stats without LLM calls",
        group="维护 / Maintenance",
    ),
    Workflow(
        key="regenerate-dashboard",
        label="重建看板 / Regenerate dashboards",
        description="用已有统计重建 HTML 看板 / Rebuild HTML dashboards from existing stats",
        group="维护 / Maintenance",
    ),
    Workflow(
        key="refresh-rarity",
        label="刷新稀有度 / Refresh rarity",
        description="离线重算 rarity/value 分数 / Recompute rarity/value fields offline",
        group="维护 / Maintenance",
    ),
    Workflow(
        key="analyze-unmapped",
        label="查看 unmapped 标签 / Inspect unmapped tags",
        description="查看超出 tag pool 的标签及样例 / Review out-of-pool tags and examples",
        group="维护 / Maintenance",
    ),
    Workflow(
        key="optimize-layout",
        label="优化历史文件布局 / Optimize legacy layout",
        description="统一旧版命名并可选清理 legacy 别名 / Normalize old naming and optionally prune legacy aliases",
        group="维护 / Maintenance",
    ),
    Workflow(
        key="export-semantic",
        label="导出语义窗口 / Export semantic rows",
        description="导出代表窗口或全部窗口 / Export representative/all semantic windows",
        group="导出 / Export",
    ),
    Workflow(
        key="export-review",
        label="导出审核 CSV/TSV / Export review CSV/TSV",
        description="导出人工审核表 / Flatten labeled data for manual review",
        group="导出 / Export",
    ),
    Workflow(
        key="validate",
        label="校验 taxonomy / Validate taxonomy",
        description="校验 taxonomy 文件与一致性 / Validate taxonomy files and consistency",
        group="维护 / Maintenance",
    ),
    Workflow(
        key="smart-resume",
        label="智能续跑（中断恢复优先） / Smart resume (best first choice for interrupted runs)",
        description="遇到中断/报错且不确定停在哪一阶段时，优先选这里 / If a run was interrupted and you're unsure which pass failed, start here",
        group="恢复 / Recovery",
    ),
]


def format_command(argv: list[str]) -> str:
    """Render argv as a shell-safe command preview."""
    return "sft-label " + " ".join(shlex.quote(part) for part in argv)


def format_env_prefix(env_overrides: dict[str, str]) -> str:
    """Render env overrides for preview output."""
    if not env_overrides:
        return ""
    return " ".join(
        f"{k}={shlex.quote(v)}" for k, v in sorted(env_overrides.items())
    )


def sanitize_prompt_input(raw: str) -> tuple[str, bool]:
    """Remove terminal control/escape sequences from user input."""
    text = "" if raw is None else str(raw)
    cleaned = ANSI_ESCAPE_RE.sub("", text).replace("\x1b", "")
    filtered = "".join(ch for ch in cleaned if ch == "\t" or ord(ch) >= 32)
    had_control = filtered != text
    return filtered.strip(), had_control


def _consume_escape_sequence(fd: int, max_bytes: int = 8, timeout: float = 0.003):
    """Best-effort consume of pending terminal escape sequence bytes."""
    for _ in range(max_bytes):
        readable, _, _ = select.select([fd], [], [], timeout)
        if not readable:
            break
        try:
            os.read(fd, 1)
        except OSError:
            break


def _drain_pending_input(max_bytes: int = 64, timeout: float = 0.001):
    """Drain any pending stdin bytes to avoid accidental key carry-over."""
    if not sys.stdin.isatty():
        return
    fd = sys.stdin.fileno()
    for _ in range(max_bytes):
        readable, _, _ = select.select([fd], [], [], timeout)
        if not readable:
            break
        try:
            os.read(fd, 1)
        except OSError:
            break


def _decode_utf8_input_byte(
    decoder: codecs.IncrementalDecoder,
    data: bytes,
) -> str:
    """Decode one raw input byte with incremental UTF-8 state."""
    try:
        return decoder.decode(data, final=False)
    except UnicodeDecodeError:
        decoder.reset()
        return ""


def interactive_input(prompt: str) -> str:
    """TTY-friendly input that suppresses arrow/control key escape echoes."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return input(prompt)

    try:
        import termios
        import tty
    except Exception:
        return input(prompt)

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    chars: list[str] = []
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")

    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        tty.setraw(fd)
        while True:
            raw = os.read(fd, 1)
            if not raw:
                raise EOFError

            if raw in (b"\r", b"\n"):
                # In raw TTY mode, '\n' only moves to next line and keeps the
                # current column. Use CRLF to avoid progressive indentation.
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                break

            if raw == b"\x03":
                raise KeyboardInterrupt

            if raw == b"\x04":
                if not chars:
                    raise EOFError
                continue

            if raw in (b"\x7f", b"\x08"):
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if raw == b"\x1b":
                _consume_escape_sequence(fd)
                decoder.reset()
                continue

            text = _decode_utf8_input_byte(decoder, raw)
            if not text:
                continue
            for c in text:
                if ord(c) >= 32:
                    chars.append(c)
                    sys.stdout.write(c)
                    sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    return "".join(chars)


def _supports_switch_panel(input_fn: InputFn) -> bool:
    return (
        input_fn is interactive_input
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    )


def _read_switch_key(fd: int, decoder: codecs.IncrementalDecoder) -> str:
    """Read one key token for interactive switch panel."""
    raw = os.read(fd, 1)
    if not raw:
        raise EOFError

    if raw in (b"\r", b"\n"):
        return "enter"
    if raw == b"\x03":
        raise KeyboardInterrupt
    if raw == b"\x1b":
        suffix = b""
        for _ in range(2):
            readable, _, _ = select.select([fd], [], [], 0.01)
            if not readable:
                break
            suffix += os.read(fd, 1)
        arrow_map = {
            b"[A": "up",
            b"[B": "down",
            b"[C": "right",
            b"[D": "left",
            b"OA": "up",
            b"OB": "down",
            b"OC": "right",
            b"OD": "left",
        }
        decoder.reset()
        return arrow_map.get(suffix, "escape")

    text = _decode_utf8_input_byte(decoder, raw)
    if not text:
        return "noop"
    ch = text.lower()
    if ch == "b":
        return "back"
    return "noop"


def _build_switch_panel_lines(
    title: str,
    fields: list[SwitchField],
    current_row: int,
    current_option_indices: list[int],
) -> list[str]:
    lines = [
        SECTION_DIVIDER,
        localize_text(title),
        _msg(
            "上下选择字段，左右切换值，回车确认，按 b 返回上一级。",
            "Use Up/Down to select field, Left/Right to switch value, Enter to confirm, b to go back.",
        ),
        SECTION_DIVIDER,
        "",
    ]
    for idx, switch_field in enumerate(fields):
        marker = ">" if idx == current_row else " "
        selected = switch_field.options[current_option_indices[idx]]
        lines.append(f"{marker} {localize_text(switch_field.label)}: {localize_text(selected.label)}")
    return lines


def _write_switch_panel_lines(lines: list[str]):
    panel = "\r\n".join(lines) + "\r\n"
    sys.stdout.write(panel)
    sys.stdout.flush()


def _refresh_switch_panel_lines(lines: list[str]):
    # Move cursor back to the start of panel block, then rewrite in place.
    line_count = len(lines)
    if line_count > 0:
        sys.stdout.write(f"\x1b[{line_count}A")
    for line in lines:
        sys.stdout.write(f"\r\x1b[2K{line}\r\n")
    sys.stdout.flush()


def _ask_switch_panel(
    input_fn: InputFn,
    output_fn: OutputFn,
    title: str,
    fields: list[SwitchField],
) -> dict[str, str]:
    if not _supports_switch_panel(input_fn):
        values: dict[str, str] = {}
        for field in fields:
            default_index = 1
            for i, opt in enumerate(field.options, start=1):
                if opt.value == field.default_value:
                    default_index = i
                    break
            choice = _ask_choice(
                input_fn,
                output_fn,
                field.label,
                [(opt.value, opt.label, "") for opt in field.options],
                default_index=default_index,
            )
            values[field.key] = choice
        return values

    try:
        import termios
        import tty
    except Exception:
        values: dict[str, str] = {}
        for field in fields:
            values[field.key] = field.default_value
        return values

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    current_row = 0
    current_option_indices = []
    for field in fields:
        default_idx = 0
        for idx, opt in enumerate(field.options):
            if opt.value == field.default_value:
                default_idx = idx
                break
        current_option_indices.append(default_idx)

    try:
        tty.setraw(fd)
        lines = _build_switch_panel_lines(title, fields, current_row, current_option_indices)
        _write_switch_panel_lines(lines)
        while True:
            token = _read_switch_key(fd, decoder)
            changed = False
            if token == "up":
                current_row = (current_row - 1) % len(fields)
                changed = True
            elif token == "down":
                current_row = (current_row + 1) % len(fields)
                changed = True
            elif token == "left":
                current_option_indices[current_row] = (
                    current_option_indices[current_row] - 1
                ) % len(fields[current_row].options)
                changed = True
            elif token == "right":
                current_option_indices[current_row] = (
                    current_option_indices[current_row] + 1
                ) % len(fields[current_row].options)
                changed = True
            elif token == "enter":
                break
            elif token == "back":
                raise BackRequested
            if changed:
                lines = _build_switch_panel_lines(title, fields, current_row, current_option_indices)
                _refresh_switch_panel_lines(lines)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    return {
        field.key: field.options[current_option_indices[idx]].value
        for idx, field in enumerate(fields)
    }


def is_back_token(value: str) -> bool:
    """Check whether an input value means 'go back'."""
    if value is None:
        return False
    token = str(value).strip().lower()
    return token in BACK_TOKENS


def _read_input(input_fn: InputFn, prompt: str, *, allow_back: bool = True) -> str:
    """Read input and sanitize terminal control sequences."""
    value, had_control = sanitize_prompt_input(input_fn(localize_text(prompt)))
    if had_control:
        print(_msg("检测到方向键/控制字符输入，已忽略。", "Detected arrow/control key input and ignored."))
    if allow_back and is_back_token(value):
        raise BackRequested
    return value


def _with_workflow_key(plan: LaunchPlan, workflow_key: str) -> LaunchPlan:
    plan.workflow_key = workflow_key
    return plan


def _find_resume_checkpoint(run_dir: Path) -> Path | None:
    candidates = (
        run_dir / "checkpoint.json",
        run_dir / "meta_label_data" / "checkpoint.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_resume_checkpoint(run_dir: Path) -> tuple[Path | None, dict | None]:
    checkpoint_path = _find_resume_checkpoint(run_dir)
    if checkpoint_path is None:
        return None, None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return checkpoint_path, None
    return checkpoint_path, payload if isinstance(payload, dict) else None


def _find_matching_artifacts(root: Path, stem_prefix: str) -> list[Path]:
    patterns = (
        f"{stem_prefix}*.json",
        f"{stem_prefix}*.jsonl",
        f"*/{stem_prefix}*.json",
        f"*/{stem_prefix}*.jsonl",
        f"**/{stem_prefix}*.json",
        f"**/{stem_prefix}*.jsonl",
    )
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def _score_command_for_input(target: Path, *, resume: bool) -> list[str]:
    argv = ["score", "--concurrency", str(DEFAULT_CONCURRENCY), "--input", str(target)]
    if resume:
        argv.append("--resume")
    return argv


def _detect_smart_resume_command(target: Path) -> tuple[list[str], str]:
    if not target.exists():
        raise ValueError(_msg(f"目录不存在：{target}", f"Path does not exist: {target}"))
    if not target.is_dir():
        raise ValueError(_msg(f"续跑目标必须是目录：{target}", f"Resume target must be a directory: {target}"))

    checkpoint, checkpoint_payload = _load_resume_checkpoint(target)
    labeled_files = _find_matching_artifacts(target, "labeled")
    scored_files = _find_matching_artifacts(target, "scored")
    checkpoint_status = (checkpoint_payload or {}).get("status")

    if checkpoint is not None and checkpoint_status != "done":
        return (
            ["run", "--resume", str(target)],
            _msg(
                f"检测到 checkpoint：{checkpoint}，将继续一阶段/联跑任务。",
                f"Detected checkpoint at {checkpoint}; resuming Pass 1 or chained run.",
            ),
        )

    if labeled_files:
        argv = _score_command_for_input(target, resume=bool(scored_files))
        if scored_files:
            return (
                argv,
                _msg(
                    f"检测到 {len(labeled_files)} 个 labeled 文件和 {len(scored_files)} 个 scored 文件，将续跑二阶段并跳过已完成样本。",
                    f"Detected {len(labeled_files)} labeled file(s) and {len(scored_files)} scored file(s); resuming Pass 2 and skipping completed samples.",
                ),
            )
        return (
            argv,
            _msg(
                f"检测到 {len(labeled_files)} 个 labeled 文件，未发现 scored 结果，将启动二阶段打分。",
                f"Detected {len(labeled_files)} labeled file(s) and no scored outputs; starting Pass 2 scoring.",
            ),
        )

    raise ValueError(
        _msg(
            "未检测到 checkpoint、labeled 或 scored 产物，请确认你输入的是 run 目录。",
            "No checkpoint, labeled, or scored artifacts found. Please provide a run directory.",
        )
    )


def build_launch_plan(
    input_fn: InputFn = interactive_input,
    output_fn: OutputFn = print,
    language: str | None = None,
) -> LaunchPlan | None:
    """Prompt for workflow/options and return executable launch plan."""
    if language:
        set_language(language)

    _say(output_fn, SECTION_DIVIDER)
    _say(output_fn, "交互式任务启动器 / Interactive task launcher")
    _say(output_fn, "请选择任务，并只填写必要参数 / Choose a workflow and fill only required options.")
    _say(output_fn, "如果任务中断或报错，且不确定停在哪一阶段，请优先选择“智能续跑” / If a run was interrupted and you're unsure which pass failed, choose 'Smart resume' first.")
    _say(output_fn, "输入 b/back 可返回上一层，输入 0 可取消 / Type b/back to go back, 0 to cancel.")
    _say(output_fn, SECTION_DIVIDER)
    _say(output_fn, "")

    while True:
        wf = _ask_workflow(input_fn, output_fn)
        if wf is None:
            return None

        _say(output_fn, "")
        _say(
            output_fn,
            _msg("正在配置作业：", "Configuring workflow: ") + localize_text(wf.label),
        )
        _say(
            output_fn,
            _msg("作业说明：", "Description: ") + localize_text(wf.description),
        )
        _say(output_fn, _msg("输入 b/back 返回任务列表。", "Type b/back to return to workflow menu."))
        _say(output_fn, "")

        try:
            if wf.key == "run-pass1":
                return _with_workflow_key(_build_run_plan(input_fn, output_fn, chain_score=False, chain_semantic=False), wf.key)
            if wf.key == "run-pass1-pass2":
                return _with_workflow_key(_build_run_plan(input_fn, output_fn, chain_score=True, chain_semantic=False), wf.key)
            if wf.key == "run-pass1-pass2-semantic":
                return _with_workflow_key(_build_run_plan(input_fn, output_fn, chain_score=True, chain_semantic=True), wf.key)
            if wf.key == "score":
                return _with_workflow_key(_build_score_plan(input_fn, output_fn), wf.key)
            if wf.key == "semantic":
                return _with_workflow_key(_build_semantic_plan(input_fn, output_fn), wf.key)
            if wf.key == "filter":
                return _with_workflow_key(_build_filter_plan(input_fn, output_fn), wf.key)
            if wf.key == "recompute-stats":
                return _with_workflow_key(_build_recompute_plan(input_fn, output_fn), wf.key)
            if wf.key == "refresh-rarity":
                return _with_workflow_key(_build_refresh_rarity_plan(input_fn, output_fn), wf.key)
            if wf.key == "regenerate-dashboard":
                return _with_workflow_key(_build_regenerate_dashboard_plan(input_fn, output_fn), wf.key)
            if wf.key == "export-semantic":
                return _with_workflow_key(_build_export_semantic_plan(input_fn, output_fn), wf.key)
            if wf.key == "export-review":
                return _with_workflow_key(_build_export_review_plan(input_fn, output_fn), wf.key)
            if wf.key == "validate":
                return LaunchPlan(argv=["validate"], workflow_key=wf.key)
            if wf.key == "analyze-unmapped":
                return _with_workflow_key(_build_analyze_unmapped_plan(input_fn, output_fn), wf.key)
            if wf.key == "optimize-layout":
                return _with_workflow_key(_build_optimize_layout_plan(input_fn, output_fn), wf.key)
            if wf.key == "dashboard-service":
                return _with_workflow_key(_build_dashboard_service_plan(input_fn, output_fn), wf.key)
            if wf.key == "smart-resume":
                return _with_workflow_key(_build_smart_resume_plan(input_fn, output_fn), wf.key)
            raise ValueError(f"Unknown workflow key: {wf.key}")
        except BackRequested:
            if input_fn is interactive_input:
                _drain_pending_input()
            _say(output_fn, "")
            _say(output_fn, "已返回任务选择 / Returned to workflow selection.")
            _say(output_fn, "")


def _build_smart_resume_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    _section(output_fn, "智能续跑 / Smart resume")
    while True:
        run_dir = Path(_ask_required_text(input_fn, "续跑目录 / Run directory to resume")).expanduser().resolve()
        try:
            argv, detection_message = _detect_smart_resume_command(run_dir)
        except ValueError as exc:
            _say(output_fn, str(exc))
            continue
        _say(output_fn, detection_message)
        argv.extend(_ask_extra_flags(input_fn))
        return LaunchPlan(argv=argv)


def _build_run_plan(
    input_fn: InputFn,
    output_fn: OutputFn,
    *,
    chain_score: bool,
    chain_semantic: bool,
) -> LaunchPlan:
    if not _supports_switch_panel(input_fn):
        return _build_run_plan_legacy(
            input_fn,
            output_fn,
            chain_score=chain_score,
            chain_semantic=chain_semantic,
        )

    argv = ["run"]
    env_overrides: dict[str, str] = {}

    _section(output_fn, "基础输入 / Basic input")
    mode = _ask_choice(
        input_fn,
        output_fn,
        "运行模式 / Run mode",
        [
            ("new", "新建运行 / Start new run", "使用 --input，可选 --output / Use --input and optional --output"),
            ("resume", "续跑已有任务 / Resume existing run", "使用 --resume 继续中断任务 / Use --resume and continue an interrupted run"),
        ],
        default_index=1,
    )
    if mode == "new":
        argv.extend(["--input", _ask_required_text(input_fn, "输入文件或目录路径 / Input file/dir path")])
    else:
        argv.extend(["--resume", _ask_required_text(input_fn, "续跑目录 / Run directory to resume")])
        if _ask_yes_no(input_fn, "是否同时设置 --input 路径 / Also set --input path", default=False):
            argv.extend(["--input", _ask_required_text(input_fn, "输入文件或目录路径 / Input file/dir path")])

    if mode == "new":
        output_path = _ask_optional_text(input_fn, "输出目录（可选） / Output directory (optional)")
        if output_path:
            argv.extend(["--output", output_path])

    inline_mode = "refresh"
    if mode == "new":
        inline_mode = _ask_choice(
            input_fn,
            output_fn,
            "标注模式 / Inline mode",
            [
                ("refresh", "refresh（默认） / refresh (default)",
                 "重算并替换整条 data_label / Recompute and replace the whole data_label"),
                ("incremental", "incremental",
                 "跳过已完成样本，仅补缺失部分 / Skip completed rows and fill missing turns"),
                ("migrate", "migrate",
                 "先按 data_id 复制旧标注，再增量补齐 / Copy labels by data_id, then fill gaps"),
                ("recompute", "recompute",
                 "不调用 LLM，只重算统计/稀有度/看板 / Rebuild stats, rarity, and dashboards without LLM"),
            ],
            default_index=1,
        )
        if inline_mode != "refresh":
            argv.extend(["--mode", inline_mode])
        if inline_mode == "migrate":
            argv.extend(["--migrate-from", _ask_required_text(
                input_fn,
                "迁移来源（旧 run / 数据集） / Migration source (previous run or dataset)",
            )])

    if inline_mode == "recompute":
        argv.extend(_ask_extra_flags(input_fn))
        return LaunchPlan(argv=argv, env_overrides=env_overrides)

    if chain_score:
        argv.append("--score")
        tag_stats = _ask_optional_text(input_fn, "稀有度统计文件路径（可选） / Tag stats path for rarity (optional)")
        if tag_stats:
            argv.extend(["--tag-stats", tag_stats])
        rarity_mode = _ask_choice(
            input_fn,
            output_fn,
            "稀有度归一化模式 / Rarity normalization mode",
            [
                ("absolute", "absolute（默认） / absolute (default)",
                 "跨数据集绝对刻度 / Cross-dataset absolute scale"),
                ("percentile", "percentile",
                 "批内百分位映射 / Within-batch percentile mapping"),
            ],
            default_index=1,
        )
        if rarity_mode != "absolute":
            argv.extend(["--rarity-mode", rarity_mode])

    if chain_semantic:
        _section(output_fn, "三阶段联动 / Pass 3 chaining")
        argv.append("--semantic-cluster")

    switch_fields = [
        SwitchField(
            key="prompt_mode",
            label="Prompt 模式 / Prompt mode",
            options=[
                SwitchOption("compact", "compact（默认） / compact (default)"),
                SwitchOption("full", "full"),
            ],
            default_value=INTERACTIVE_DEFAULT_PROMPT_MODE,
        ),
        SwitchField(
            key="shuffle",
            label="处理前打乱样本 / Shuffle samples before processing",
            options=[
                SwitchOption("off", "关闭 / Off"),
                SwitchOption("on", "开启 / On"),
            ],
            default_value="off",
        ),
        SwitchField(
            key="arbitration",
            label="启用仲裁补跑 / Enable arbitration pass",
            options=[
                SwitchOption("on", "开启 / On"),
                SwitchOption("off", "关闭 / Off"),
            ],
            default_value="on",
        ),
        SwitchField(
            key="limit",
            label="每个文件采样上限 / Sample limit per file",
            options=[
                SwitchOption("", _format_default_value(0)),
                SwitchOption("10", "10"),
                SwitchOption("50", "50"),
                SwitchOption("100", "100"),
                SwitchOption("200", "200"),
                SwitchOption("500", "500"),
                SwitchOption("1000", "1000"),
            ],
            default_value="",
        ),
        SwitchField(
            key="concurrency",
            label="LLM 最大并发上限（阶段一/二共用） / Shared max LLM concurrency cap (pass1/pass2)",
            options=[
                SwitchOption("", _format_default_value(DEFAULT_CONCURRENCY)),
                SwitchOption("0", "0"),
                SwitchOption("50", "50"),
                SwitchOption("100", "100"),
                SwitchOption("150", "150"),
                SwitchOption("200", "200"),
                SwitchOption("250", "250"),
                SwitchOption("300", "300"),
                SwitchOption("400", "400"),
                SwitchOption("500", "500"),
            ],
            default_value="",
        ),
        SwitchField(
            key="rps_limit",
            label="RPS 最大上限 / Max RPS cap",
            options=[
                SwitchOption("", _format_default_value(DEFAULT_RPS_LIMIT)),
                SwitchOption("0", "0"),
                SwitchOption("10", "10"),
                SwitchOption("20", "20"),
                SwitchOption("30", "30"),
                SwitchOption("40", "40"),
                SwitchOption("50", "50"),
                SwitchOption("75", "75"),
                SwitchOption("100", "100"),
            ],
            default_value="",
        ),
        SwitchField(
            key="rps_warmup",
            label="RPS 预热秒数 / RPS warmup seconds",
            options=[
                SwitchOption("", _format_default_value(DEFAULT_RPS_WARMUP)),
                SwitchOption("0", "0"),
                SwitchOption("10", "10"),
                SwitchOption("20", "20"),
                SwitchOption("30", "30"),
                SwitchOption("40", "40"),
                SwitchOption("50", "50"),
                SwitchOption("60", "60"),
            ],
            default_value="",
        ),
        SwitchField(
            key="adaptive_runtime",
            label="自适应运行时 / Adaptive runtime",
            options=[
                SwitchOption("on", "开启（默认） / On (default)"),
                SwitchOption("off", "关闭 / Off"),
            ],
            default_value="on",
        ),
        SwitchField(
            key="recovery_sweep",
            label="阶段收尾恢复补跑 / End-of-phase recovery sweep",
            options=[
                SwitchOption("on", "开启（默认） / On (default)"),
                SwitchOption("off", "关闭 / Off"),
            ],
            default_value="on",
        ),
        SwitchField(
            key="request_timeout",
            label="请求超时秒数 / Request timeout seconds",
            options=[
                SwitchOption("", _format_default_value(REQUEST_TIMEOUT)),
                SwitchOption("60", "60"),
                SwitchOption("90", "90"),
                SwitchOption("120", "120"),
                SwitchOption("180", "180"),
            ],
            default_value="",
        ),
        SwitchField(
            key="max_retries",
            label="最大重试次数 / Max retries",
            options=[
                SwitchOption("", _format_default_value(MAX_RETRIES)),
                SwitchOption("1", "1"),
                SwitchOption("3", "3"),
                SwitchOption("5", "5"),
            ],
            default_value="",
        ),
        SwitchField(
            key="model_override",
            label="覆盖 LLM 模型 / Override LLM model",
            options=[
                SwitchOption("no", "否 / No"),
                SwitchOption("yes", "是 / Yes"),
            ],
            default_value="no",
        ),
        SwitchField(
            key="env_override",
            label="覆盖 LITELLM_BASE/LITELLM_KEY / Override LITELLM_BASE/LITELLM_KEY",
            options=[
                SwitchOption("no", "否 / No"),
                SwitchOption("yes", "是 / Yes"),
            ],
            default_value="no",
        ),
    ]
    switch_values = _ask_switch_panel(
        input_fn,
        output_fn,
        "运行配置（上下选择，左右切值） / Run config (Up/Down select, Left/Right switch)",
        switch_fields,
    )

    if switch_values["prompt_mode"] != "full":
        argv.extend(["--prompt-mode", switch_values["prompt_mode"]])
    if switch_values["shuffle"] == "on":
        argv.append("--shuffle")
    if switch_values["arbitration"] == "off":
        argv.append("--no-arbitration")

    if switch_values["model_override"] == "yes":
        model = _ask_required_text(input_fn, "输入覆盖模型名 / Enter override model name")
        argv.extend(["--model", model])

    limit = switch_values.get("limit", "")
    if limit != "":
        argv.extend(["--limit", limit])

    for key, flag in (
        ("concurrency", "--concurrency"),
        ("rps_limit", "--rps-limit"),
        ("rps_warmup", "--rps-warmup"),
        ("request_timeout", "--request-timeout"),
        ("max_retries", "--max-retries"),
    ):
        val = switch_values.get(key, "")
        if val != "":
            argv.extend([flag, val])
    if switch_values["adaptive_runtime"] == "off":
        argv.append("--no-adaptive-runtime")
    if switch_values["recovery_sweep"] == "off":
        argv.append("--no-recovery-sweep")

    _section(output_fn, "环境与附加参数 / Environment and extra flags")
    if switch_values["env_override"] == "yes":
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_score_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    if not _supports_switch_panel(input_fn):
        return _build_score_plan_legacy(input_fn, output_fn)

    argv = ["score"]
    env_overrides: dict[str, str] = {}

    _section(output_fn, "基础输入 / Basic input")
    argv.extend(["--input", _ask_required_text(input_fn, "输入已标注文件或目录 / Input labeled file/dir")])

    _section(output_fn, "评分参数 / Scoring options")
    tag_stats = _ask_optional_text(input_fn, "稀有度统计文件路径（可选） / Tag stats path for rarity (optional)")
    if tag_stats:
        argv.extend(["--tag-stats", tag_stats])

    switch_values = _ask_switch_panel(
        input_fn,
        output_fn,
        "评分配置（上下选择，左右切值） / Score config (Up/Down select, Left/Right switch)",
        [
            SwitchField(
                key="prompt_mode",
                label="Prompt 模式 / Prompt mode",
                options=[
                    SwitchOption("compact", "compact（默认） / compact (default)"),
                    SwitchOption("full", "full"),
                ],
                default_value=INTERACTIVE_DEFAULT_PROMPT_MODE,
            ),
            SwitchField(
                key="resume",
                label="续跑并跳过已存在样本 / Resume and skip existing samples",
                options=[
                    SwitchOption("off", "关闭 / Off"),
                    SwitchOption("on", "开启 / On"),
                ],
                default_value="off",
            ),
            SwitchField(
                key="rarity_mode",
                label="稀有度归一化 / Rarity normalization",
                options=[
                    SwitchOption("absolute", "absolute（默认） / absolute (default)"),
                    SwitchOption("percentile", "percentile"),
                ],
                default_value="absolute",
            ),
            SwitchField(
                key="limit",
                label="采样上限 / Sample limit",
                options=[
                    SwitchOption("", _format_default_value(0)),
                    SwitchOption("10", "10"),
                    SwitchOption("50", "50"),
                    SwitchOption("100", "100"),
                    SwitchOption("200", "200"),
                    SwitchOption("500", "500"),
                    SwitchOption("1000", "1000"),
                ],
                default_value="",
            ),
            SwitchField(
                key="concurrency",
                label="LLM 最大并发上限（阶段一/二共用） / Shared max LLM concurrency cap (pass1/pass2)",
                options=[
                    SwitchOption("100", _format_default_value(DEFAULT_CONCURRENCY)),
                    SwitchOption("0", "0"),
                    SwitchOption("50", "50"),
                    SwitchOption("150", "150"),
                    SwitchOption("200", "200"),
                    SwitchOption("250", "250"),
                    SwitchOption("300", "300"),
                    SwitchOption("400", "400"),
                    SwitchOption("500", "500"),
                ],
                default_value="100",
            ),
            SwitchField(
                key="rps_limit",
                label="RPS 最大上限 / Max RPS cap",
                options=[
                    SwitchOption("", _format_default_value(DEFAULT_RPS_LIMIT)),
                    SwitchOption("0", "0"),
                    SwitchOption("10", "10"),
                    SwitchOption("20", "20"),
                    SwitchOption("30", "30"),
                    SwitchOption("40", "40"),
                    SwitchOption("50", "50"),
                    SwitchOption("75", "75"),
                    SwitchOption("100", "100"),
                ],
                default_value="",
            ),
            SwitchField(
                key="rps_warmup",
                label="RPS 预热秒数 / RPS warmup seconds",
                options=[
                    SwitchOption("", _format_default_value(DEFAULT_RPS_WARMUP)),
                    SwitchOption("0", "0"),
                    SwitchOption("10", "10"),
                    SwitchOption("20", "20"),
                    SwitchOption("30", "30"),
                    SwitchOption("40", "40"),
                    SwitchOption("50", "50"),
                    SwitchOption("60", "60"),
                ],
                default_value="",
            ),
            SwitchField(
                key="adaptive_runtime",
                label="自适应运行时 / Adaptive runtime",
                options=[
                    SwitchOption("on", "开启（默认） / On (default)"),
                    SwitchOption("off", "关闭 / Off"),
                ],
                default_value="on",
            ),
            SwitchField(
                key="recovery_sweep",
                label="阶段收尾恢复补跑 / End-of-phase recovery sweep",
                options=[
                    SwitchOption("on", "开启（默认） / On (default)"),
                    SwitchOption("off", "关闭 / Off"),
                ],
                default_value="on",
            ),
            SwitchField(
                key="request_timeout",
                label="请求超时秒数 / Request timeout seconds",
                options=[
                    SwitchOption("", _format_default_value(REQUEST_TIMEOUT)),
                    SwitchOption("60", "60"),
                    SwitchOption("90", "90"),
                    SwitchOption("120", "120"),
                    SwitchOption("180", "180"),
                ],
                default_value="",
            ),
            SwitchField(
                key="max_retries",
                label="最大重试次数 / Max retries",
                options=[
                    SwitchOption("", _format_default_value(MAX_RETRIES)),
                    SwitchOption("1", "1"),
                    SwitchOption("3", "3"),
                    SwitchOption("5", "5"),
                ],
                default_value="",
            ),
            SwitchField(
                key="model_override",
                label="覆盖 LLM 模型 / Override LLM model",
                options=[
                    SwitchOption("no", "否 / No"),
                    SwitchOption("yes", "是 / Yes"),
                ],
                default_value="no",
            ),
            SwitchField(
                key="env_override",
                label="覆盖 LITELLM_BASE/LITELLM_KEY / Override LITELLM_BASE/LITELLM_KEY",
                options=[
                    SwitchOption("no", "否 / No"),
                    SwitchOption("yes", "是 / Yes"),
                ],
                default_value="no",
            ),
        ],
    )
    if switch_values["prompt_mode"] != "full":
        argv.extend(["--prompt-mode", switch_values["prompt_mode"]])
    if switch_values["resume"] == "on":
        argv.append("--resume")
    if switch_values.get("rarity_mode") == "percentile":
        argv.extend(["--rarity-mode", "percentile"])

    if switch_values["model_override"] == "yes":
        model = _ask_required_text(input_fn, "输入覆盖模型名 / Enter override model name")
        argv.extend(["--model", model])

    limit = switch_values.get("limit", "")
    if limit != "":
        argv.extend(["--limit", limit])

    for key, flag in (
        ("concurrency", "--concurrency"),
        ("rps_limit", "--rps-limit"),
        ("rps_warmup", "--rps-warmup"),
        ("request_timeout", "--request-timeout"),
        ("max_retries", "--max-retries"),
    ):
        val = switch_values.get(key, "")
        if val != "":
            argv.extend([flag, val])
    if switch_values["adaptive_runtime"] == "off":
        argv.append("--no-adaptive-runtime")
    if switch_values["recovery_sweep"] == "off":
        argv.append("--no-recovery-sweep")

    if switch_values["env_override"] == "yes":
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_run_plan_legacy(
    input_fn: InputFn,
    output_fn: OutputFn,
    *,
    chain_score: bool,
    chain_semantic: bool,
) -> LaunchPlan:
    argv = ["run"]
    env_overrides: dict[str, str] = {}

    _section(output_fn, "基础输入 / Basic input")
    mode = _ask_choice(
        input_fn,
        output_fn,
        "运行模式 / Run mode",
        [
            ("new", "新建运行 / Start new run", "使用 --input，可选 --output / Use --input and optional --output"),
            ("resume", "续跑已有任务 / Resume existing run", "使用 --resume 继续中断任务 / Use --resume and continue an interrupted run"),
        ],
        default_index=1,
    )
    if mode == "new":
        argv.extend(["--input", _ask_required_text(input_fn, "输入文件或目录路径 / Input file/dir path")])
    else:
        argv.extend(["--resume", _ask_required_text(input_fn, "续跑目录 / Run directory to resume")])
        if _ask_yes_no(input_fn, "是否同时设置 --input 路径 / Also set --input path", default=False):
            argv.extend(["--input", _ask_required_text(input_fn, "输入文件或目录路径 / Input file/dir path")])

    if mode == "new":
        output_path = _ask_optional_text(input_fn, "输出目录（可选） / Output directory (optional)")
        if output_path:
            argv.extend(["--output", output_path])

    inline_mode = "refresh"
    if mode == "new":
        inline_mode = _ask_choice(
            input_fn,
            output_fn,
            "标注模式 / Inline mode",
            [
                ("refresh", "refresh（默认） / refresh (default)",
                 "重算并替换整条 data_label / Recompute and replace the whole data_label"),
                ("incremental", "incremental",
                 "跳过已完成样本，仅补缺失部分 / Skip completed rows and fill missing turns"),
                ("migrate", "migrate",
                 "先按 data_id 复制旧标注，再增量补齐 / Copy labels by data_id, then fill gaps"),
                ("recompute", "recompute",
                 "不调用 LLM，只重算统计/稀有度/看板 / Rebuild stats, rarity, and dashboards without LLM"),
            ],
            default_index=1,
        )
        if inline_mode != "refresh":
            argv.extend(["--mode", inline_mode])
        if inline_mode == "migrate":
            argv.extend(["--migrate-from", _ask_required_text(
                input_fn,
                "迁移来源（旧 run / 数据集） / Migration source (previous run or dataset)",
            )])

    if inline_mode == "recompute":
        argv.extend(_ask_extra_flags(input_fn))
        return LaunchPlan(argv=argv, env_overrides=env_overrides)

    limit = _ask_int(input_fn, "每个文件采样上限（0=全部） / Sample limit per file (0 = all)", default=0)
    if limit:
        argv.extend(["--limit", str(limit)])

    if _ask_yes_no(input_fn, "处理前打乱样本 / Shuffle samples before processing", default=False):
        argv.append("--shuffle")

    if not _ask_yes_no(input_fn, "启用仲裁补跑 / Enable arbitration pass", default=True):
        argv.append("--no-arbitration")

    _section(output_fn, "模型与推理 / Model and inference")
    if not _ask_yes_no(input_fn, "启用自适应运行时 / Enable adaptive runtime", default=True):
        argv.append("--no-adaptive-runtime")
    if not _ask_yes_no(input_fn, "阶段收尾执行恢复补跑 / Run end-of-phase recovery sweep", default=True):
        argv.append("--no-recovery-sweep")
    prompt_mode = _ask_choice(
        input_fn,
        output_fn,
        "Prompt 模式 / Prompt mode",
        [
            ("full", "full", "质量更高，负载更大 / Higher quality, larger payload"),
            ("compact", "compact", "负载更小，适合受限端点 / Smaller payload for size-limited endpoints"),
        ],
        default_index=1,
    )
    if prompt_mode != "full":
        argv.extend(["--prompt-mode", prompt_mode])

    model = _ask_optional_text(input_fn, "LLM 模型覆盖（可选） / LLM model override (optional)")
    if model:
        argv.extend(["--model", model])

    if chain_score:
        _section(output_fn, "二阶段联动 / Pass 2 chaining")
        argv.append("--score")
        tag_stats = _ask_optional_text(input_fn, "稀有度统计文件路径（可选） / Tag stats path for rarity (optional)")
        if tag_stats:
            argv.extend(["--tag-stats", tag_stats])
        rarity_mode = _ask_choice(
            input_fn,
            output_fn,
            "稀有度归一化模式 / Rarity normalization mode",
            [
                ("absolute", "absolute（默认） / absolute (default)",
                 "跨数据集绝对刻度 / Cross-dataset absolute scale"),
                ("percentile", "percentile",
                 "批内百分位映射 / Within-batch percentile mapping"),
            ],
            default_index=1,
        )
        if rarity_mode != "absolute":
            argv.extend(["--rarity-mode", rarity_mode])

    if chain_semantic:
        _section(output_fn, "三阶段联动 / Pass 3 chaining")
        argv.append("--semantic-cluster")

    _section(output_fn, "环境与附加参数 / Environment and extra flags")
    if _ask_yes_no(input_fn, "覆盖本次 LITELLM_BASE/LITELLM_KEY / Override LITELLM_BASE/LITELLM_KEY for this run", default=False):
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_score_plan_legacy(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["score", "--concurrency", str(DEFAULT_CONCURRENCY)]
    env_overrides: dict[str, str] = {}

    _section(output_fn, "基础输入 / Basic input")
    argv.extend(["--input", _ask_required_text(input_fn, "输入已标注文件或目录 / Input labeled file/dir")])

    _section(output_fn, "评分参数 / Scoring options")
    tag_stats = _ask_optional_text(input_fn, "稀有度统计文件路径（可选） / Tag stats path for rarity (optional)")
    if tag_stats:
        argv.extend(["--tag-stats", tag_stats])
    rarity_mode = _ask_choice(
        input_fn,
        output_fn,
        "稀有度归一化模式 / Rarity normalization mode",
        [
            ("absolute", "absolute（默认） / absolute (default)",
             "跨数据集绝对刻度 / Cross-dataset absolute scale"),
            ("percentile", "percentile",
             "批内百分位映射 / Within-batch percentile mapping"),
        ],
        default_index=1,
    )
    if rarity_mode != "absolute":
        argv.extend(["--rarity-mode", rarity_mode])

    limit = _ask_int(input_fn, "采样上限（0=全部） / Sample limit (0 = all)", default=0)
    if limit:
        argv.extend(["--limit", str(limit)])

    if _ask_yes_no(input_fn, "续跑评分并跳过已存在样本 / Resume scoring and skip existing samples", default=False):
        argv.append("--resume")

    if not _ask_yes_no(input_fn, "启用自适应运行时 / Enable adaptive runtime", default=True):
        argv.append("--no-adaptive-runtime")
    if not _ask_yes_no(input_fn, "阶段收尾执行恢复补跑 / Run end-of-phase recovery sweep", default=True):
        argv.append("--no-recovery-sweep")

    prompt_mode = _ask_choice(
        input_fn,
        output_fn,
        "Prompt 模式 / Prompt mode",
        [
            ("full", "full", "质量更高，负载更大 / Higher quality, larger payload"),
            ("compact", "compact", "负载更小，适合受限端点 / Smaller payload for size-limited endpoints"),
        ],
        default_index=1,
    )
    if prompt_mode != "full":
        argv.extend(["--prompt-mode", prompt_mode])

    _section(output_fn, "模型与环境 / Model and environment")
    model = _ask_optional_text(input_fn, "LLM 模型覆盖（可选） / LLM model override (optional)")
    if model:
        argv.extend(["--model", model])

    if _ask_yes_no(input_fn, "覆盖本次 LITELLM_BASE/LITELLM_KEY / Override LITELLM_BASE/LITELLM_KEY for this run", default=False):
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_semantic_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["semantic-cluster"]
    env_overrides: dict[str, str] = {}

    _section(output_fn, "基础参数 / Basic options")
    argv.extend(["--input", _ask_required_text(input_fn, "输入文件或目录 / Input file/dir")])
    output_path = _ask_optional_text(input_fn, "输出目录（可选） / Output directory (optional)")
    if output_path:
        argv.extend(["--output", output_path])

    limit = _ask_int(input_fn, "采样上限（0=全部） / Sample limit (0 = all)", default=0)
    if limit:
        argv.extend(["--limit", str(limit)])

    if _ask_yes_no(input_fn, "启用 resume 兼容性检查 / Enable resume compatibility check", default=False):
        argv.append("--resume")

    if not _ask_yes_no(input_fn, "导出代表窗口 / Export representative windows", default=True):
        argv.append("--no-export-representatives")

    provider = "local"
    if _ask_yes_no(input_fn, "配置语义高级参数 / Configure semantic advanced parameters", default=False):
        _section(output_fn, "语义高级参数 / Advanced semantic options")
        overrides = _ask_semantic_overrides(input_fn, output_fn)
        argv.extend(overrides["argv"])
        provider = overrides["provider"]

    _section(output_fn, "环境与附加参数 / Environment and extra flags")
    if provider == "api" and _ask_yes_no(
        input_fn,
        "provider=api，是否现在覆盖 LITELLM_BASE/LITELLM_KEY / Provider is api. Override LITELLM_BASE/LITELLM_KEY now",
        default=False,
    ):
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_filter_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["filter"]
    criteria_count = 0

    _section(output_fn, "输出设置 / Output settings")
    argv.extend(["--input", _ask_required_text(input_fn, "输入评分文件、镜像 JSONL 或目录 / Input scored file, mirrored JSONL, or dir")])
    output_path = _ask_optional_text(input_fn, "输出文件路径（可选） / Output file path (optional)")
    if output_path:
        argv.extend(["--output", output_path])

    output_format = _ask_choice(
        input_fn,
        output_fn,
        "输出格式 / Output format",
        [
            ("scored", "scored", "保留元数据与评分字段 / Keep metadata and scoring fields"),
            ("training", "training", "导出训练格式精简输出 / Export training-ready stripped output"),
        ],
        default_index=1,
    )
    if output_format != "scored":
        argv.extend(["--format", output_format])

    include_unscored = _ask_yes_no(input_fn, "包含未评分样本 / Include unscored samples", default=False)
    if include_unscored:
        argv.append("--include-unscored")

    _section(output_fn, "样本级筛选条件 / Sample-level criteria")
    value_min = _ask_optional_float(input_fn, "--value-min（可选） / --value-min (optional)")
    if value_min is not None:
        argv.extend(["--value-min", str(value_min)])
        criteria_count += 1

    selection_min = _ask_optional_float(input_fn, "--selection-min（可选） / --selection-min (optional)")
    if selection_min is not None:
        argv.extend(["--selection-min", str(selection_min)])
        criteria_count += 1

    difficulty = _ask_optional_text(input_fn, "难度列表（逗号分隔，可选） / Difficulty list (comma-separated, optional)")
    if difficulty:
        argv.extend(["--difficulty", difficulty])
        criteria_count += 1

    thinking_mode = _ask_choice(
        input_fn,
        output_fn,
        "思考模式过滤 / Thinking mode filter",
        [
            ("", "any", "不过滤 thinking_mode / No thinking_mode filter"),
            ("slow", "slow", "要求显式推理链路 / Require explicit reasoning traces"),
            ("fast", "fast", "要求简洁推理模式 / Require concise reasoning mode"),
        ],
        default_index=1,
    )
    if thinking_mode:
        argv.extend(["--thinking-mode", thinking_mode])
        criteria_count += 1

    include_tags = _ask_optional_text(input_fn, "包含标签（dim:tag，逗号分隔，可选） / Include tags (dim:tag, comma-separated, optional)")
    if include_tags:
        argv.extend(["--include-tags", include_tags])
        criteria_count += 1

    exclude_tags = _ask_optional_text(input_fn, "排除标签（dim:tag，逗号分隔，可选） / Exclude tags (dim:tag, comma-separated, optional)")
    if exclude_tags:
        argv.extend(["--exclude-tags", exclude_tags])
        criteria_count += 1

    if _ask_yes_no(input_fn, "排除继承标签 / Exclude inherited labels", default=False):
        argv.append("--exclude-inherited")
        criteria_count += 1

    verify_source = _ask_optional_text(input_fn, "来源路径过滤（可选） / Verify source path (optional)")
    if verify_source:
        argv.extend(["--verify-source", verify_source])
        criteria_count += 1

    _section(output_fn, "会话与轮次级筛选 / Conversation and turn-level criteria")
    if _ask_yes_no(input_fn, "配置会话级过滤条件 / Configure conversation-level filters", default=False):
        conv_value_min = _ask_optional_float(input_fn, "--conv-value-min（可选） / --conv-value-min (optional)")
        if conv_value_min is not None:
            argv.extend(["--conv-value-min", str(conv_value_min)])
            criteria_count += 1

        conv_selection_min = _ask_optional_float(input_fn, "--conv-selection-min（可选） / --conv-selection-min (optional)")
        if conv_selection_min is not None:
            argv.extend(["--conv-selection-min", str(conv_selection_min)])
            criteria_count += 1

        peak_complexity_min = _ask_optional_float(input_fn, "--peak-complexity-min（可选） / --peak-complexity-min (optional)")
        if peak_complexity_min is not None:
            argv.extend(["--peak-complexity-min", str(peak_complexity_min)])
            criteria_count += 1

        turn_count_min = _ask_optional_int(input_fn, "--turn-count-min（可选） / --turn-count-min (optional)")
        if turn_count_min is not None:
            argv.extend(["--turn-count-min", str(turn_count_min)])
            criteria_count += 1

        turn_count_max = _ask_optional_int(input_fn, "--turn-count-max（可选） / --turn-count-max (optional)")
        if turn_count_max is not None:
            argv.extend(["--turn-count-max", str(turn_count_max)])
            criteria_count += 1

    if _ask_yes_no(input_fn, "配置轮次级剪枝 / Configure turn-level pruning", default=False):
        turn_value_min = _ask_optional_float(input_fn, "--turn-value-min（可选） / --turn-value-min (optional)")
        if turn_value_min is not None:
            argv.extend(["--turn-value-min", str(turn_value_min)])
            criteria_count += 1

        turn_quality_min = _ask_optional_float(input_fn, "--turn-quality-min（可选） / --turn-quality-min (optional)")
        if turn_quality_min is not None:
            argv.extend(["--turn-quality-min", str(turn_quality_min)])
            criteria_count += 1

        correctness_min = _ask_optional_float(input_fn, "--correctness-min（可选） / --correctness-min (optional)")
        if correctness_min is not None:
            argv.extend(["--correctness-min", str(correctness_min)])
            criteria_count += 1

        max_pruned_ratio = _ask_optional_float(input_fn, "--max-pruned-ratio（可选） / --max-pruned-ratio (optional)")
        if max_pruned_ratio is not None:
            argv.extend(["--max-pruned-ratio", str(max_pruned_ratio)])

        if not _ask_yes_no(input_fn, "剪枝时保留首尾轮次 / Keep first/last turns when pruning", default=True):
            argv.append("--no-keep-first-last")

    if criteria_count == 0 and not include_unscored:
        _say(output_fn, "")
        _say(output_fn, "过滤至少需要一个条件，自动补充 --value-min / Filter requires at least one criterion. Setting --value-min now.")
        fallback = _ask_float(input_fn, "兜底 --value-min / Fallback --value-min", default=6.0)
        argv.extend(["--value-min", str(fallback)])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_recompute_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["recompute-stats"]
    _section(output_fn, "基础输入 / Basic input")
    argv.extend(["--input", _ask_required_text(input_fn, "输入文件或目录 / Input file/dir")])

    _section(output_fn, "重算范围 / Recompute scope")
    pass_num = _ask_choice(
        input_fn,
        output_fn,
        "阶段选择 / Pass selection",
        [
            ("both", "both", "重算一二阶段统计 / Recompute pass 1 and pass 2 stats"),
            ("1", "1", "仅重算一阶段统计 / Recompute pass 1 stats"),
            ("2", "2", "仅重算二阶段统计 / Recompute pass 2 stats"),
        ],
        default_index=1,
    )
    if pass_num != "both":
        argv.extend(["--pass", pass_num])

    output_dir = _ask_optional_text(input_fn, "输出目录覆盖（可选） / Output directory override (optional)")
    if output_dir:
        argv.extend(["--output", output_dir])

    _section(output_fn, "并发配置 / Parallelism")
    workers = _ask_maintenance_workers(input_fn, output_fn)
    if workers != DEFAULT_MAINTENANCE_WORKERS:
        argv.extend(["--workers", str(workers)])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_regenerate_dashboard_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["regenerate-dashboard"]
    _section(output_fn, "基础输入 / Basic input")
    argv.extend(["--input", _ask_required_text(input_fn, "运行目录 / Run directory")])

    _section(output_fn, "看板范围 / Dashboard scope")
    pass_num = _ask_choice(
        input_fn,
        output_fn,
        "阶段选择 / Pass selection",
        [
            ("both", "both", "重建两个看板 / Regenerate both dashboards"),
            ("1", "1", "仅重建一阶段看板 / Regenerate pass 1 dashboard"),
            ("2", "2", "仅重建二阶段看板 / Regenerate pass 2 dashboard"),
        ],
        default_index=1,
    )
    if pass_num != "both":
        argv.extend(["--pass", pass_num])

    if _ask_yes_no(input_fn, "生成后自动打开浏览器 / Open dashboards in browser", default=False):
        argv.append("--open")

    _section(output_fn, "并发配置 / Parallelism")
    workers = _ask_maintenance_workers(input_fn, output_fn)
    if workers != DEFAULT_MAINTENANCE_WORKERS:
        argv.extend(["--workers", str(workers)])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_refresh_rarity_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["refresh-rarity"]
    _section(output_fn, "基础输入 / Basic input")
    argv.extend(["--input", _ask_required_text(input_fn, "Scored 文件或运行目录 / Scored file or run directory")])

    _section(output_fn, "稀有度基准 / Rarity baseline")
    tag_stats = _ask_optional_text(input_fn, "全量 stats 路径（可选） / Global stats path (optional)")
    if tag_stats:
        argv.extend(["--tag-stats", tag_stats])

    mode = _ask_choice(
        input_fn,
        output_fn,
        "归一化模式 / Normalization mode",
        [
            ("absolute", "absolute（默认） / absolute (default)",
             "跨数据集绝对刻度 / Cross-dataset absolute scale"),
            ("percentile", "percentile", "批内百分位映射 / Within-batch percentile mapping"),
        ],
        default_index=1,
    )
    if mode != "absolute":
        argv.extend(["--mode", mode])

    output_dir = _ask_optional_text(input_fn, "输出目录覆盖（可选） / Output directory override (optional)")
    if output_dir:
        argv.extend(["--output", output_dir])

    _section(output_fn, "并发配置 / Parallelism")
    workers = _ask_maintenance_workers(input_fn, output_fn)
    if workers != DEFAULT_MAINTENANCE_WORKERS:
        argv.extend(["--workers", str(workers)])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_analyze_unmapped_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["analyze-unmapped"]
    _section(output_fn, "基础输入 / Basic input")
    argv.extend(["--input", _ask_required_text(
        input_fn,
        "已标注/已评分文件、运行目录、inline run，或 stats_labeling.json / "
        "Labeled/scored file, run dir, inline run, or stats_labeling.json",
    )])

    _section(output_fn, "查看选项 / Inspection options")
    dimension = _ask_optional_text(
        input_fn,
        "仅查看某个维度（可选，如 task/concept/unknown） / Only one dimension (optional, e.g. task/concept/unknown)",
    )
    if dimension:
        argv.extend(["--dimension", dimension])

    top = _ask_int(input_fn, "每个维度显示多少个标签 / Tags to show per dimension", default=20)
    if top != 20:
        argv.extend(["--top", str(top)])

    examples = _ask_int(input_fn, "每个标签显示多少条样例 / Examples to show per tag", default=2)
    if examples != 2:
        argv.extend(["--examples", str(examples)])

    if _ask_yes_no(input_fn, "仅从 stats 汇总读取 / Read summary from stats only", default=False):
        argv.append("--stats-only")

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _ask_maintenance_workers(input_fn: InputFn, output_fn: OutputFn) -> int:
    options = [
        (
            str(v),
            f"{v}" + ("（默认） / 8 (default)" if v == DEFAULT_MAINTENANCE_WORKERS else ""),
            f"最多并发 {v} 个文件/目录 / Up to {v} files/dirs in parallel",
        )
        for v in MAINTENANCE_WORKER_CHOICES
    ]
    picked = _ask_choice(
        input_fn,
        output_fn,
        "并发 worker 数 / Parallel workers",
        options,
        default_index=3,
    )
    return int(picked)


def _build_optimize_layout_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["optimize-layout"]
    _section(output_fn, "基础输入 / Basic input")
    argv.extend(["--input", _ask_required_text(input_fn, "运行目录或结果目录 / Run dir or output dir")])

    _section(output_fn, "执行模式 / Execution mode")
    if _ask_yes_no(input_fn, "直接应用变更（否则仅 dry-run） / Apply changes now (otherwise dry-run only)", default=False):
        argv.append("--apply")

    if _ask_yes_no(input_fn, "清理 legacy 别名文件 / Prune legacy alias files", default=False):
        argv.append("--prune-legacy")

    manifest = _ask_optional_text(input_fn, "manifest 路径（可选） / Manifest path (optional)")
    if manifest:
        argv.extend(["--manifest", manifest])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_export_semantic_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    del output_fn  # unused
    argv = ["export-semantic"]
    argv.extend(["--input", _ask_required_text(input_fn, "语义产物目录 / Semantic artifacts directory")])
    argv.extend(["--output", _ask_required_text(input_fn, "输出 JSONL 路径 / Output JSONL path")])
    if _ask_yes_no(input_fn, "包含非代表窗口 / Include non-representative windows", default=False):
        argv.append("--include-all")
    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_export_review_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["export-review"]
    _section(output_fn, "基础输入 / Basic input")
    argv.extend(["--input", _ask_required_text(input_fn, "输入已标注 JSON/镜像 JSONL/目录 / Input labeled JSON, mirrored JSONL, or directory")])
    argv.extend(["--output", _ask_required_text(input_fn, "输出 CSV/TSV 路径 / Output CSV/TSV path")])

    _section(output_fn, "导出选项 / Export options")
    monitor = _ask_optional_text(input_fn, "监控 JSONL 路径（可选） / Monitor JSONL path (optional)")
    if monitor:
        argv.extend(["--monitor", monitor])

    fmt = _ask_choice(
        input_fn,
        output_fn,
        "输出格式覆盖 / Output format override",
        [
            ("", "auto", "按扩展名自动判断 / Auto detect from --output extension"),
            ("csv", "csv", "强制 CSV / Force CSV"),
            ("tsv", "tsv", "强制 TSV / Force TSV"),
        ],
        default_index=1,
    )
    if fmt:
        argv.extend(["--format", fmt])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_dashboard_service_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["dashboard-service"]

    _section(output_fn, "服务操作 / Service actions")
    action = _ask_choice(
        input_fn,
        output_fn,
        "选择 dashboard 服务操作 / Choose a dashboard service action",
        [
            ("init", "初始化服务 / Init service", "创建或更新静态服务配置 / Create or update service config"),
            ("status", "检查状态 / Check status", "查看服务状态 / Inspect service health"),
            ("start", "启动服务 / Start service", "启动静态服务 / Start the static server"),
            ("restart", "重启服务 / Restart service", "重启静态服务 / Restart the static server"),
            ("stop", "停止服务 / Stop service", "停止静态服务 / Stop the static server"),
            ("list", "列出服务 / List services", "查看全部 dashboard 服务 / Show all configured services"),
            ("runs", "查看已发布 runs / List published runs", "查看某个服务已发布的结果 / Show published runs for a service"),
            ("register-run", "发布已有 run / Publish existing run", "将已有 run 注册到静态服务 / Publish an existing run directory"),
            ("set-default", "设置默认服务 / Set default service", "切换默认 dashboard 服务 / Change the default service"),
        ],
        default_index=2,
    )
    argv.append(action)

    if action == "init":
        name = _ask_text(input_fn, "服务名称 / Service name", required=False, default="default")
        argv.extend(["--name", name])
        default_root = str((Path.home() / "sft-label-dashboard").resolve())
        web_root = _ask_text(
            input_fn,
            "静态目录 / Shared web root",
            required=False,
            default=default_root,
        )
        argv.extend(["--web-root", web_root])
        host = _ask_text(input_fn, "监听地址 / Host", required=False, default="127.0.0.1")
        if host != "127.0.0.1":
            argv.extend(["--host", host])
        port = _ask_int(input_fn, "端口 / Port", default=8765)
        if port != 8765:
            argv.extend(["--port", str(port)])
        service_type = _ask_choice(
            input_fn,
            output_fn,
            "服务后端 / Service backend",
            [
                ("pm2", "pm2（默认） / pm2 (default)", "生产推荐，支持守护/重启 / Production-ready with daemon management"),
                ("builtin", "builtin", "内置 http.server，适合本地调试 / Built-in http.server for local debugging"),
            ],
            default_index=1,
        )
        if service_type != "pm2":
            argv.extend(["--service-type", service_type])
        public_base_url = _ask_optional_text(
            input_fn,
            "对外访问 URL（可选） / Public base URL (optional)",
        )
        if public_base_url:
            argv.extend(["--public-base-url", public_base_url])
        if service_type == "pm2":
            pm2_name = _ask_optional_text(
                input_fn,
                "PM2 进程名（可选） / PM2 process name (optional)",
            )
            if pm2_name:
                argv.extend(["--pm2-name", pm2_name])
    elif action in {"status", "start", "restart", "stop", "runs", "register-run"}:
        name = _ask_optional_text(input_fn, "服务名称（可选） / Service name (optional)")
        if name:
            argv.extend(["--name", name])
        if action == "register-run":
            argv.extend(["--run-dir", _ask_required_text(input_fn, "已有 run 目录 / Existing run directory")])
    elif action == "set-default":
        argv.extend(["--name", _ask_required_text(input_fn, "默认服务名称 / Default service name")])

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def build_dashboard_service_plan(
    input_fn: InputFn = interactive_input,
    output_fn: OutputFn = print,
) -> LaunchPlan:
    """Build one dashboard-service maintenance command plan."""
    return _with_workflow_key(_build_dashboard_service_plan(input_fn, output_fn), "dashboard-service")


def _ask_semantic_overrides(input_fn: InputFn, output_fn: OutputFn) -> dict:
    """Collect optional semantic flags."""
    argv: list[str] = []

    long_turn_threshold = _ask_optional_int(input_fn, "--semantic-long-turn-threshold（可选） / --semantic-long-turn-threshold (optional)")
    if long_turn_threshold is not None:
        argv.extend(["--semantic-long-turn-threshold", str(long_turn_threshold)])

    window_size = _ask_optional_int(input_fn, "--semantic-window-size（可选） / --semantic-window-size (optional)")
    if window_size is not None:
        argv.extend(["--semantic-window-size", str(window_size)])

    window_stride = _ask_optional_int(input_fn, "--semantic-window-stride（可选） / --semantic-window-stride (optional)")
    if window_stride is not None:
        argv.extend(["--semantic-window-stride", str(window_stride)])

    prefix_turns = _ask_optional_int(input_fn, "--semantic-pinned-prefix-max-turns（可选） / --semantic-pinned-prefix-max-turns (optional)")
    if prefix_turns is not None:
        argv.extend(["--semantic-pinned-prefix-max-turns", str(prefix_turns)])

    provider = _ask_choice(
        input_fn,
        output_fn,
        "向量提供方 / Embedding provider",
        [
            ("", "default(local)", "保持当前默认值 / Keep existing default"),
            ("local", "local", "本地 CPU hash embedding / CPU-first local hash embedding"),
            ("api", "api", "兼容 OpenAI 的 /v1/embeddings / OpenAI-compatible /v1/embeddings backend"),
        ],
        default_index=1,
    )
    effective_provider = provider or "local"
    if provider:
        argv.extend(["--semantic-embedding-provider", provider])

    model = _ask_optional_text(input_fn, "--semantic-embedding-model（可选） / --semantic-embedding-model (optional)")
    if model:
        argv.extend(["--semantic-embedding-model", model])

    embedding_dim = _ask_optional_int(input_fn, "--semantic-embedding-dim（可选） / --semantic-embedding-dim (optional)")
    if embedding_dim is not None:
        argv.extend(["--semantic-embedding-dim", str(embedding_dim)])

    embedding_batch = _ask_optional_int(input_fn, "--semantic-embedding-batch-size（可选） / --semantic-embedding-batch-size (optional)")
    if embedding_batch is not None:
        argv.extend(["--semantic-embedding-batch-size", str(embedding_batch)])

    embedding_workers = _ask_optional_int(input_fn, "--semantic-embedding-max-workers（可选） / --semantic-embedding-max-workers (optional)")
    if embedding_workers is not None:
        argv.extend(["--semantic-embedding-max-workers", str(embedding_workers)])

    semhash_bits = _ask_choice(
        input_fn,
        output_fn,
        "SemHash 位宽 / SemHash bits",
        [
            ("", "default(256)", "保持当前默认值 / Keep existing default"),
            ("64", "64", "更小 hash，内存更低 / Smaller hash, lower memory"),
            ("128", "128", "平衡配置 / Balanced hash width"),
            ("256", "256", "默认值 / Default"),
            ("512", "512", "更宽 hash，区分更强 / Wider hash, better separation"),
        ],
        default_index=1,
    )
    if semhash_bits:
        argv.extend(["--semantic-semhash-bits", semhash_bits])

    semhash_seed = _ask_optional_int(input_fn, "--semantic-semhash-seed（可选） / --semantic-semhash-seed (optional)")
    if semhash_seed is not None:
        argv.extend(["--semantic-semhash-seed", str(semhash_seed)])

    semhash_bands = _ask_optional_int(input_fn, "--semantic-semhash-bands（可选） / --semantic-semhash-bands (optional)")
    if semhash_bands is not None:
        argv.extend(["--semantic-semhash-bands", str(semhash_bands)])

    hamming_radius = _ask_optional_int(input_fn, "--semantic-hamming-radius（可选） / --semantic-hamming-radius (optional)")
    if hamming_radius is not None:
        argv.extend(["--semantic-hamming-radius", str(hamming_radius)])

    ann_top_k = _ask_optional_int(input_fn, "--semantic-ann-top-k（可选） / --semantic-ann-top-k (optional)")
    if ann_top_k is not None:
        argv.extend(["--semantic-ann-top-k", str(ann_top_k)])

    ann_threshold = _ask_optional_float(input_fn, "--semantic-ann-sim-threshold（可选） / --semantic-ann-sim-threshold (optional)")
    if ann_threshold is not None:
        argv.extend(["--semantic-ann-sim-threshold", str(ann_threshold)])

    output_prefix = _ask_optional_text(input_fn, "--semantic-output-prefix（可选） / --semantic-output-prefix (optional)")
    if output_prefix:
        argv.extend(["--semantic-output-prefix", output_prefix])

    return {"argv": argv, "provider": effective_provider}


def _ask_llm_env_overrides(input_fn: InputFn) -> dict[str, str]:
    base_default = os.environ.get("LITELLM_BASE", DEFAULT_LITELLM_BASE)
    key_default = os.environ.get("LITELLM_KEY", "")
    base = _ask_text(input_fn, "LITELLM_BASE", default=base_default)
    if key_default:
        raw = _read_input(
            input_fn,
            "LITELLM_KEY [回车=保持当前, '-'=清空 / enter=keep current, '-'=clear]: ",
        )
        if raw == "":
            key = key_default
        elif raw == "-":
            key = ""
        else:
            key = raw
    else:
        key = _ask_optional_text(input_fn, "LITELLM_KEY（可选） / LITELLM_KEY (optional)")
    return {"LITELLM_BASE": base, "LITELLM_KEY": key}


def _ask_workflow(input_fn: InputFn, output_fn: OutputFn) -> Workflow | None:
    group_order = []
    grouped: dict[str, list[Workflow]] = {}
    for wf in WORKFLOWS:
        if wf.group not in grouped:
            grouped[wf.group] = []
            group_order.append(wf.group)
        grouped[wf.group].append(wf)

    index_to_workflow: dict[int, Workflow] = {}
    idx = 1
    for group in group_order:
        _say(output_fn, f"[{localize_text(group)}]")
        for wf in grouped[group]:
            index_to_workflow[idx] = wf
            label = localize_text(wf.label)
            desc = localize_text(wf.description)
            _say(output_fn, f"{idx}. {label} - {desc}")
            idx += 1
        _say(output_fn, "")
    _say(output_fn, "0. 取消 / 0. Cancel")
    _say(output_fn, "")

    max_choice = len(WORKFLOWS)
    while True:
        raw = _read_input(
            input_fn,
            f"请选择任务编号 [0-{max_choice}, 默认 1]： / Select workflow number [0-{max_choice}, default 1]: ",
            allow_back=False,
        )
        if is_back_token(raw):
            return None
        if raw == "":
            raw = "1"
        if not raw.isdigit():
            _say(output_fn, "请输入数字 / Enter a number.")
            continue
        num = int(raw)
        if num == 0:
            return None
        if num in index_to_workflow:
            return index_to_workflow[num]
        _say(output_fn, "选择超出范围 / Selection out of range.")


def _ask_extra_flags(input_fn: InputFn) -> list[str]:
    while True:
        raw = _read_input(
            input_fn,
            "附加原始 CLI 参数（可选，例如 --foo 1 --bar x）： / "
            "Extra raw CLI flags (optional, e.g. --foo 1 --bar x): "
        )
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError as e:
            print(_msg(f"无效 shell 风格参数：{e}", f"Invalid shell-style flags: {e}"))


def _ask_required_text(input_fn: InputFn, prompt: str) -> str:
    return _ask_text(input_fn, prompt, required=True)


def _ask_optional_text(input_fn: InputFn, prompt: str) -> str:
    return _ask_text(input_fn, prompt, required=False)


def _ask_text(
    input_fn: InputFn,
    prompt: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
    while True:
        prompt_text = localize_text(prompt)
        suffix = f" [{default}]" if default is not None else ""
        value = _read_input(input_fn, f"{prompt_text}{suffix}{_msg('：', ': ')}")
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print(_msg("该字段必填。", "This field is required."))


def _ask_int(input_fn: InputFn, prompt: str, *, default: int) -> int:
    while True:
        prompt_text = localize_text(prompt)
        raw = _read_input(input_fn, f"{prompt_text} [{default}]{_msg('：', ': ')}")
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print(_msg("请输入整数。", "Enter an integer."))


def _ask_optional_int(input_fn: InputFn, prompt: str) -> int | None:
    while True:
        prompt_text = localize_text(prompt)
        raw = _read_input(input_fn, f"{prompt_text}{_msg('：', ': ')}")
        if raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            print(_msg("请输入整数，或留空。", "Enter an integer or leave blank."))


def _ask_float(input_fn: InputFn, prompt: str, *, default: float) -> float:
    while True:
        prompt_text = localize_text(prompt)
        raw = _read_input(input_fn, f"{prompt_text} [{default}]{_msg('：', ': ')}")
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print(_msg("请输入数字。", "Enter a number."))


def _ask_optional_float(input_fn: InputFn, prompt: str) -> float | None:
    while True:
        prompt_text = localize_text(prompt)
        raw = _read_input(input_fn, f"{prompt_text}{_msg('：', ': ')}")
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print(_msg("请输入数字，或留空。", "Enter a number or leave blank."))


def _ask_yes_no(input_fn: InputFn, prompt: str, *, default: bool) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        prompt_text = localize_text(prompt)
        raw = _read_input(input_fn, f"{prompt_text} [{default_hint}]{_msg('：', ': ')}").lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(_msg("请输入 y 或 n。", "Please enter y or n."))


def _ask_choice(
    input_fn: InputFn,
    output_fn: OutputFn,
    title: str,
    options: list[tuple[str, str, str]],
    *,
    default_index: int,
) -> str:
    _say(output_fn, "")
    _say(output_fn, title)
    _say(output_fn, "0. 返回上一级 / 0. Back")
    for idx, (_, label, desc) in enumerate(options, start=1):
        _say(output_fn, f"{idx}. {localize_text(label)} - {localize_text(desc)}")
    _say(output_fn, "")

    while True:
        raw = _read_input(
            input_fn,
            f"请选择 [0-{len(options)}，默认 {default_index}]： / "
            f"Select [0-{len(options)}, default {default_index}]: ",
        )
        if raw == "0":
            raise BackRequested
        if raw == "":
            raw = str(default_index)
        if not raw.isdigit():
            _say(output_fn, "请输入数字 / Enter a number.")
            continue
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1][0]
        _say(output_fn, "选择超出范围 / Selection out of range.")


def _say(output_fn: OutputFn, text: str):
    output_fn(localize_text(text))


def _section(output_fn: OutputFn, title: str):
    _say(output_fn, "")
    _say(output_fn, SECTION_DIVIDER)
    _say(output_fn, title)
    _say(output_fn, SECTION_DIVIDER)
