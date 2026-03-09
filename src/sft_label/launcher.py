"""Interactive launcher for sft-label CLI workflows."""

from __future__ import annotations

import os
import re
import select
import shlex
import sys
from dataclasses import dataclass, field
from typing import Callable


DEFAULT_LITELLM_BASE = "http://localhost:4000/v1"
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
BACK_TOKENS = {
    "b", "back", "/b", ":b",
    "返回", "上一步", "上一步", "上一级",
}


class BackRequested(Exception):
    """Raised when user asks to return to previous level."""


@dataclass
class LaunchPlan:
    """Executable launch plan built from interactive prompts."""

    argv: list[str]
    env_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Workflow:
    key: str
    label: str
    description: str
    group: str


WORKFLOWS = [
    Workflow(
        key="run-pass1",
        label="一阶段标注 / Pass 1 labeling",
        description="仅运行标签标注 / Run tag labeling only",
        group="流水线 Pipeline",
    ),
    Workflow(
        key="run-pass1-pass2",
        label="一阶段 + 二阶段 / Pass 1 + Pass 2",
        description="先标注后打分 / Labeling followed by value scoring",
        group="流水线 Pipeline",
    ),
    Workflow(
        key="run-pass1-pass2-semantic",
        label="一阶段 + 二阶段 + 四阶段 / Pass 1 + Pass 2 + Pass 4",
        description="标注、打分、语义聚类 / Labeling, scoring, then semantic clustering",
        group="流水线 Pipeline",
    ),
    Workflow(
        key="score",
        label="仅二阶段打分 / Pass 2 scoring only",
        description="对已标注数据评分 / Score pre-labeled data",
        group="流水线 Pipeline",
    ),
    Workflow(
        key="semantic",
        label="四阶段语义聚类 / Pass 4 semantic clustering",
        description="运行 SemHash + ANN 聚类 / Run SemHash + ANN clustering",
        group="流水线 Pipeline",
    ),
    Workflow(
        key="filter",
        label="三阶段过滤 / Pass 3 filtering",
        description="筛选高价值训练样本 / Filter scored data for training selection",
        group="数据整理 Data Curation",
    ),
    Workflow(
        key="recompute-stats",
        label="重算统计 / Recompute stats",
        description="不调用 LLM 重新计算统计 / Rebuild stats without LLM calls",
        group="维护 Maintenance",
    ),
    Workflow(
        key="regenerate-dashboard",
        label="重建看板 / Regenerate dashboards",
        description="用已有统计重建 HTML 看板 / Rebuild HTML dashboards from existing stats",
        group="维护 Maintenance",
    ),
    Workflow(
        key="export-semantic",
        label="导出语义窗口 / Export semantic rows",
        description="导出代表窗口或全部窗口 / Export representative/all semantic windows",
        group="导出 Export",
    ),
    Workflow(
        key="export-review",
        label="导出审核 CSV/TSV / Export review CSV/TSV",
        description="导出人工审核表 / Flatten labeled data for manual review",
        group="导出 Export",
    ),
    Workflow(
        key="validate",
        label="校验 taxonomy / Validate taxonomy",
        description="校验 taxonomy 文件与一致性 / Validate taxonomy files and consistency",
        group="维护 Maintenance",
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

    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        tty.setraw(fd)
        while True:
            ch = os.read(fd, 1)
            if not ch:
                raise EOFError
            c = ch.decode("utf-8", errors="ignore")

            if c in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                break

            if c == "\x03":
                raise KeyboardInterrupt

            if c == "\x04":
                if not chars:
                    raise EOFError
                continue

            if c in ("\x7f", "\b"):
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue

            if c == "\x1b":
                _consume_escape_sequence(fd)
                continue

            if c and ord(c) >= 32:
                chars.append(c)
                sys.stdout.write(c)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    return "".join(chars)


def is_back_token(value: str) -> bool:
    """Check whether an input value means 'go back'."""
    if value is None:
        return False
    token = str(value).strip().lower()
    return token in BACK_TOKENS


def _read_input(input_fn: InputFn, prompt: str, *, allow_back: bool = True) -> str:
    """Read input and sanitize terminal control sequences."""
    value, had_control = sanitize_prompt_input(input_fn(prompt))
    if had_control:
        print(
            "检测到方向键/控制字符输入，已忽略 / "
            "Detected arrow/control key input and ignored."
        )
    if allow_back and is_back_token(value):
        raise BackRequested
    return value


def build_launch_plan(
    input_fn: InputFn = interactive_input,
    output_fn: OutputFn = print,
) -> LaunchPlan | None:
    """Prompt for workflow/options and return executable launch plan."""
    _say(output_fn, "交互式任务启动器 / Interactive task launcher")
    _say(output_fn, "请选择任务，并只填写必要参数 / Choose a workflow and fill only required options.")
    _say(output_fn, "输入 b/back 可返回上一层，输入 0 可取消 / Type b/back to go back, 0 to cancel.")
    _say(output_fn, "")

    while True:
        wf = _ask_workflow(input_fn, output_fn)
        if wf is None:
            return None

        try:
            if wf.key == "run-pass1":
                return _build_run_plan(input_fn, output_fn, chain_score=False, chain_semantic=False)
            if wf.key == "run-pass1-pass2":
                return _build_run_plan(input_fn, output_fn, chain_score=True, chain_semantic=False)
            if wf.key == "run-pass1-pass2-semantic":
                return _build_run_plan(input_fn, output_fn, chain_score=True, chain_semantic=True)
            if wf.key == "score":
                return _build_score_plan(input_fn, output_fn)
            if wf.key == "semantic":
                return _build_semantic_plan(input_fn, output_fn)
            if wf.key == "filter":
                return _build_filter_plan(input_fn, output_fn)
            if wf.key == "recompute-stats":
                return _build_recompute_plan(input_fn, output_fn)
            if wf.key == "regenerate-dashboard":
                return _build_regenerate_dashboard_plan(input_fn, output_fn)
            if wf.key == "export-semantic":
                return _build_export_semantic_plan(input_fn, output_fn)
            if wf.key == "export-review":
                return _build_export_review_plan(input_fn, output_fn)
            if wf.key == "validate":
                return LaunchPlan(argv=["validate"])
            raise ValueError(f"Unknown workflow key: {wf.key}")
        except BackRequested:
            _say(output_fn, "")
            _say(output_fn, "已返回任务选择 / Returned to workflow selection.")
            _say(output_fn, "")


def _build_run_plan(
    input_fn: InputFn,
    output_fn: OutputFn,
    *,
    chain_score: bool,
    chain_semantic: bool,
) -> LaunchPlan:
    argv = ["run"]
    env_overrides: dict[str, str] = {}

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

    output_path = _ask_optional_text(input_fn, "输出目录（可选） / Output directory (optional)")
    if output_path:
        argv.extend(["--output", output_path])

    limit = _ask_int(input_fn, "每个文件采样上限（0=全部） / Sample limit per file (0 = all)", default=0)
    if limit:
        argv.extend(["--limit", str(limit)])

    if _ask_yes_no(input_fn, "处理前打乱样本 / Shuffle samples before processing", default=False):
        argv.append("--shuffle")

    if not _ask_yes_no(input_fn, "启用仲裁补跑 / Enable arbitration pass", default=True):
        argv.append("--no-arbitration")

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
        argv.append("--score")
        tag_stats = _ask_optional_text(input_fn, "稀有度统计文件路径（可选） / Tag stats path for rarity (optional)")
        if tag_stats:
            argv.extend(["--tag-stats", tag_stats])

    if chain_semantic:
        argv.append("--semantic-cluster")

    if _ask_yes_no(input_fn, "覆盖本次 LITELLM_BASE / LITELLM_KEY / Override LITELLM_BASE / LITELLM_KEY for this run", default=False):
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_score_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["score", "--input", _ask_required_text(input_fn, "输入已标注文件或目录 / Input labeled file/dir")]
    env_overrides: dict[str, str] = {}

    tag_stats = _ask_optional_text(input_fn, "稀有度统计文件路径（可选） / Tag stats path for rarity (optional)")
    if tag_stats:
        argv.extend(["--tag-stats", tag_stats])

    limit = _ask_int(input_fn, "采样上限（0=全部） / Sample limit (0 = all)", default=0)
    if limit:
        argv.extend(["--limit", str(limit)])

    if _ask_yes_no(input_fn, "续跑评分并跳过已存在样本 / Resume scoring and skip existing samples", default=False):
        argv.append("--resume")

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

    if _ask_yes_no(input_fn, "覆盖本次 LITELLM_BASE / LITELLM_KEY / Override LITELLM_BASE / LITELLM_KEY for this run", default=False):
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_semantic_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["semantic-cluster", "--input", _ask_required_text(input_fn, "输入文件或目录 / Input file/dir")]
    env_overrides: dict[str, str] = {}

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
        overrides = _ask_semantic_overrides(input_fn, output_fn)
        argv.extend(overrides["argv"])
        provider = overrides["provider"]

    if provider == "api" and _ask_yes_no(
        input_fn,
        "provider=api，是否现在覆盖 LITELLM_BASE / LITELLM_KEY / Provider is api. Override LITELLM_BASE / LITELLM_KEY now",
        default=False,
    ):
        env_overrides = _ask_llm_env_overrides(input_fn)

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv, env_overrides=env_overrides)


def _build_filter_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["filter", "--input", _ask_required_text(input_fn, "输入评分文件或目录 / Input scored file/dir")]
    criteria_count = 0

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
    argv = ["recompute-stats", "--input", _ask_required_text(input_fn, "输入文件或目录 / Input file/dir")]
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

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_regenerate_dashboard_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = ["regenerate-dashboard", "--input", _ask_required_text(input_fn, "运行目录 / Run directory")]
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

    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_export_semantic_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    del output_fn  # unused
    argv = [
        "export-semantic",
        "--input",
        _ask_required_text(input_fn, "语义产物目录 / Semantic artifacts directory"),
        "--output",
        _ask_required_text(input_fn, "输出 JSONL 路径 / Output JSONL path"),
    ]
    if _ask_yes_no(input_fn, "包含非代表窗口 / Include non-representative windows", default=False):
        argv.append("--include-all")
    argv.extend(_ask_extra_flags(input_fn))
    return LaunchPlan(argv=argv)


def _build_export_review_plan(input_fn: InputFn, output_fn: OutputFn) -> LaunchPlan:
    argv = [
        "export-review",
        "--input",
        _ask_required_text(input_fn, "输入已标注 JSON 路径 / Input labeled JSON path"),
        "--output",
        _ask_required_text(input_fn, "输出 CSV/TSV 路径 / Output CSV/TSV path"),
    ]

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
        _say(output_fn, f"[{group}]")
        for wf in grouped[group]:
            index_to_workflow[idx] = wf
            _say(output_fn, f"  {idx}. {wf.label} - {wf.description}")
            idx += 1
        _say(output_fn, "")
    _say(output_fn, "  0. 取消 / Cancel")
    _say(output_fn, "")

    max_choice = len(WORKFLOWS)
    while True:
        raw = _read_input(
            input_fn,
            f"Select workflow number [0-{max_choice}, default 1]: ",
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
            "附加原始 CLI 参数（可选，例如 --foo 1 --bar x） / Extra raw CLI flags (optional, e.g. --foo 1 --bar x): "
        )
        if not raw:
            return []
        try:
            return shlex.split(raw)
        except ValueError as e:
            print(f"无效 shell 风格参数 / Invalid shell-style flags: {e}")


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
        suffix = f" [{default}]" if default is not None else ""
        value = _read_input(input_fn, f"{prompt}{suffix}: ")
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("该字段必填 / This field is required.")


def _ask_int(input_fn: InputFn, prompt: str, *, default: int) -> int:
    while True:
        raw = _read_input(input_fn, f"{prompt} [{default}]: ")
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("请输入整数 / Enter an integer.")


def _ask_optional_int(input_fn: InputFn, prompt: str) -> int | None:
    while True:
        raw = _read_input(input_fn, f"{prompt}: ")
        if raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            print("请输入整数，或留空 / Enter an integer or leave blank.")


def _ask_float(input_fn: InputFn, prompt: str, *, default: float) -> float:
    while True:
        raw = _read_input(input_fn, f"{prompt} [{default}]: ")
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("请输入数字 / Enter a number.")


def _ask_optional_float(input_fn: InputFn, prompt: str) -> float | None:
    while True:
        raw = _read_input(input_fn, f"{prompt}: ")
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print("请输入数字，或留空 / Enter a number or leave blank.")


def _ask_yes_no(input_fn: InputFn, prompt: str, *, default: bool) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        raw = _read_input(input_fn, f"{prompt} [{default_hint}]: ").lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("请输入 y 或 n / Enter y or n.")


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
    _say(output_fn, "  0. 返回上一级 / Back")
    for idx, (_, label, desc) in enumerate(options, start=1):
        _say(output_fn, f"  {idx}. {label} - {desc}")
    _say(output_fn, "")

    while True:
        raw = _read_input(
            input_fn,
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
    output_fn(text)
