"""Tests for interactive launcher command planning."""

from __future__ import annotations

from sft_label.launcher import (
    _decode_utf8_input_byte,
    build_launch_plan,
    format_command,
    format_env_prefix,
    set_language,
    sanitize_prompt_input,
)
from sft_label.cli import build_parser, cmd_start


class StubIO:
    def __init__(self, answers):
        self._answers = iter(answers)
        self.outputs = []

    def input(self, prompt):
        self.outputs.append(prompt)
        try:
            return next(self._answers)
        except StopIteration as exc:
            raise AssertionError(f"Unexpected prompt: {prompt}") from exc

    def output(self, text):
        self.outputs.append(text)


def test_build_run_pass1_pass2_semantic_plan():
    io = StubIO(
        [
            "3",          # workflow: pass1+pass2+pass4
            "1",          # run mode: new
            "data.json",  # --input
            "",           # --output
            "",           # inline mode (refresh)
            "",           # --limit (default 0)
            "",           # shuffle (default no)
            "",           # arbitration (default yes)
            "",           # prompt mode (full)
            "",           # model
            "",           # tag-stats
            "",           # rarity mode (default absolute)
            "n",          # env override
            "",           # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "run",
        "--input",
        "data.json",
        "--score",
        "--semantic-cluster",
    ]
    assert plan.env_overrides == {}


def test_build_score_plan_with_llm_env_override():
    io = StubIO(
        [
            "4",                      # workflow: score
            "labeled.json",           # --input
            "",                       # --tag-stats
            "",                       # rarity mode (default absolute)
            "",                       # --limit
            "",                       # --resume
            "",                       # prompt mode
            "",                       # model
            "y",                      # env override
            "http://proxy/v1",        # LITELLM_BASE
            "secret",                 # LITELLM_KEY
            "",                       # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == ["score", "--input", "labeled.json"]
    assert plan.env_overrides == {
        "LITELLM_BASE": "http://proxy/v1",
        "LITELLM_KEY": "secret",
    }


def test_build_filter_plan_enforces_at_least_one_criterion():
    io = StubIO(
        [
            "6",            # workflow: filter
            "scored.json",  # --input
            "",             # --output
            "",             # --format (scored)
            "",             # include-unscored
            "",             # --value-min
            "",             # --selection-min
            "",             # --difficulty
            "",             # thinking mode
            "",             # include-tags
            "",             # exclude-tags
            "",             # exclude-inherited
            "",             # verify-source
            "",             # conversation filters
            "",             # turn-level pruning
            "",             # fallback value-min -> default 6.0
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == ["filter", "--input", "scored.json", "--value-min", "6.0"]


def test_build_semantic_plan_api_provider_supports_env_override():
    io = StubIO(
        [
            "5",               # workflow: semantic
            "run_dir",         # --input
            "",                # --output
            "",                # --limit
            "",                # --resume
            "",                # export representatives
            "y",               # configure advanced semantic params
            "",                # --semantic-long-turn-threshold
            "",                # --semantic-window-size
            "",                # --semantic-window-stride
            "",                # --semantic-pinned-prefix-max-turns
            "3",               # provider: api
            "",                # --semantic-embedding-model
            "",                # --semantic-embedding-dim
            "",                # --semantic-embedding-batch-size
            "",                # --semantic-embedding-max-workers
            "",                # semhash bits
            "",                # --semantic-semhash-seed
            "",                # --semantic-semhash-bands
            "",                # --semantic-hamming-radius
            "",                # --semantic-ann-top-k
            "",                # --semantic-ann-sim-threshold
            "",                # --semantic-output-prefix
            "y",               # env override for api provider
            "http://embed/v1", # LITELLM_BASE
            "embed-key",       # LITELLM_KEY
            "",                # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "semantic-cluster",
        "--input",
        "run_dir",
        "--semantic-embedding-provider",
        "api",
    ]
    assert plan.env_overrides == {
        "LITELLM_BASE": "http://embed/v1",
        "LITELLM_KEY": "embed-key",
    }


def test_build_dashboard_service_init_plan():
    io = StubIO(
        [
            "15",                    # workflow: dashboard maintenance
            "1",                     # action: init
            "",                      # name -> default
            "/srv/sft-label-dashboard",  # web root
            "",                      # host default
            "",                      # port default
            "",                      # service-type default pm2
            "https://dash.example.com",  # public base url
            "sft-label-dashboard-prod",  # pm2 name
            "",                      # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "dashboard-service",
        "init",
        "--name",
        "default",
        "--web-root",
        "/srv/sft-label-dashboard",
        "--public-base-url",
        "https://dash.example.com",
        "--pm2-name",
        "sft-label-dashboard-prod",
    ]


def test_build_dashboard_service_runs_plan():
    io = StubIO(
        [
            "15",     # workflow: dashboard maintenance
            "7",      # action: runs
            "prod",   # service name
            "",       # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)

    assert plan is not None
    assert plan.argv == [
        "dashboard-service",
        "runs",
        "--name",
        "prod",
    ]


def test_build_launch_plan_can_cancel():
    io = StubIO(["0"])
    assert build_launch_plan(input_fn=io.input, output_fn=io.output) is None


def test_can_back_to_workflow_selection_with_zero_in_submenu():
    io = StubIO(
        [
            "1",  # select run-pass1 workflow
            "0",  # back from run mode selection
            "0",  # cancel at workflow menu
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is None
    assert any("已返回任务选择" in str(item) for item in io.outputs)


def test_can_back_to_workflow_selection_with_back_token():
    io = StubIO(
        [
            "1",          # select run-pass1 workflow
            "1",          # run mode: new
            "b",          # back from required input prompt
            "0",          # cancel at workflow menu
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is None
    assert any("已返回任务选择" in str(item) for item in io.outputs)


def test_format_helpers():
    assert format_command(["score", "--input", "a b.json"]) == "sft-label score --input 'a b.json'"
    assert (
        format_env_prefix({"LITELLM_KEY": "abc", "LITELLM_BASE": "http://x/v1"})
        == "LITELLM_BASE=http://x/v1 LITELLM_KEY=abc"
    )


def test_sanitize_prompt_input_removes_arrow_escape_sequences():
    clean, had_control = sanitize_prompt_input("\x1b[A")
    assert clean == ""
    assert had_control is True

    clean2, had_control2 = sanitize_prompt_input("abc\x1b[D")
    assert clean2 == "abc"
    assert had_control2 is True


def test_decode_utf8_input_byte_keeps_multibyte_chars():
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    parts = []
    for b in "中文路径".encode("utf-8"):
        parts.append(_decode_utf8_input_byte(decoder, bytes([b])))
    assert "".join(parts) == "中文路径"


def test_all_workflows_generate_parseable_argv():
    parser = build_parser()
    workflow_answers = {
        # 1. run-pass1
        1: ["1", "data.json", "", "", "", "", "", "", "", "n", ""],
        # 2. run-pass1-pass2
        2: ["1", "data.json", "", "", "", "", "", "", "", "stats.json", "", "n", ""],
        # 3. run-pass1-pass2-semantic
        3: ["1", "data.json", "", "", "", "", "", "", "", "", "", "n", ""],
        # 4. score
        4: ["labeled.json", "", "", "", "", "", "", "n", ""],
        # 5. semantic
        5: ["run_dir", "", "", "", "", "", ""],
        # 6. filter
        6: ["scored.json", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        # 7. recompute-stats
        7: ["run_dir/", "", "", "", ""],
        # 8. refresh-rarity
        8: ["run_dir/", "", "", "", "", ""],
        # 9. regenerate-dashboard
        9: ["run_dir/", "", "", "", ""],
        # 10. validate
        10: [],
        # 11. analyze-unmapped
        11: ["run_dir/", "", "", "", "", ""],
        # 12. optimize-layout
        12: ["run_dir/", "", "", "", ""],
        # 13. export-semantic
        13: ["run_dir/", "out.jsonl", "", ""],
        # 14. export-review
        14: ["labeled.json", "review.csv", "", "", ""],
        # 15. dashboard-service maintenance
        15: ["2", "", ""],  # action=status, name default, extra flags
    }

    for wf_num, answers in workflow_answers.items():
        io = StubIO([str(wf_num)] + answers)
        plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
        assert plan is not None
        parsed = parser.parse_args(plan.argv)
        assert parsed.command is not None


def test_llm_key_can_be_cleared_from_existing_env(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE", "http://example/v1")
    monkeypatch.setenv("LITELLM_KEY", "existing-key")

    io = StubIO(
        [
            "4",            # score
            "labeled.json", # --input
            "",             # --tag-stats
            "",             # rarity mode (default absolute)
            "",             # --limit
            "",             # --resume
            "",             # prompt mode
            "",             # model
            "y",            # override env
            "",             # keep LITELLM_BASE default
            "-",            # clear key
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.env_overrides["LITELLM_BASE"] == "http://example/v1"
    assert plan.env_overrides["LITELLM_KEY"] == ""


def test_required_text_prompt_ignores_arrow_key_input(capsys):
    io = StubIO(
        [
            "1",          # workflow: run-pass1
            "1",          # run mode: new
            "\x1b[A",     # arrow key input on required --input field
            "data.json",  # actual --input
            "",           # --output
            "",           # inline mode
            "",           # --limit
            "",           # --shuffle
            "",           # --arbitration
            "",           # --prompt mode
            "",           # --model
            "n",          # env override
            "",           # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    captured = capsys.readouterr()

    assert plan is not None
    assert plan.argv[:3] == ["run", "--input", "data.json"]
    assert "检测到方向键/控制字符输入，已忽略" in captured.out


def test_build_run_plan_can_set_migrate_mode():
    io = StubIO(
        [
            "1",             # workflow: run-pass1
            "1",             # run mode: new
            "dataset",       # --input
            "",              # --output
            "3",             # inline mode: migrate
            "old-run",       # --migrate-from
            "",              # --limit
            "",              # --shuffle
            "",              # --arbitration
            "",              # --prompt mode
            "",              # --model
            "n",             # env override
            "",              # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv[:7] == [
        "run",
        "--input",
        "dataset",
        "--mode",
        "migrate",
        "--migrate-from",
        "old-run",
    ]


def test_build_run_plan_can_set_recompute_mode():
    io = StubIO(
        [
            "1",        # workflow: run-pass1
            "1",        # run mode: new
            "dataset",  # --input
            "",         # --output
            "4",        # inline mode: recompute
            "",         # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "run",
        "--input",
        "dataset",
        "--mode",
        "recompute",
    ]


def test_build_run_resume_plan_skips_output_prompt():
    io = StubIO(
        [
            "1",        # workflow: run-pass1
            "2",        # run mode: resume
            "run-dir",  # --resume
            "",         # also set --input? no
            "",         # --limit
            "",         # --shuffle
            "",         # --arbitration
            "",         # --prompt mode
            "",         # --model
            "n",        # env override
            "",         # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "run",
        "--resume",
        "run-dir",
    ]
    assert "--output" not in plan.argv
    assert not any("输出目录（可选）" in str(item) for item in io.outputs)


def test_build_launch_plan_can_render_english():
    io = StubIO(["10"])
    try:
        plan = build_launch_plan(input_fn=io.input, output_fn=io.output, language="en")
        assert plan is not None
        assert plan.argv == ["validate"]
        assert any("Interactive task launcher" in str(item) for item in io.outputs)
    finally:
        set_language("zh")


def test_chinese_prompt_uses_fullwidth_colon_without_english_suffix():
    io = StubIO(["0"])
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output, language="zh")
    assert plan is None
    assert any("请选择任务编号 [0-15, 默认 1]：" in str(item) for item in io.outputs)
    assert not any("Select workflow number" in str(item) for item in io.outputs)


def test_chinese_llm_override_prompt_keeps_full_key_name():
    io = StubIO(
        [
            "4",            # workflow: score
            "labeled.json", # --input
            "",             # --tag-stats
            "",             # rarity mode (default absolute)
            "",             # --limit
            "",             # --resume
            "",             # prompt mode
            "",             # model
            "n",            # env override
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output, language="zh")
    assert plan is not None
    assert any("覆盖本次 LITELLM_BASE/LITELLM_KEY [y/N]：" in str(item) for item in io.outputs)
    assert not any("覆盖本次 LITELLM_BASE / LITELLM_KEY" in str(item) for item in io.outputs)


def test_build_optimize_layout_plan():
    io = StubIO(
        [
            "12",          # workflow: optimize-layout
            "run_dir",     # --input
            "y",           # --apply
            "y",           # --prune-legacy
            "plan.json",   # --manifest
            "",            # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "optimize-layout",
        "--input",
        "run_dir",
        "--apply",
        "--prune-legacy",
        "--manifest",
        "plan.json",
    ]


def test_build_analyze_unmapped_plan():
    io = StubIO(
        [
            "11",          # workflow: analyze-unmapped
            "run_dir",     # --input
            "task",        # --dimension
            "10",          # --top
            "1",           # --examples
            "y",           # --stats-only
            "",            # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "analyze-unmapped",
        "--input",
        "run_dir",
        "--dimension",
        "task",
        "--top",
        "10",
        "--examples",
        "1",
        "--stats-only",
    ]


def test_build_score_plan_can_set_percentile_rarity_mode():
    io = StubIO(
        [
            "4",            # workflow: score
            "labeled.json", # --input
            "",             # --tag-stats
            "2",            # rarity mode: percentile
            "",             # --limit
            "",             # --resume
            "",             # prompt mode
            "",             # model
            "n",            # env override
            "",             # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "score",
        "--input",
        "labeled.json",
        "--rarity-mode",
        "percentile",
    ]


def test_build_recompute_plan_can_set_workers():
    io = StubIO(
        [
            "7",        # workflow: recompute-stats
            "run_dir",  # --input
            "",         # pass selection -> both
            "",         # --output
            "2",        # workers -> 4
            "",         # extra flags
        ]
    )
    plan = build_launch_plan(input_fn=io.input, output_fn=io.output)
    assert plan is not None
    assert plan.argv == [
        "recompute-stats",
        "--input",
        "run_dir",
        "--workers",
        "4",
    ]



def test_cmd_start_can_auto_publish_dashboard(monkeypatch, capsys, tmp_path):
    from sft_label.launcher import LaunchPlan
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr("sft_label.launcher.build_launch_plan", lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json"]))

    answers = iter(["", "y", ""])  # confirm, auto-publish yes, restart=no(default)
    monkeypatch.setattr("sft_label.launcher.interactive_input", lambda prompt: next(answers), raising=False)

    service = DashboardServiceConfig(name="default", web_root=str(tmp_path / "web"), host="127.0.0.1", port=8765)
    store = DashboardServiceStore(default_service="default", services={"default": service})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr("sft_label.cli.dashboard_service_status", lambda svc: {"state": "running", "reachable": True, "url": svc.base_url()}, raising=False)

    published = {
        "run_id": "demo_run",
        "dashboards": {
            "labeling": {"url": "http://127.0.0.1:8765/runs/demo_run/dashboard_labeling_demo.html"},
            "scoring": {"url": "http://127.0.0.1:8765/runs/demo_run/dashboard_scoring_demo.html"},
        },
    }
    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", lambda svc, run_dir, config_path=None: published, raising=False)
    monkeypatch.setattr("sft_label.cli.dispatch_command", lambda args, parser: {"run_dir": str(run_dir)}, raising=False)

    args = parser.parse_args(["start"])
    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert "dashboard_labeling_demo.html" in out
    assert "dashboard_scoring_demo.html" in out


def test_cmd_start_can_choose_named_dashboard_service(monkeypatch, capsys, tmp_path):
    from sft_label.launcher import LaunchPlan
    from sft_label.dashboard_service import DashboardServiceConfig, DashboardServiceStore

    parser = build_parser()
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    monkeypatch.setattr("sft_label.launcher.build_launch_plan", lambda **kwargs: LaunchPlan(argv=["run", "--input", "data.json"]))

    answers = iter(["", "y", "2", ""])  # confirm, auto publish, choose 2nd service, don't restart
    monkeypatch.setattr("sft_label.launcher.interactive_input", lambda prompt: next(answers), raising=False)

    service_a = DashboardServiceConfig(name="a", web_root=str(tmp_path / "a"), host="127.0.0.1", port=8765)
    service_b = DashboardServiceConfig(
        name="b",
        web_root=str(tmp_path / "b"),
        host="127.0.0.1",
        port=9000,
        public_base_url="https://dash.example.com",
        service_type="pm2",
    )
    store = DashboardServiceStore(default_service="a", services={"a": service_a, "b": service_b})
    monkeypatch.setattr("sft_label.cli.load_dashboard_service_store", lambda config_path=None: store, raising=False)
    monkeypatch.setattr("sft_label.cli.dashboard_service_status", lambda svc: {"state": "running", "reachable": True, "url": svc.base_url()}, raising=False)

    published_calls = {}

    def _fake_publish(service, run_dir, config_path=None):
        published_calls["service"] = service.name
        return {
            "run_id": "demo_run",
            "dashboards": {
                "labeling": {"url": "https://dash.example.com/runs/demo_run/dashboard_labeling_demo.html"},
            },
        }

    monkeypatch.setattr("sft_label.cli.publish_run_dashboards", _fake_publish, raising=False)
    monkeypatch.setattr("sft_label.cli.dispatch_command", lambda args, parser: {"run_dir": str(run_dir)}, raising=False)

    args = parser.parse_args(["start"])
    cmd_start(args, parser)

    out = capsys.readouterr().out
    assert published_calls["service"] == "b"
    assert "https://dash.example.com/runs/demo_run/dashboard_labeling_demo.html" in out
