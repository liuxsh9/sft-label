"""Tests for interactive launcher command planning."""

from __future__ import annotations

from sft_label.launcher import (
    build_launch_plan,
    format_command,
    format_env_prefix,
)
from sft_label.cli import build_parser


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
            "",           # --limit (default 0)
            "",           # shuffle (default no)
            "",           # arbitration (default yes)
            "",           # prompt mode (full)
            "",           # model
            "",           # tag-stats
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


def test_build_launch_plan_can_cancel():
    io = StubIO(["0"])
    assert build_launch_plan(input_fn=io.input, output_fn=io.output) is None


def test_format_helpers():
    assert format_command(["score", "--input", "a b.json"]) == "sft-label score --input 'a b.json'"
    assert (
        format_env_prefix({"LITELLM_KEY": "abc", "LITELLM_BASE": "http://x/v1"})
        == "LITELLM_BASE=http://x/v1 LITELLM_KEY=abc"
    )


def test_all_workflows_generate_parseable_argv():
    parser = build_parser()
    workflow_answers = {
        # 1. run-pass1
        1: ["1", "data.json", "", "", "", "", "", "", "n", ""],
        # 2. run-pass1-pass2
        2: ["1", "data.json", "", "", "", "", "", "", "stats.json", "n", ""],
        # 3. run-pass1-pass2-semantic
        3: ["1", "data.json", "", "", "", "", "", "", "", "n", ""],
        # 4. score
        4: ["labeled.json", "", "", "", "", "", "n", ""],
        # 5. semantic
        5: ["run_dir", "", "", "", "", "", ""],
        # 6. filter
        6: ["scored.json", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        # 7. recompute-stats
        7: ["run_dir/", "", "", ""],
        # 8. regenerate-dashboard
        8: ["run_dir/", "", "", ""],
        # 9. validate
        9: [],
        # 10. export-semantic
        10: ["run_dir/", "out.jsonl", "", ""],
        # 11. export-review
        11: ["labeled.json", "review.csv", "", "", ""],
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
