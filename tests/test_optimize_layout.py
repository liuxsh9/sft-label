import json

from sft_label.tools.optimize_layout import MANIFEST_FILENAME, run_optimize_layout


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_optimize_layout_dry_run_keeps_files(tmp_path):
    sub = tmp_path / "code" / "foo"
    _write(sub / "labeled_foo.json", '[{"id":"s1"}]')
    _write(sub / "labeled_foo.jsonl", '{"id":"s1"}\n')
    _write(sub / "monitor_foo.jsonl", '{"id":"s1"}\n')
    _write(sub / "stats_foo.json", '{"total_samples":1}')
    _write(sub / "dashboard_foo.html", "<html>pass1</html>")

    summary = run_optimize_layout(tmp_path, apply=False, prune_legacy=False)

    assert summary["apply"] is False
    assert summary["planned_actions"] > 0
    assert (sub / "labeled_foo.json").exists()
    assert not (sub / "labeled.json").exists()
    assert (tmp_path / MANIFEST_FILENAME).exists()


def test_optimize_layout_apply_with_prune(tmp_path):
    sub = tmp_path / "code" / "foo"

    _write(sub / "labeled_foo.json", '[{"id":"s1","labels":{"intent":"build"}}]')
    _write(sub / "labeled_foo.jsonl", '{"id":"s1","labels":{"intent":"build"}}\n')
    _write(sub / "monitor_foo.jsonl", '{"sample_id":"s1"}\n')
    _write(sub / "stats_foo.json", '{"total_samples":1,"success":1}')
    _write(sub / "dashboard_foo.html", "<html>pass1 legacy suffix</html>")

    _write(sub / "stats_value.json", '{"total_scored":1}')
    _write(sub / "dashboard_value.html", "<html>pass2 legacy</html>")

    _write(tmp_path / "summary_stats.json", '{"total_samples":1}')
    _write(tmp_path / "failed_value.jsonl", "")

    summary = run_optimize_layout(tmp_path, apply=True, prune_legacy=True)

    assert summary["apply"] is True
    assert summary["errors"] == 0
    assert summary["executed_done"] > 0

    # Suffix-normalized outputs
    assert (sub / "labeled.json").exists()
    assert (sub / "labeled.jsonl").exists()
    assert (sub / "monitor.jsonl").exists()
    assert not (sub / "labeled_foo.json").exists()
    assert not (sub / "labeled_foo.jsonl").exists()
    assert not (sub / "monitor_foo.jsonl").exists()

    # Pass 1 canonical promoted, legacy pruned
    assert (sub / "stats_labeling.json").exists()
    assert (sub / "dashboard_labeling.html").exists()
    assert not (sub / "stats.json").exists()
    assert not (sub / "dashboard.html").exists()

    # Pass 2 canonical promoted, legacy pruned
    assert (sub / "stats_scoring.json").exists()
    assert (sub / "dashboard_scoring.html").exists()
    assert not (sub / "stats_value.json").exists()
    assert not (sub / "dashboard_value.html").exists()

    # Root summary canonicalized
    assert (tmp_path / "summary_stats_labeling.json").exists()
    assert not (tmp_path / "summary_stats.json").exists()

    # Empty failure files removed when prune enabled
    assert not (tmp_path / "failed_value.jsonl").exists()

    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["summary"]["apply"] is True
    assert manifest["summary"]["prune_legacy"] is True
