from __future__ import annotations

from pathlib import Path

import sft_label.artifacts as artifacts_module
from sft_label.artifacts import dashboard_data_dirname, remove_dashboard_bundle


def test_remove_dashboard_bundle_tolerates_missing_appledouble_children(monkeypatch, tmp_path):
    html_path = tmp_path / "dashboards" / "dashboard_scoring_test.html"
    data_dir = html_path.with_name(dashboard_data_dirname(html_path.name))
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("<html/>", encoding="utf-8")
    (html_path.parent / f"._{html_path.name}").write_bytes(b"\x00\x01")
    data_dir.mkdir()
    (data_dir / "global.json").write_text("{}", encoding="utf-8")

    real_rmtree = artifacts_module.shutil.rmtree

    def _fake_rmtree(path, onerror=None):
        missing = FileNotFoundError(2, "No such file or directory", "._global.json")
        assert onerror is not None
        onerror(Path.unlink, str(Path(path) / "._global.json"), (FileNotFoundError, missing, None))
        real_rmtree(path)

    monkeypatch.setattr(artifacts_module.shutil, "rmtree", _fake_rmtree)

    remove_dashboard_bundle(html_path)

    assert not html_path.exists()
    assert not (html_path.parent / f"._{html_path.name}").exists()
    assert not data_dir.exists()


def test_remove_dashboard_bundle_removes_dashboard_sidecar_files(tmp_path):
    html_path = tmp_path / "dashboards" / "dashboard_labeling_test.html"
    data_dir = html_path.with_name(dashboard_data_dirname(html_path.name))
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text("<html/>", encoding="utf-8")
    (html_path.parent / f"._{html_path.name}").write_bytes(b"\x00\x01")
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text("{}", encoding="utf-8")
    data_dir.with_name(f"._{data_dir.name}").write_bytes(b"\x00\x01")

    remove_dashboard_bundle(html_path)

    assert not html_path.exists()
    assert not (html_path.parent / f"._{html_path.name}").exists()
    assert not data_dir.exists()
    assert not data_dir.with_name(f"._{data_dir.name}").exists()
