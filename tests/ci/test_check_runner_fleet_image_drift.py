# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the runner fleet image-drift detector (OMN-15104).

OMN-13946 (add libatomic1, image_version 5 -> 6) merged 2026-07-09 and was
marked Done, but every one of the 64 live runner containers was still on
image_version 5 sixteen days later, deterministically failing Pyright
fleet-wide. ``runner-monitor.sh`` had zero visibility into image identity, so
the drift was silent. These tests hold the pure comparison/reporting logic
(``find_stale_containers`` / ``render_report``) to the fail-closed contract:
an unreadable/unknown container must be reported, never silently treated as
passing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
DRIFT_SCRIPT = SCRIPTS_CI / "check_runner_fleet_image_drift.py"


def _load_drift_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_runner_fleet_image_drift", DRIFT_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS_CI))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(SCRIPTS_CI) in sys.path:
            sys.path.remove(str(SCRIPTS_CI))
    return module


def test_all_containers_current_reports_no_findings() -> None:
    module = _load_drift_module()
    findings = module.find_stale_containers(
        expected_image_version=6,
        observed_versions={
            "omninode-runner-1": 6,
            "omninode-runner-2": 7,  # ahead of expected is fine, not stale
        },
    )
    assert findings == []


def test_stale_container_is_reported() -> None:
    module = _load_drift_module()
    findings = module.find_stale_containers(
        expected_image_version=6,
        observed_versions={
            "omninode-runner-1": 6,
            "omninode-runner-48": 5,
        },
    )
    assert len(findings) == 1
    assert findings[0].container == "omninode-runner-48"
    assert findings[0].observed_version == 5
    assert findings[0].expected_version == 6


def test_unreadable_container_lock_is_reported_not_skipped() -> None:
    """Fail-closed: a container we cannot verify must be flagged, never
    silently treated as passing (the exact failure mode this check exists to
    close — 16 days of "64/64 healthy" while every container was stale)."""
    module = _load_drift_module()
    findings = module.find_stale_containers(
        expected_image_version=6,
        observed_versions={"omninode-runner-9": None},
    )
    assert len(findings) == 1
    assert findings[0].container == "omninode-runner-9"
    assert findings[0].observed_version is None
    assert "UNKNOWN" in findings[0].as_line()


def test_render_report_ok_when_no_findings() -> None:
    module = _load_drift_module()
    report = module.render_report(expected_image_version=6, findings=[])
    assert "OK" in report
    assert "6" in report


def test_render_report_lists_every_finding() -> None:
    module = _load_drift_module()
    findings = module.find_stale_containers(
        expected_image_version=6,
        observed_versions={
            "omninode-runner-1": 5,
            "omninode-runner-2": 5,
            "omninode-runner-3": 6,
        },
    )
    report = module.render_report(expected_image_version=6, findings=findings)
    assert "omninode-runner-1" in report
    assert "omninode-runner-2" in report
    assert "omninode-runner-3" not in report
    assert "2 container(s) stale" in report


def test_load_expected_version_reads_lock_file(tmp_path: Path) -> None:
    module = _load_drift_module()
    lock = tmp_path / "runner-image.lock.json"
    lock.write_text('{"image_version": 6}', encoding="utf-8")
    assert module._load_expected_version(lock) == 6


def test_load_expected_version_rejects_non_int(tmp_path: Path) -> None:
    module = _load_drift_module()
    lock = tmp_path / "runner-image.lock.json"
    lock.write_text('{"image_version": "6"}', encoding="utf-8")
    with pytest.raises(TypeError):
        module._load_expected_version(lock)


def test_discover_observed_versions_uses_read_baked_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_drift_module()
    calls: list[str] = []

    def _fake_read(container: str) -> int | None:
        calls.append(container)
        return {"omninode-runner-1": 6, "omninode-runner-2": None}[container]

    monkeypatch.setattr(module, "_read_baked_version", _fake_read)
    observed = module.discover_observed_versions(
        ["omninode-runner-1", "omninode-runner-2"]
    )
    assert observed == {"omninode-runner-1": 6, "omninode-runner-2": None}
    assert calls == ["omninode-runner-1", "omninode-runner-2"]


def test_current_lock_file_image_version_is_at_least_six() -> None:
    """Regression guard: the checked-in lock (post-OMN-13946) must never
    regress below the libatomic1 fix's image_version. A future edit that
    drops the version without deploying would reintroduce exactly the
    incident this ticket exists to close."""
    module = _load_drift_module()
    lock_path = REPO_ROOT / "docker" / "runners" / "runner-image.lock.json"
    assert module._load_expected_version(lock_path) >= 6
