# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/check_lane_census_age.py (OMN-13034).

Tests pin the staleness gate behavior: a census older than MAX_AGE_DAYS must
fail with exit code 1 unless age is advisory; a fresh census must pass with
exit code 0. Invalid snapshots always fail. These tests do NOT touch the lab host —
they drive the pure Python checker with fixture data.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.helpers.lane_census_fixtures import use_neutral_lane_census

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "check_lane_census_age.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("check_lane_census_age", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _write_snapshot(path: Path, emitted_at: str, schema_version: str = "1.0.0") -> None:
    snapshot = {
        "schema_version": schema_version,
        "event_type": "lane-census-drift",
        "topic": "onex.evt.infra.lane-census-drift.v1",
        "host": "192.168.86.201",
        "emitted_at": emitted_at,
        "severity": "warning",
        "lanes_checked": ["stability-test", "prod", "judge"],
        "drift_count": 0,
        "findings": [],
        "alert_key": "lane-census-drift:192.168.86.201:no-drift",
        "ticket_title": "lane census: no drift",
        "ticket_body": "No drift.",
    }
    path.write_text(json.dumps(snapshot), encoding="utf-8")


# ---------------------------------------------------------------------------
# Freshness tests
# ---------------------------------------------------------------------------


def test_fresh_snapshot_passes() -> None:
    """A snapshot emitted 1 hour ago must pass the 7-day gate."""
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    emitted = (now - timedelta(hours=1)).isoformat()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    _write_snapshot(path, emitted)
    result = MOD.check_census_age(path, 7, now=now)
    assert result == 0
    path.unlink(missing_ok=True)


def test_snapshot_exactly_at_limit_passes() -> None:
    """A snapshot exactly 7 days old (not exceeding) must pass."""
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    # exactly 7 days = timedelta(days=7) which is NOT > timedelta(days=7)
    emitted = (now - timedelta(days=7)).isoformat()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    _write_snapshot(path, emitted)
    result = MOD.check_census_age(path, 7, now=now)
    assert result == 0
    path.unlink(missing_ok=True)


def test_snapshot_one_second_over_limit_fails() -> None:
    """A snapshot 7 days + 1 second old must fail."""
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    emitted = (now - timedelta(days=7, seconds=1)).isoformat()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    _write_snapshot(path, emitted)
    result = MOD.check_census_age(path, 7, now=now)
    assert result == 1
    path.unlink(missing_ok=True)


def test_stale_snapshot_fails() -> None:
    """A snapshot 30 days old must fail the default 7-day gate."""
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    emitted = (now - timedelta(days=30)).isoformat()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    _write_snapshot(path, emitted)
    result = MOD.check_census_age(path, 7, now=now)
    assert result == 1
    path.unlink(missing_ok=True)


def test_stale_snapshot_is_advisory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Advisory age warns about a stale snapshot without failing the PR."""
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    path = tmp_path / "census-snapshot.json"
    _write_snapshot(path, (now - timedelta(days=30)).isoformat())
    assert MOD.check_census_age(path, 7, now=now, stale_is_advisory=True) == 0
    stderr = capsys.readouterr().err
    assert "::warning title=LANE-CENSUS-STALE" in stderr
    assert "::error" not in stderr
    assert "not a property of the pull request's diff" in stderr


# ---------------------------------------------------------------------------
# Missing / malformed snapshot tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stale_is_advisory", [False, True])
def test_missing_snapshot_fails(stale_is_advisory: bool) -> None:
    """A missing snapshot file must fail immediately."""
    path = Path("/nonexistent/census-snapshot.json")
    result = MOD.check_census_age(path, 7, stale_is_advisory=stale_is_advisory)
    assert result == 1


@pytest.mark.parametrize("stale_is_advisory", [False, True])
def test_malformed_json_fails(stale_is_advisory: bool) -> None:
    """A snapshot that is not valid JSON must fail."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("{not valid json")
        path = Path(f.name)
    result = MOD.check_census_age(path, 7, stale_is_advisory=stale_is_advisory)
    assert result == 1
    path.unlink(missing_ok=True)


@pytest.mark.parametrize("stale_is_advisory", [False, True])
def test_missing_emitted_at_fails(stale_is_advisory: bool) -> None:
    """A snapshot without 'emitted_at' must fail."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"schema_version": "1.0.0", "lanes_checked": []}, f)
        path = Path(f.name)
    result = MOD.check_census_age(path, 7, stale_is_advisory=stale_is_advisory)
    assert result == 1
    path.unlink(missing_ok=True)


@pytest.mark.parametrize("stale_is_advisory", [False, True])
def test_unsupported_schema_version_fails(stale_is_advisory: bool) -> None:
    """A snapshot with an unsupported schema_version must fail."""
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    emitted = (now - timedelta(hours=1)).isoformat()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    _write_snapshot(path, emitted, schema_version="99.0.0")
    result = MOD.check_census_age(path, 7, now=now, stale_is_advisory=stale_is_advisory)
    assert result == 1
    path.unlink(missing_ok=True)


@pytest.mark.parametrize("stale_is_advisory", [False, True])
def test_invalid_timestamp_fails(stale_is_advisory: bool) -> None:
    """A snapshot with a non-ISO-8601 emitted_at must fail."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(
            {
                "schema_version": "1.0.0",
                "emitted_at": "not-a-date",
                "lanes_checked": [],
            },
            f,
        )
        path = Path(f.name)
    result = MOD.check_census_age(path, 7, stale_is_advisory=stale_is_advisory)
    assert result == 1
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Custom max-age tests
# ---------------------------------------------------------------------------


def test_custom_max_age_days_respected() -> None:
    """A snapshot 2 days old should pass with max_age_days=3 but fail with max_age_days=1."""
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    emitted = (now - timedelta(days=2)).isoformat()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    _write_snapshot(path, emitted)

    assert MOD.check_census_age(path, 3, now=now) == 0  # 2 days < 3 day limit
    assert MOD.check_census_age(path, 1, now=now) == 1  # 2 days > 1 day limit

    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Advisory CLI / workflow tests
# ---------------------------------------------------------------------------


def test_main_stale_advisory_flag(tmp_path: Path) -> None:
    """The CLI only accepts a stale snapshot when explicitly advisory."""
    path = tmp_path / "census-snapshot.json"
    _write_snapshot(path, (datetime.now(UTC) - timedelta(days=30)).isoformat())
    assert MOD.main(["--stale-advisory", "--snapshot", str(path)]) == 0
    assert MOD.main(["--snapshot", str(path)]) == 1


def test_workflow_stale_advisory_only_on_pull_requests() -> None:
    """Push and manual runs enforce age; only PRs pass --stale-advisory."""
    workflow = yaml.safe_load(
        (_REPO / ".github/workflows/lane-census-staleness.yml").read_text(
            encoding="utf-8"
        )
    )
    steps = workflow["jobs"]["lane-census-staleness"]["steps"]
    step = next(s for s in steps if s["name"] == "Check census snapshot staleness")
    assert step["env"]["STALE_ADVISORY"] == (
        "${{ github.event_name == 'pull_request' }}"
    )
    assert 'if [[ "${STALE_ADVISORY}" == "true" ]]; then\n' in step["run"]
    assert "args+=(--stale-advisory)\n" in step["run"]
    assert '--max-age-days "${MAX_AGE_DAYS}" "${args[@]}"' in step["run"]


# ---------------------------------------------------------------------------
# Default resolution (OMN-20934)
# ---------------------------------------------------------------------------


def test_default_snapshot_resolves_under_the_workspace_config_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no --snapshot the gate reads the census under the config root.

    The committed census is private config (OMN-20934); this repository ships
    none. The neutral fixture stands in for it here.
    """
    use_neutral_lane_census(monkeypatch)
    assert MOD.main(["--max-age-days", "100000"]) == 0


def test_default_snapshot_without_a_config_root_fails_loud(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No config root is a red gate naming the variable, never a pass."""
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    monkeypatch.delenv("OMNI_HOME", raising=False)
    assert MOD.main([]) == 1
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in capsys.readouterr().err
