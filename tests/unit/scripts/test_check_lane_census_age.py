# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/check_lane_census_age.py (OMN-13034).

Tests pin the staleness gate behavior: a census older than MAX_AGE_DAYS must
fail with exit code 1; a fresh census must pass with exit code 0. These tests
do NOT touch .201 — they drive the pure Python checker with fixture data.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

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


# ---------------------------------------------------------------------------
# Missing / malformed snapshot tests
# ---------------------------------------------------------------------------


def test_missing_snapshot_fails() -> None:
    """A missing snapshot file must fail immediately."""
    path = Path("/nonexistent/census-snapshot.json")
    result = MOD.check_census_age(path, 7)
    assert result == 1


def test_malformed_json_fails() -> None:
    """A snapshot that is not valid JSON must fail."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("{not valid json")
        path = Path(f.name)
    result = MOD.check_census_age(path, 7)
    assert result == 1
    path.unlink(missing_ok=True)


def test_missing_emitted_at_fails() -> None:
    """A snapshot without 'emitted_at' must fail."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"schema_version": "1.0.0", "lanes_checked": []}, f)
        path = Path(f.name)
    result = MOD.check_census_age(path, 7)
    assert result == 1
    path.unlink(missing_ok=True)


def test_unsupported_schema_version_fails() -> None:
    """A snapshot with an unsupported schema_version must fail."""
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    emitted = (now - timedelta(hours=1)).isoformat()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    _write_snapshot(path, emitted, schema_version="99.0.0")
    result = MOD.check_census_age(path, 7, now=now)
    assert result == 1
    path.unlink(missing_ok=True)


def test_invalid_timestamp_fails() -> None:
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
    result = MOD.check_census_age(path, 7)
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
# Real snapshot file test
# ---------------------------------------------------------------------------


def test_committed_snapshot_is_present_and_valid() -> None:
    """The committed census-snapshot.json must exist and be parseable."""
    snapshot_path = _REPO / "deploy" / "lane-census" / "census-snapshot.json"
    assert snapshot_path.exists(), (
        "deploy/lane-census/census-snapshot.json is missing. "
        "This file must be committed and kept fresh (OMN-13034)."
    )
    with open(snapshot_path, encoding="utf-8") as fh:
        snapshot = json.load(fh)
    assert snapshot.get("schema_version") == "1.0.0"
    assert "emitted_at" in snapshot
    assert "lanes_checked" in snapshot
