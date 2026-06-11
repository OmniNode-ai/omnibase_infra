# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the disk-watermark bus event builder (OMN-13008)."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timezone
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "disk_watermark_event", _SCRIPTS / "disk_watermark_event.py"
)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

NOW = datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC)
TOPIC = "onex.evt.infra.disk-watermark.v1"


@pytest.mark.unit
class TestDiskWatermarkEvent:
    def _build(self, *, used_pct: int, severity: str) -> dict[str, object]:
        return mod.build_event(
            mount="/data",
            used_pct=used_pct,
            avail_kb=123456,
            severity=severity,
            warn_pct=85,
            crit_pct=90,
            host="server201",
            topic=TOPIC,
            now=NOW,
        )

    def test_warning_event_shape(self) -> None:
        ev = self._build(used_pct=87, severity="warning")
        assert ev["severity"] == "warning"
        assert ev["used_pct"] == 87
        assert ev["topic"] == TOPIC
        assert ev["event_type"] == "disk-watermark"
        assert ev["schema_version"] == "1.0.0"
        assert ev["emitted_at"] == NOW.isoformat()

    def test_critical_event_shape(self) -> None:
        ev = self._build(used_pct=95, severity="critical")
        assert ev["severity"] == "critical"
        assert ev["used_pct"] == 95

    def test_alert_key_is_stable_dedupe_key(self) -> None:
        a = self._build(used_pct=87, severity="warning")
        b = self._build(used_pct=88, severity="warning")
        # Same host/mount/severity collapses to one open ticket.
        assert (
            a["alert_key"] == b["alert_key"] == "disk-watermark:server201:/data:warning"
        )

    def test_invalid_severity_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._build(used_pct=50, severity="info")
