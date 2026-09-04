# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the disk-watermark bus event builder (OMN-13008, schema 2.0.0 OMN-17872)."""

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
    def _build(
        self,
        *,
        used_pct: int,
        severity: str,
        avail_gb: int = 120,
        halt_reason: str = "used_pct_advisory",
    ) -> dict[str, object]:
        return mod.build_event(
            mount="/data",
            used_pct=used_pct,
            avail_kb=avail_gb * 1024 * 1024,
            avail_gb=avail_gb,
            severity=severity,
            halt_reason=halt_reason,
            warn_pct=85,
            warn_free_gb=100,
            crit_free_gb=50,
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
        assert ev["schema_version"] == "2.0.0"
        assert ev["emitted_at"] == NOW.isoformat()

    def test_critical_event_shape(self) -> None:
        ev = self._build(
            used_pct=95,
            severity="critical",
            avail_gb=20,
            halt_reason="free_space_below_crit_floor",
        )
        assert ev["severity"] == "critical"
        assert ev["used_pct"] == 95
        assert ev["avail_gb"] == 20
        assert ev["crit_free_gb"] == 50

    def test_percentage_can_never_be_critical(self) -> None:
        """The advisory percentage must not be able to mint a critical event."""
        with pytest.raises(ValueError, match="only free space below"):
            self._build(
                used_pct=99,
                severity="critical",
                avail_gb=400,
                halt_reason="used_pct_advisory",
            )

    def test_unknown_halt_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="halt_reason"):
            self._build(used_pct=87, severity="warning", halt_reason="vibes")

    def test_message_names_both_numbers(self) -> None:
        """A receipt quoting the message must see free-GB vs floor AND pct vs line."""
        ev = self._build(
            used_pct=99,
            severity="critical",
            avail_gb=20,
            halt_reason="free_space_below_crit_floor",
        )
        message = ev["message"]
        assert isinstance(message, str)
        assert "20 GiB free" in message
        assert "crit floor 50 GiB" in message
        assert "warn floor 100 GiB" in message
        assert "99% used" in message
        assert "advisory warn>=85%" in message

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
