# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for typed Durable-findings / Runtime-state-as-of sections in generate_deep_dive.py.

Covers OMN-13043 (C-3 retro item): the generator must produce typed section
headers so that stale runtime-state claims can't poison the corpus.

Validates:
  - parse_runtime_state_timestamp: extracts the UTC datetime from a
    '## Runtime State — as of <YYYY-MM-DDTHH:MMZ>' header.
  - validate_deep_dive_ingest: rejects deep-dives whose Runtime State
    section is older than the provided handoff cutoff.
  - Generated output: both '## Durable Findings' and
    '## Runtime State — as of' sections are present in generated Markdown.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "generate_deep_dive.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_deep_dive", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Use a distinct module key to avoid interfering with other test files
    # that also load generate_deep_dive under the name "generate_deep_dive".
    sys.modules.setdefault("generate_deep_dive", mod)
    spec.loader.exec_module(mod)
    return sys.modules["generate_deep_dive"]


MOD = _load_module()

_UTC = ZoneInfo("UTC")


# ---------------------------------------------------------------------------
# parse_runtime_state_timestamp
# ---------------------------------------------------------------------------


class TestParseRuntimeStateTimestamp:
    """parse_runtime_state_timestamp must return a UTC-aware datetime or None."""

    def test_valid_header_returns_datetime(self) -> None:
        text = (
            "## Some header\n\n"
            "## Runtime State — as of 2026-06-28T14:30Z\n\n"
            "Some content."
        )
        ts = MOD.parse_runtime_state_timestamp(text)
        assert ts is not None
        assert ts.year == 2026
        assert ts.month == 6
        assert ts.day == 28
        assert ts.hour == 14
        assert ts.minute == 30
        assert ts.tzinfo is not None

    def test_missing_section_returns_none(self) -> None:
        text = "## Durable Findings\n\nSome findings.\n"
        assert MOD.parse_runtime_state_timestamp(text) is None

    def test_malformed_timestamp_returns_none(self) -> None:
        text = "## Runtime State — as of NOT-A-DATE\n\nContent."
        assert MOD.parse_runtime_state_timestamp(text) is None

    def test_timestamp_is_utc_aware(self) -> None:
        text = "## Runtime State — as of 2026-01-01T00:00Z\n"
        ts = MOD.parse_runtime_state_timestamp(text)
        assert ts is not None
        # Must be UTC-aware (offset zero)
        utc_offset = ts.utcoffset()
        assert utc_offset is not None
        assert utc_offset.total_seconds() == 0

    def test_returns_none_for_empty_string(self) -> None:
        assert MOD.parse_runtime_state_timestamp("") is None

    def test_section_in_middle_of_document(self) -> None:
        text = (
            "# Daily Deep Dive\n\n"
            "## Metrics\n\nSome data.\n\n"
            "## Runtime State — as of 2026-03-15T09:45Z\n\n"
            "Runtime observations.\n\n"
            "## Appendix\n\nMore stuff."
        )
        ts = MOD.parse_runtime_state_timestamp(text)
        assert ts is not None
        assert ts.day == 15
        assert ts.month == 3


# ---------------------------------------------------------------------------
# validate_deep_dive_ingest
# ---------------------------------------------------------------------------


def _write_deep_dive(content: str) -> Path:
    with tempfile.NamedTemporaryFile(
        suffix=".md", mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(content)
        return Path(f.name)


class TestValidateDeepDiveIngest:
    """validate_deep_dive_ingest returns (ok, message) based on freshness."""

    def _make_cutoff(self, iso: str) -> dt.datetime:
        return dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))

    def test_fresh_runtime_state_accepted(self) -> None:
        """Runtime state newer than handoff cutoff must be accepted."""
        content = (
            "## Durable Findings\n\nFinding A.\n\n"
            "## Runtime State — as of 2026-06-28T16:00Z\n\nDrift: green."
        )
        path = _write_deep_dive(content)
        cutoff = self._make_cutoff("2026-06-28T14:00Z")
        ok, msg = MOD.validate_deep_dive_ingest(path, cutoff)
        assert ok is True
        assert "fresh" in msg.lower()

    def test_equal_timestamp_accepted(self) -> None:
        """Runtime state exactly at the handoff cutoff must be accepted."""
        content = "## Runtime State — as of 2026-06-28T14:00Z\n\nDrift: yellow."
        path = _write_deep_dive(content)
        cutoff = self._make_cutoff("2026-06-28T14:00Z")
        ok, _ = MOD.validate_deep_dive_ingest(path, cutoff)
        assert ok is True

    def test_stale_runtime_state_rejected(self) -> None:
        """Runtime state older than handoff cutoff must be rejected."""
        content = (
            "## Durable Findings\n\nFinding B.\n\n"
            "## Runtime State — as of 2026-06-10T20:00Z\n\nProd broker is down."
        )
        path = _write_deep_dive(content)
        cutoff = self._make_cutoff("2026-06-10T22:00Z")
        ok, msg = MOD.validate_deep_dive_ingest(path, cutoff)
        assert ok is False
        assert "stale" in msg.lower()

    def test_missing_runtime_state_section_rejected(self) -> None:
        """Deep-dives without a Runtime State section must be rejected."""
        content = "## Durable Findings\n\nSome findings.\n\n## Metrics\n\nNumbers."
        path = _write_deep_dive(content)
        cutoff = self._make_cutoff("2026-06-01T00:00Z")
        ok, msg = MOD.validate_deep_dive_ingest(path, cutoff)
        assert ok is False
        assert "missing" in msg.lower()

    def test_nonexistent_file_rejected(self, tmp_path: Path) -> None:
        """Non-existent file must be rejected without raising."""
        cutoff = self._make_cutoff("2026-06-01T00:00Z")
        # Use tmp_path for a path that is guaranteed not to exist.
        missing = tmp_path / "does_not_exist_omn13043.md"
        ok, msg = MOD.validate_deep_dive_ingest(missing, cutoff)
        assert ok is False
        assert "cannot read" in msg.lower() or "no such" in msg.lower()

    def test_message_includes_timestamps_on_rejection(self) -> None:
        """Rejection message must cite both timestamps for debuggability."""
        content = "## Runtime State — as of 2026-06-10T20:00Z\n\nContent."
        path = _write_deep_dive(content)
        cutoff = self._make_cutoff("2026-06-11T00:00Z")
        ok, msg = MOD.validate_deep_dive_ingest(path, cutoff)
        assert ok is False
        # Both the section timestamp and the handoff cutoff must appear
        assert "2026-06-10T20:00" in msg
        assert "2026-06-11T00:00" in msg


# ---------------------------------------------------------------------------
# Generated-output section presence
# ---------------------------------------------------------------------------


class TestGeneratedOutputSections:
    """The generated deep-dive must contain both typed section headers."""

    def _run_generator(self, tmp_path: Path, date_str: str) -> str:
        """Run generate_deep_dive.main() with a no-op empty root and return output."""
        # Use a temporary directory as the workspace root.  It contains no
        # git repos, so repo_days will be empty — but the static sections
        # (Durable Findings, Runtime State) must still be emitted.
        out_file = tmp_path / "test_deep_dive.md"
        orig_argv = sys.argv
        try:
            sys.argv = [
                "generate_deep_dive.py",
                "--root",
                str(tmp_path),
                "--date",
                date_str,
                "--out",
                str(out_file),
            ]
            MOD.main()
        finally:
            sys.argv = orig_argv
        return out_file.read_text(encoding="utf-8")

    def test_durable_findings_section_present(self, tmp_path: Path) -> None:
        content = self._run_generator(tmp_path, "2026-06-28")
        assert "## Durable Findings" in content

    def test_runtime_state_section_present(self, tmp_path: Path) -> None:
        content = self._run_generator(tmp_path, "2026-06-28")
        assert "## Runtime State — as of" in content

    def test_runtime_state_timestamp_is_parseable(self, tmp_path: Path) -> None:
        content = self._run_generator(tmp_path, "2026-06-28")
        ts = MOD.parse_runtime_state_timestamp(content)
        assert ts is not None, (
            "Generated output must contain a parseable Runtime State timestamp"
        )

    def test_runtime_state_timestamp_is_utc(self, tmp_path: Path) -> None:
        content = self._run_generator(tmp_path, "2026-06-28")
        ts = MOD.parse_runtime_state_timestamp(content)
        assert ts is not None
        utc_offset = ts.utcoffset()
        assert utc_offset is not None
        assert utc_offset.total_seconds() == 0

    def test_generated_deep_dive_passes_own_validate(self, tmp_path: Path) -> None:
        """A freshly generated deep-dive must pass validate_deep_dive_ingest."""
        out_file = tmp_path / "test_deep_dive.md"
        orig_argv = sys.argv
        try:
            sys.argv = [
                "generate_deep_dive.py",
                "--root",
                str(tmp_path),
                "--date",
                "2026-06-28",
                "--out",
                str(out_file),
            ]
            MOD.main()
        finally:
            sys.argv = orig_argv
        # Use a cutoff well before generation — must pass.
        old_cutoff = dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt.UTC)
        ok, msg = MOD.validate_deep_dive_ingest(out_file, old_cutoff)
        assert ok is True, f"Fresh deep-dive rejected: {msg}"
