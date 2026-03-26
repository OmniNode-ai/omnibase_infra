# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for monitor_logs.py Slack alert fixes (OMN-3311)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# monitor_logs.py lives in scripts/ and has no package install; add scripts/ to path.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

# Lazy import helper — reload each time to avoid cross-test pollution from
# the module-level _load_omnibase_env() side-effect.
_MODULE_NAME = "monitor_logs"


def _import() -> Any:
    if _MODULE_NAME in sys.modules:
        return importlib.reload(sys.modules[_MODULE_NAME])
    return importlib.import_module(_MODULE_NAME)


# ---------------------------------------------------------------------------
# _sanitize_log_text
# ---------------------------------------------------------------------------


class TestSanitizeLogText:
    """Tests for _sanitize_log_text()."""

    @pytest.mark.unit
    def test_strips_ansi_color_codes(self) -> None:
        """ANSI SGR sequences (colors, bold, reset) must be removed."""
        m = _import()
        raw = "\x1b[31mERROR\x1b[0m: something went wrong"
        result = m._sanitize_log_text(raw)
        assert "\x1b" not in result
        assert "ERROR" in result
        assert "something went wrong" in result

    @pytest.mark.unit
    def test_strips_ansi_cursor_sequences(self) -> None:
        """ANSI cursor-movement escape sequences must be removed."""
        m = _import()
        # ESC[K (erase to end of line), ESC[H (cursor home)
        raw = "line1\x1b[Kline2\x1b[H"
        result = m._sanitize_log_text(raw)
        assert "\x1b" not in result
        assert "line1" in result
        assert "line2" in result

    @pytest.mark.unit
    def test_strips_osc_sequences(self) -> None:
        """OSC hyperlink/title sequences (ESC ] ... BEL) must be removed."""
        m = _import()
        raw = "\x1b]0;window title\x07some log text"
        result = m._sanitize_log_text(raw)
        assert "\x1b" not in result
        assert "some log text" in result

    @pytest.mark.unit
    def test_preserves_newlines(self) -> None:
        """Newline characters must be kept intact."""
        m = _import()
        raw = "line one\nline two\nline three"
        result = m._sanitize_log_text(raw)
        assert result.count("\n") == 2

    @pytest.mark.unit
    def test_replaces_control_chars_with_question_mark(self) -> None:
        """Non-newline control characters (e.g. \\x01, \\x07, \\x0c) become '?'."""
        m = _import()
        raw = "before\x01\x07\x0cafter"
        result = m._sanitize_log_text(raw)
        assert "\x01" not in result
        assert "\x07" not in result
        assert "\x0c" not in result
        assert "?" in result
        assert "before" in result
        assert "after" in result

    @pytest.mark.unit
    def test_passthrough_for_clean_text(self) -> None:
        """Plain ASCII log text should be returned unchanged."""
        m = _import()
        raw = "2026-01-01 ERROR: disk full\nStack trace follows"
        assert m._sanitize_log_text(raw) == raw


# ---------------------------------------------------------------------------
# post_slack — truncation
# ---------------------------------------------------------------------------


class TestPostSlackTruncation:
    """Verify the mrkdwn block field never exceeds MAX_SLACK_CHARS."""

    @pytest.mark.unit
    def test_mrkdwn_block_text_within_limit(self) -> None:
        """The assembled mrkdwn text field must be <= MAX_SLACK_CHARS chars."""
        m = _import()
        # Generate 4000 chars of log — well over the 3000-char limit.
        long_line = "A" * 4000
        lines = [long_line]

        captured_payload: dict[str, Any] = {}

        def fake_urlopen(req: Any, timeout: int = 10) -> Any:
            body = json.loads(req.data)
            captured_payload.update(body)
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            m.post_slack("tok", "chan", "my-container", lines, dry_run=False)

        # Find the code-fence block
        blocks = captured_payload.get("blocks", [])
        assert len(blocks) == 2
        mrkdwn_text: str = blocks[1]["text"]["text"]
        assert len(mrkdwn_text) <= m.MAX_SLACK_CHARS, (
            f"mrkdwn text is {len(mrkdwn_text)} chars, expected <= {m.MAX_SLACK_CHARS}"
        )

    @pytest.mark.unit
    def test_short_log_not_truncated(self) -> None:
        """Short log text must not be truncated."""
        m = _import()
        lines = ["ERROR: disk full", "Traceback: ..."]

        captured_payload: dict[str, Any] = {}

        def fake_urlopen(req: Any, timeout: int = 10) -> Any:
            body = json.loads(req.data)
            captured_payload.update(body)
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            m.post_slack("tok", "chan", "my-container", lines, dry_run=False)

        blocks = captured_payload.get("blocks", [])
        mrkdwn_text: str = blocks[1]["text"]["text"]
        # Full content must be present (no truncation for short text)
        assert "ERROR: disk full" in mrkdwn_text
        assert "Traceback: ..." in mrkdwn_text


# ---------------------------------------------------------------------------
# post_slack — invalid_blocks fallback
# ---------------------------------------------------------------------------


class TestPostSlackInvalidBlocksFallback:
    """Verify post_slack retries with plain text when API returns invalid_blocks."""

    @pytest.mark.unit
    def test_retries_with_plain_text_on_invalid_blocks(self) -> None:
        """When Slack returns invalid_blocks, a plain-text fallback must be posted."""
        m = _import()
        lines = ["ERROR: container crash"]
        call_count = 0
        plain_text_payload: dict[str, Any] = {}

        def fake_urlopen(req: Any, timeout: int = 10) -> Any:
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            if call_count == 1:
                # First call: blocks request — return invalid_blocks
                mock_resp.read.return_value = json.dumps(
                    {"ok": False, "error": "invalid_blocks"}
                ).encode()
            else:
                # Second call: plain-text fallback — capture payload and succeed
                plain_text_payload.update(json.loads(req.data))
                mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            m.post_slack("tok", "chan", "crash-container", lines, dry_run=False)

        assert call_count == 2, "Expected exactly 2 Slack API calls (blocks + fallback)"
        assert "blocks" not in plain_text_payload, "Fallback must not include blocks"
        assert "text" in plain_text_payload
        assert "ERROR: container crash" in plain_text_payload["text"]

    @pytest.mark.unit
    def test_no_retry_on_other_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Non-invalid_blocks errors must be logged once without retrying."""
        m = _import()
        lines = ["ERROR: something"]
        call_count = 0

        def fake_urlopen(req: Any, timeout: int = 10) -> Any:
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps(
                {"ok": False, "error": "channel_not_found"}
            ).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            m.post_slack("tok", "chan", "my-container", lines, dry_run=False)

        assert call_count == 1, "No retry expected for non-invalid_blocks error"
        captured = capsys.readouterr()
        assert "channel_not_found" in captured.err


# ---------------------------------------------------------------------------
# post_slack — normal success path
# ---------------------------------------------------------------------------


class TestPostSlackSuccess:
    """Verify normal (non-error) post_slack path still works correctly."""

    @pytest.mark.unit
    def test_success_posts_blocks_payload(self) -> None:
        """A successful response must post a blocks payload with the right shape."""
        m = _import()
        lines = ["INFO: startup complete", "ERROR: disk full"]
        captured_payload: dict[str, Any] = {}

        def fake_urlopen(req: Any, timeout: int = 10) -> Any:
            captured_payload.update(json.loads(req.data))
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({"ok": True}).encode()
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            m.post_slack("xoxb-token", "C123", "my-container", lines, dry_run=False)

        assert captured_payload["channel"] == "C123"
        blocks = captured_payload["blocks"]
        assert blocks[0]["text"]["type"] == "mrkdwn"
        assert "my-container" in blocks[0]["text"]["text"]
        assert "```" in blocks[1]["text"]["text"]
        assert "ERROR: disk full" in blocks[1]["text"]["text"]

    @pytest.mark.unit
    def test_dry_run_does_not_call_urlopen(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """dry_run=True must print the payload without making any HTTP calls."""
        m = _import()
        lines = ["ERROR: crash"]

        with patch("urllib.request.urlopen") as mock_open:
            m.post_slack("tok", "chan", "ctr", lines, dry_run=True)
            mock_open.assert_not_called()

        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "ctr" in captured.out
