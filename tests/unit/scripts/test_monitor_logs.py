# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for monitor_logs.py Slack alert fixes (OMN-3311)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# monitor_logs.py lives in scripts/ and has no package install; add scripts/ to path.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

# Lazy import helper — reload each time so tests receive fresh module globals.
_MODULE_NAME = "monitor_logs"
_ENV_SENTINEL = "OMN15572_MONITOR_IMPORT_SENTINEL"
_MONITOR_SCRIPT = _SCRIPTS_DIR / "monitor_logs.py"


def _import() -> Any:
    if _MODULE_NAME in sys.modules:
        return importlib.reload(sys.modules[_MODULE_NAME])
    return importlib.import_module(_MODULE_NAME)


def _controlled_subprocess_env(tmp_path: Path) -> dict[str, str]:
    """Return a child env whose home file is the only sentinel authority."""
    home_dir = tmp_path / "home"
    env_dir = home_dir / ".omnibase"
    env_dir.mkdir(parents=True)
    restart_hwm_file = home_dir / "runtime" / "restart-hwm.json"
    onex_state_dir = home_dir / "controlled-onex-state"
    (env_dir / ".env").write_text(
        f"{_ENV_SENTINEL}=loaded-from-home-env\n"
        "MONITOR_WARNING_COOLDOWN=2468\n"
        f"MONITOR_RESTART_HWM_FILE={restart_hwm_file}\n"
        f"ONEX_STATE_DIR={onex_state_dir}\n",
        encoding="utf-8",
    )

    subprocess_env = dict(os.environ)
    subprocess_env["HOME"] = str(home_dir)
    subprocess_env["OMN15572_EXPECTED_RESTART_HWM_FILE"] = str(restart_hwm_file)
    subprocess_env["OMN15572_EXPECTED_ONEX_STATE_DIR"] = str(onex_state_dir)
    subprocess_env.pop(_ENV_SENTINEL, None)
    for key in (
        "MONITOR_WARNING_COOLDOWN",
        "MONITOR_RESTART_HWM_FILE",
        "ONEX_STATE_DIR",
    ):
        subprocess_env.pop(key, None)
    return subprocess_env


def _config_assertions_source() -> str:
    """Return child-process assertions for every env-derived module constant."""
    return f"""
from pathlib import Path


def assert_monitor_config(module_globals):
    state_dir = Path(os.environ["OMN15572_EXPECTED_ONEX_STATE_DIR"])
    expected_logs = [
        str(state_dir / "logs" / "env-sync.log"),
        str(state_dir / "logs" / "hooks.log"),
        str(state_dir / "logs" / "pipeline-trace.log"),
    ]
    assert os.environ[{_ENV_SENTINEL!r}] == "loaded-from-home-env"
    assert module_globals["WARNING_COOLDOWN_SECONDS"] == 2468
    assert module_globals["_RESTART_HWM_FILE"] == Path(
        os.environ["OMN15572_EXPECTED_RESTART_HWM_FILE"]
    )
    assert module_globals["_ONEX_STATE_DIR"] == state_dir
    assert module_globals["_DEFAULT_FILE_LOGS"] == expected_logs
"""


class TestImportEnvironmentIsolation:
    """Importing the monitor library must not mutate its caller's environment."""

    @pytest.mark.unit
    def test_import_does_not_load_home_env(self, tmp_path: Path) -> None:
        subprocess_env = _controlled_subprocess_env(tmp_path)
        subprocess_env["OMN15572_MONITOR_SCRIPTS_DIR"] = str(_SCRIPTS_DIR)
        probe = (
            "import os, sys; "
            "sys.path.insert(0, os.environ['OMN15572_MONITOR_SCRIPTS_DIR']); "
            "import monitor_logs; "
            f"assert {_ENV_SENTINEL!r} not in os.environ"
        )

        result = subprocess.run(
            [sys.executable, "-c", probe],
            env=subprocess_env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0, result.stderr

    @pytest.mark.unit
    def test_repeated_callable_main_refreshes_all_module_config(
        self,
        tmp_path: Path,
    ) -> None:
        """Each callable entry resolves current env while loading the home file once."""
        first_state_dir = tmp_path / "first-state"
        first_restart_hwm = tmp_path / "first-restart-hwm.json"
        second_state_dir = tmp_path / "second-state"
        second_restart_hwm = tmp_path / "second-restart-hwm.json"
        first_env = {
            "MONITOR_WARNING_COOLDOWN": "1111",
            "MONITOR_RESTART_HWM_FILE": str(first_restart_hwm),
            "ONEX_STATE_DIR": str(first_state_dir),
        }
        second_env = {
            "MONITOR_WARNING_COOLDOWN": "2222",
            "MONITOR_RESTART_HWM_FILE": str(second_restart_hwm),
            "ONEX_STATE_DIR": str(second_state_dir),
        }

        with patch.dict(os.environ, first_env):
            m = _import()

        class ConfigObservedError(Exception):
            pass

        def config_observer(
            *,
            warning_cooldown: int,
            restart_hwm: Path,
            state_dir: Path,
        ) -> Callable[..., None]:
            def assert_config(
                _parser: argparse.ArgumentParser,
                *_args: Any,
                **_kwargs: Any,
            ) -> None:
                assert warning_cooldown == m.WARNING_COOLDOWN_SECONDS
                assert restart_hwm == m._RESTART_HWM_FILE
                assert state_dir == m._ONEX_STATE_DIR
                assert [
                    str(state_dir / "logs" / "env-sync.log"),
                    str(state_dir / "logs" / "hooks.log"),
                    str(state_dir / "logs" / "pipeline-trace.log"),
                ] == m._DEFAULT_FILE_LOGS
                raise ConfigObservedError

            return assert_config

        with patch.object(m, "_load_omnibase_env", autospec=True) as load_env:
            for env, warning_cooldown, restart_hwm, state_dir in (
                (first_env, 1111, first_restart_hwm, first_state_dir),
                (second_env, 2222, second_restart_hwm, second_state_dir),
            ):
                observe = config_observer(
                    warning_cooldown=warning_cooldown,
                    restart_hwm=restart_hwm,
                    state_dir=state_dir,
                )
                with (
                    patch.dict(os.environ, env),
                    patch.object(argparse.ArgumentParser, "parse_args", observe),
                    pytest.raises(ConfigObservedError),
                ):
                    m.main()

            assert load_env.call_count == 1

    @pytest.mark.unit
    def test_callable_main_recovers_after_invalid_first_resolve(
        self,
        tmp_path: Path,
    ) -> None:
        """A failed first resolve cannot poison later callable configuration."""
        baseline_env = {
            "MONITOR_WARNING_COOLDOWN": "1000",
            "MONITOR_RESTART_HWM_FILE": str(tmp_path / "baseline-hwm.json"),
            "ONEX_STATE_DIR": str(tmp_path / "baseline-state"),
        }
        invalid_env = {
            "MONITOR_WARNING_COOLDOWN": "invalid",
            "MONITOR_RESTART_HWM_FILE": str(tmp_path / "invalid-hwm.json"),
            "ONEX_STATE_DIR": str(tmp_path / "invalid-state"),
        }
        recovered_state_dir = tmp_path / "recovered-state"
        recovered_restart_hwm = tmp_path / "recovered-hwm.json"
        recovered_env = {
            "MONITOR_WARNING_COOLDOWN": "3333",
            "MONITOR_RESTART_HWM_FILE": str(recovered_restart_hwm),
            "ONEX_STATE_DIR": str(recovered_state_dir),
        }

        with patch.dict(os.environ, baseline_env):
            m = _import()

        class ConfigObservedError(Exception):
            pass

        def observe_recovered_config(
            _parser: argparse.ArgumentParser,
            *_args: Any,
            **_kwargs: Any,
        ) -> None:
            assert m.WARNING_COOLDOWN_SECONDS == 3333
            assert recovered_restart_hwm == m._RESTART_HWM_FILE
            assert recovered_state_dir == m._ONEX_STATE_DIR
            assert [
                str(recovered_state_dir / "logs" / "env-sync.log"),
                str(recovered_state_dir / "logs" / "hooks.log"),
                str(recovered_state_dir / "logs" / "pipeline-trace.log"),
            ] == m._DEFAULT_FILE_LOGS
            raise ConfigObservedError

        with patch.object(m, "_load_omnibase_env", autospec=True) as load_env:
            with (
                patch.dict(os.environ, invalid_env),
                pytest.raises(ValueError, match="invalid literal"),
            ):
                m.main()

            with (
                patch.dict(os.environ, recovered_env),
                patch.object(
                    argparse.ArgumentParser,
                    "parse_args",
                    observe_recovered_config,
                ),
                pytest.raises(ConfigObservedError),
            ):
                m.main()

            assert load_env.call_count == 1

    @pytest.mark.unit
    def test_import_then_callable_main_bootstraps_all_module_config(
        self,
        tmp_path: Path,
    ) -> None:
        subprocess_env = _controlled_subprocess_env(tmp_path)
        subprocess_env["OMN15572_MONITOR_SCRIPTS_DIR"] = str(_SCRIPTS_DIR)
        probe = (
            _config_assertions_source()
            + f"""
import argparse
import os
import sys

sys.path.insert(0, os.environ["OMN15572_MONITOR_SCRIPTS_DIR"])
import monitor_logs

assert {_ENV_SENTINEL!r} not in os.environ


class ConfigObserved(Exception):
    pass


def stop_after_config(_parser, *_args, **_kwargs):
    assert_monitor_config(vars(monitor_logs))
    raise ConfigObserved


argparse.ArgumentParser.parse_args = stop_after_config
try:
    monitor_logs.main()
except ConfigObserved:
    pass
else:
    raise AssertionError("callable main did not reach config observation boundary")
"""
        )

        result = subprocess.run(
            [sys.executable, "-c", probe],
            env=subprocess_env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0, result.stderr

    @pytest.mark.unit
    def test_direct_script_loads_env_before_all_module_config(
        self,
        tmp_path: Path,
    ) -> None:
        subprocess_env = _controlled_subprocess_env(tmp_path)
        subprocess_env["OMN15572_MONITOR_SCRIPT"] = str(_MONITOR_SCRIPT)
        probe = (
            _config_assertions_source()
            + """
import argparse
import inspect
import os
import runpy


class ConfigObserved(Exception):
    pass


def stop_after_config(_parser, *_args, **_kwargs):
    frame = inspect.currentframe()
    module_globals = None
    while frame is not None:
        if frame.f_globals.get("__file__") == os.environ["OMN15572_MONITOR_SCRIPT"]:
            module_globals = frame.f_globals
            break
        frame = frame.f_back
    assert module_globals is not None
    assert_monitor_config(module_globals)
    raise ConfigObserved


argparse.ArgumentParser.parse_args = stop_after_config
try:
    runpy.run_path(os.environ["OMN15572_MONITOR_SCRIPT"], run_name="__main__")
except ConfigObserved:
    pass
else:
    raise AssertionError("monitor main did not reach config observation boundary")
"""
        )

        result = subprocess.run(
            [sys.executable, "-c", probe],
            env=subprocess_env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0, result.stderr


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
