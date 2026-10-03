# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate --prompt-file`` and ``-`` carry a prompt past the argv cap (OMN-20291).

One command-line argument is capped near 128 KB on Linux, which truncated the
history of the code-edit node's long multi-file tasks. The prompt can now come
from a file or from stdin, and reaches the same request path unchanged. The
positional form is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from click.testing import CliRunner, Result

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command, resolve_prompt_source

pytestmark = pytest.mark.unit

# 200 KB, over the ~128 KB one-argument cap, with multi-byte text, CRLF, a
# trailing newline and surrounding whitespace, none of which may be altered.
_LONG_PROMPT = ("  line one — café\r\nline two\n" * 7000) + "tail\n\n"


@pytest.fixture(autouse=True)
def _no_drift_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)


def _invoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    *,
    stdin: bytes | None = None,
) -> tuple[Result, dict[str, object]]:
    """Run the CLI and return its result and the payload the request path got."""
    captured: dict[str, object] = {}

    def _fake_run_receipt_mode(**kwargs: object) -> int:
        captured.update(
            json.loads(Path(str(kwargs["input_path"])).read_text(encoding="utf-8"))
        )
        return 0

    monkeypatch.setattr(
        cli_delegate, "_resolve_packaged_contract", lambda _n: tmp_path / "c.yaml"
    )
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)
    result = CliRunner().invoke(
        delegate_command,
        [
            *args,
            "--state-root",
            str(tmp_path / "state"),
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ],
        input=stdin,
    )
    return result, captured


def test_the_long_prompt_is_larger_than_one_argument_can_carry() -> None:
    assert len(_LONG_PROMPT.encode("utf-8")) > 200 * 1024


def test_a_200kb_prompt_file_reaches_the_request_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_bytes(_LONG_PROMPT.encode("utf-8"))
    result, captured = _invoke(
        tmp_path, monkeypatch, ["--prompt-file", str(prompt_file)]
    )
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == _LONG_PROMPT


def test_a_200kb_prompt_on_stdin_reaches_the_request_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, captured = _invoke(
        tmp_path,
        monkeypatch,
        ["--prompt-file", "-"],
        stdin=_LONG_PROMPT.encode("utf-8"),
    )
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == _LONG_PROMPT


def test_a_dash_argument_reads_stdin_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, captured = _invoke(
        tmp_path, monkeypatch, ["-"], stdin=_LONG_PROMPT.encode("utf-8")
    )
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == _LONG_PROMPT


def test_the_positional_form_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, captured = _invoke(tmp_path, monkeypatch, ["  keep  my\nspaces \n"])
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == "  keep  my\nspaces \n"


def test_the_positional_form_ignores_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, captured = _invoke(tmp_path, monkeypatch, ["inline"], stdin=b"from stdin")
    assert result.exit_code == 0, result.output
    assert captured["prompt"] == "inline"


def test_both_sources_are_refused() -> None:
    with pytest.raises(click.UsageError, match="not both"):
        resolve_prompt_source("inline", "prompt.md")


def test_no_source_is_refused() -> None:
    with pytest.raises(click.UsageError, match="Missing prompt"):
        resolve_prompt_source(None, None)


def test_a_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(click.UsageError, match="Cannot read the prompt"):
        resolve_prompt_source(None, str(tmp_path / "absent.md"))


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_bytes(b"")
    with pytest.raises(click.UsageError, match="is empty"):
        resolve_prompt_source(None, str(empty))


def test_a_non_utf8_file_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(click.UsageError, match="Cannot read the prompt"):
        resolve_prompt_source(None, str(bad))
