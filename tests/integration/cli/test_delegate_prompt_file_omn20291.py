# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` reads a prompt past the argv cap from a real file and a real pipe (OMN-20291).

The unit tests drive the click command in process. This one starts a separate
interpreter, so the 200 KB prompt arrives through an operating-system pipe and
a file on disk, the two channels the code-edit node uses, and the child reports
a digest of the text the delegate request path would receive.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_PROMPT = ("history line — café\r\n" * 12_000) + "tail\n\n"

_CHILD = (
    "import hashlib, sys\n"
    "from omnibase_infra.cli.cli_delegate import resolve_prompt_source\n"
    "prompt = None if sys.argv[1] == 'None' else sys.argv[1]\n"
    "prompt_file = None if sys.argv[2] == 'None' else sys.argv[2]\n"
    "text = resolve_prompt_source(prompt, prompt_file)\n"
    "sys.stdout.write(hashlib.sha256(text.encode('utf-8')).hexdigest())\n"
)


def _resolve_in_child(
    prompt: str | None, prompt_file: str | None, stdin: bytes = b""
) -> str:
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, str(prompt), str(prompt_file)],
        input=stdin,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return done.stdout.decode("ascii")


def test_the_prompt_is_larger_than_one_argv_word_can_carry() -> None:
    assert len(_PROMPT.encode("utf-8")) > 200 * 1024


def test_a_prompt_file_on_disk_arrives_intact(tmp_path: Path) -> None:
    path = tmp_path / "prompt.md"
    path.write_bytes(_PROMPT.encode("utf-8"))
    expected = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()
    assert _resolve_in_child(None, str(path)) == expected


def test_a_prompt_through_a_pipe_arrives_intact() -> None:
    expected = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()
    assert _resolve_in_child(None, "-", _PROMPT.encode("utf-8")) == expected
    assert _resolve_in_child("-", None, _PROMPT.encode("utf-8")) == expected
