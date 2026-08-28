# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/pr_body.py (OMN-16839).

The defect this file locks down: ``gh pr edit <n> --body ...`` exits 1 on
OmniNode-ai repos because its edit path calls the deprecated Projects-classic
GraphQL API, and **the PR body is not written**. The failure is silent in the
direction that matters -- a caller that does not inspect the exit code
believes the write landed. Observed three separate times; PR bodies are how
``Evidence-Source:`` / ``Evidence-Ticket:`` lines reach the Receipt Gate, so a
dropped write is a missing evidence stamp the author thinks they filed.

The regression contract proven here:

* a write is never reported as successful on the strength of the mutation
  call alone -- the helper re-reads the PR and compares what the API now
  serves against what it intended, so a no-op write is a LOUD failure;
* the helper writes only to the PR the caller named -- an identity mismatch
  between the requested owner/repo/number and the PR the API returns refuses
  the write (the OMN-15564 cross-repo overwrite class);
* ``--append`` preserves the existing body and adds the text exactly once,
  so an evidence-line append does not require the caller to hand-splice;
* every failure the helper originates carries a ``pr_body:`` marker and a
  distinct exit code, so "the helper ran and refused" stays mechanically
  distinguishable from "the helper is not there" -- the lesson OMN-16822
  encoded after ``flock(1)``'s absence produced a markerless 127.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "pr_body.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("pr_body", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# --------------------------------------------------------------------------
# A scripted stand-in for `gh`, so these tests never touch the network.
#
# The fake reads a JSON "world" file: the body it currently serves, plus an
# optional instruction to IGNORE the PATCH (which is exactly what the
# Projects-classic breakage does -- accept the call, change nothing).
# --------------------------------------------------------------------------

_FAKE_GH = r"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

world_path = Path(os.environ["FAKE_GH_WORLD"])
world = json.loads(world_path.read_text())
argv = sys.argv[1:]
world.setdefault("calls", []).append(argv)

def emit(payload):
    sys.stdout.write(json.dumps(payload))

method = "GET"
if "--method" in argv:
    method = argv[argv.index("--method") + 1]
elif "-X" in argv:
    method = argv[argv.index("-X") + 1]

if method == "GET":
    emit({
        "number": world["number"],
        "html_url": world["html_url"],
        "body": world["body"],
    })
else:
    # Read the requested body out of the --input file (or --field body=@file).
    new_body = None
    if "--input" in argv:
        src = argv[argv.index("--input") + 1]
        payload = json.loads(sys.stdin.read() if src == "-" else Path(src).read_text())
        new_body = payload.get("body")
    if world.get("patch_is_a_noop"):
        pass  # the gh pr edit failure mode: accepted, nothing written
    elif new_body is not None:
        world["body"] = new_body
    emit({
        "number": world["number"],
        "html_url": world["html_url"],
        "body": world["body"],
    })

world_path.write_text(json.dumps(world))
"""


def _write_fake_gh(tmp_path: Path) -> Path:
    fake = tmp_path / "fake_gh.py"
    fake.write_text(_FAKE_GH)
    fake.chmod(0o755)
    return fake


def _world(
    tmp_path: Path,
    *,
    body: str = "original body\n",
    number: int = 42,
    html_url: str = "https://github.com/OmniNode-ai/omnibase_infra/pull/42",
    patch_is_a_noop: bool = False,
) -> Path:
    path = tmp_path / "world.json"
    path.write_text(
        json.dumps(
            {
                "number": number,
                "html_url": html_url,
                "body": body,
                "patch_is_a_noop": patch_is_a_noop,
                "calls": [],
            }
        )
    )
    return path


def _run_cli(
    args: list[str], world: Path, tmp_path: Path, timeout: float = 60
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, FAKE_GH_WORLD=str(world))
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--gh-bin",
            f"{sys.executable} {_write_fake_gh(tmp_path)}",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


def _served_body(world: Path) -> str:
    return json.loads(world.read_text())["body"]


# --------------------------------------------------------------------------
# AC1 -- the helper is a real, committed, runnable script
# --------------------------------------------------------------------------


def test_script_is_committed_and_executable() -> None:
    assert _SCRIPT.is_file()
    assert os.access(_SCRIPT, os.X_OK)


def test_header_states_why_gh_pr_edit_is_not_used() -> None:
    header = _SCRIPT.read_text(encoding="utf-8")[:4000]
    assert "gh pr edit" in header
    assert "Projects" in header


def test_helper_never_shells_out_to_gh_pr_edit() -> None:
    """The whole point: the broken verb must not appear as a CALL SITE.

    Prose about ``gh pr edit`` is expected (the header explains why it is not
    used); what must not exist is an argument the helper actually hands to
    ``gh``. So this walks the AST for real string constants -- skipping every
    docstring -- and asserts neither ``pr`` nor ``edit`` is ever passed.
    """
    import ast

    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        )
    }
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }

    # A gh argv element is a bare token, so these two exact literals are what
    # a `gh pr edit` call site would need. Prose mentioning the verb inside a
    # longer --help string is fine and expected; a bare token is not.
    assert "edit" not in literals
    assert "pr" not in literals
    # Positive control: the REST verb IS what it passes.
    assert "api" in literals
    assert "PATCH" in literals


# --------------------------------------------------------------------------
# AC2(a) -- a write that silently does nothing is a LOUD failure
# --------------------------------------------------------------------------


def test_noop_patch_is_detected_by_readback_and_fails_loudly(tmp_path: Path) -> None:
    """The `gh pr edit` failure mode, reproduced: accepted call, no write.

    The helper must not conclude success from the mutation call returning.
    It re-reads the PR and compares; a body that did not change is the defect
    this ticket exists for and must exit non-zero saying so.
    """
    world = _world(tmp_path, body="original body\n", patch_is_a_noop=True)
    result = _run_cli(
        [
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--pr",
            "42",
            "--set",
            "--body",
            "replacement body",
        ],
        world,
        tmp_path,
    )

    assert result.returncode != 0
    assert result.returncode == MOD.EXIT_READBACK_MISMATCH
    assert "pr_body:" in result.stderr
    assert "read-back" in result.stderr.lower() or "readback" in result.stderr.lower()
    # And it did not lie about what is now published.
    assert _served_body(world) == "original body\n"


def test_successful_set_is_confirmed_by_readback(tmp_path: Path) -> None:
    world = _world(tmp_path, body="original body\n")
    result = _run_cli(
        [
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--pr",
            "42",
            "--set",
            "--body",
            "replacement body",
        ],
        world,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert _served_body(world) == "replacement body"


# --------------------------------------------------------------------------
# AC2(b) -- --append preserves the body and adds the text exactly once
# --------------------------------------------------------------------------


def test_append_preserves_existing_body_and_adds_text(tmp_path: Path) -> None:
    world = _world(tmp_path, body="## Summary\n\nsome prose\n")
    result = _run_cli(
        [
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--pr",
            "42",
            "--append",
            "--body",
            "Evidence-Source: OCC#123",
        ],
        world,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    served = _served_body(world)
    assert served.startswith("## Summary\n\nsome prose\n")
    assert "Evidence-Source: OCC#123" in served


def test_append_is_idempotent_and_never_duplicates_the_line(tmp_path: Path) -> None:
    """Re-running an append must not stack a second copy of the same stamp.

    Evidence appends get retried (a lane re-runs its closeout, a sweep passes
    twice). Two `Evidence-Source:` lines in one body is the OMN-14675
    over-match class, so the append must no-op when the text is already there.
    """
    world = _world(tmp_path, body="## Summary\n")
    args = [
        "--repo",
        "OmniNode-ai/omnibase_infra",
        "--pr",
        "42",
        "--append",
        "--body",
        "Evidence-Source: OCC#123",
    ]
    first = _run_cli(args, world, tmp_path)
    assert first.returncode == 0, first.stderr
    second = _run_cli(args, world, tmp_path)
    assert second.returncode == 0, second.stderr

    assert _served_body(world).count("Evidence-Source: OCC#123") == 1


# --------------------------------------------------------------------------
# AC2(c) -- the helper writes only to the PR the caller named
# --------------------------------------------------------------------------


def test_identity_mismatch_refuses_to_write(tmp_path: Path) -> None:
    """OMN-15564's class: a body written into some OTHER repo's PR.

    The API is asked which PR this is; if the served identity is not the
    owner/repo/number the caller named, nothing is written.
    """
    world = _world(
        tmp_path,
        body="untouched\n",
        number=42,
        html_url="https://github.com/OmniNode-ai/onex_change_control/pull/42",
    )
    result = _run_cli(
        [
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--pr",
            "42",
            "--set",
            "--body",
            "would have clobbered a peer repo",
        ],
        world,
        tmp_path,
    )

    assert result.returncode == MOD.EXIT_IDENTITY_MISMATCH
    assert "pr_body:" in result.stderr
    assert "onex_change_control" in result.stderr
    assert _served_body(world) == "untouched\n"


def test_number_mismatch_refuses_to_write(tmp_path: Path) -> None:
    world = _world(
        tmp_path,
        body="untouched\n",
        number=99,
        html_url="https://github.com/OmniNode-ai/omnibase_infra/pull/99",
    )
    result = _run_cli(
        [
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--pr",
            "42",
            "--set",
            "--body",
            "wrong number",
        ],
        world,
        tmp_path,
    )

    assert result.returncode == MOD.EXIT_IDENTITY_MISMATCH
    assert _served_body(world) == "untouched\n"


# --------------------------------------------------------------------------
# AC2(d) -- helper-originated failure is distinguishable from helper-absent
# --------------------------------------------------------------------------


def test_missing_gh_binary_is_distinct_from_a_missing_helper(tmp_path: Path) -> None:
    world = _world(tmp_path)
    env = dict(os.environ, FAKE_GH_WORLD=str(world))
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--gh-bin",
            "definitely-not-a-real-gh-xyz",
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--pr",
            "42",
            "--set",
            "--body",
            "x",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )

    assert result.returncode == MOD.EXIT_GH_UNAVAILABLE
    assert result.returncode != 127
    assert "pr_body:" in result.stderr

    # Negative control: the helper itself being absent is markerless.
    absent = subprocess.run(
        [sys.executable, str(tmp_path / "no_such_helper.py")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert absent.returncode != 0
    assert "pr_body:" not in absent.stderr


def test_set_and_append_are_mutually_exclusive(tmp_path: Path) -> None:
    world = _world(tmp_path)
    result = _run_cli(
        [
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--pr",
            "42",
            "--set",
            "--append",
            "--body",
            "x",
        ],
        world,
        tmp_path,
    )
    assert result.returncode == 2


def test_dry_run_reports_the_intended_body_without_writing(tmp_path: Path) -> None:
    world = _world(tmp_path, body="before\n")
    result = _run_cli(
        [
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--pr",
            "42",
            "--append",
            "--body",
            "Evidence-Ticket: OMN-16839",
            "--dry-run",
        ],
        world,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert _served_body(world) == "before\n"
    assert "Evidence-Ticket: OMN-16839" in result.stdout


# --------------------------------------------------------------------------
# Pure-function coverage of the body composition rule
# --------------------------------------------------------------------------


def test_compose_append_separates_with_a_blank_line() -> None:
    assert MOD.compose_append("a", "b") == "a\n\nb"
    assert MOD.compose_append("a\n", "b") == "a\n\nb"
    assert MOD.compose_append("", "b") == "b"
    assert MOD.compose_append(None, "b") == "b"


def test_compose_append_is_a_noop_when_text_already_present() -> None:
    existing = "## Summary\n\nEvidence-Source: OCC#7\n"
    assert MOD.compose_append(existing, "Evidence-Source: OCC#7") == existing
