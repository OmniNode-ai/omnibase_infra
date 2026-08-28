# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/lane_scratch.py (OMN-16842).

The defect this file locks down: parallel lanes in one session are handed the
**same** scratchpad directory and independently converge on the same obvious
basenames. Three collisions were observed in a single day (2026-08-27): a
shared log filename, a shared ``msg.txt``, and six lanes each writing a
private ``heavy.py`` wrapper. A collision here does not crash -- it produces a
*plausible* artifact, which is why two of the three were found by a human
reading the result rather than by any mechanism.

The standing answer was prose in the dispatch brief ("use lane-unique
scratchpad filenames"). A rule is not a mechanism: a forgotten prefix silently
reinstates the bug. This helper is the mechanism.

The regression contract proven here:

* the same label minted concurrently by many lanes never collides -- the
  uniqueness is structural (pid + a `secrets` suffix), not a convention;
* a mint whose target already exists FAILS rather than silently reusing the
  path -- creation is ``O_EXCL``, so "collided" and "succeeded" can never be
  the same observable outcome;
* an omitted or empty label still yields an isolated path -- a forgotten
  label degrades to isolated, never to a shared bare name (OMN-15678 AC3);
* the label is recoverable from the minted path, so a stray artifact is
  attributable to the lane that wrote it -- the property whose absence made
  all three incidents undiagnosable.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "lane_scratch.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("lane_scratch", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _run_cli(args: list[str], timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# --------------------------------------------------------------------------
# AC1 -- the helper is a real, committed, runnable script
# --------------------------------------------------------------------------


def test_script_is_committed_and_executable() -> None:
    assert _SCRIPT.is_file()
    assert os.access(_SCRIPT, os.X_OK)


def test_header_records_the_observed_collisions() -> None:
    header = _SCRIPT.read_text(encoding="utf-8")[:4000]
    assert "OMN-16842" in header
    assert "collision" in header.lower()


# --------------------------------------------------------------------------
# AC2(a) -- the SAME label from the SAME root never collides
# --------------------------------------------------------------------------


def test_same_label_many_times_never_collides(tmp_path: Path) -> None:
    """The incident shape: every lane picks the same obvious name.

    All three observed collisions were same-label collisions ("log", "msg",
    "heavy"). So the interesting case is not distinct labels -- it is many
    lanes asking for the identical one.
    """
    minted = [
        MOD.mint_path(root=tmp_path, label="msg", suffix=".txt") for _ in range(200)
    ]
    assert len(set(minted)) == 200
    for path in minted:
        assert path.exists()


def test_concurrent_mints_of_one_label_are_all_distinct(tmp_path: Path) -> None:
    """Same label, genuinely concurrent -- the live condition."""

    def mint(_: int) -> Path:
        return MOD.mint_path(root=tmp_path, label="heavy", suffix=".py")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(mint, range(96)))

    assert len(set(results)) == 96
    assert len({p.name for p in results}) == 96


def test_two_processes_with_the_same_label_do_not_collide(tmp_path: Path) -> None:
    """Cross-process, which is what lanes actually are."""
    first = _run_cli(["--root", str(tmp_path), "--label", "runA", "--suffix", ".txt"])
    second = _run_cli(["--root", str(tmp_path), "--label", "runA", "--suffix", ".txt"])

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout.strip() != second.stdout.strip()
    assert Path(first.stdout.strip()).exists()
    assert Path(second.stdout.strip()).exists()


# --------------------------------------------------------------------------
# AC2(b) -- an existing target FAILS; it is never silently reused
# --------------------------------------------------------------------------


def test_existing_target_is_a_hard_error_not_a_silent_reuse(tmp_path: Path) -> None:
    """``O_EXCL``, so "it collided" can never look like "it worked".

    The randomness makes a real collision vanishingly unlikely; this asserts
    the *behaviour* if one ever happens, because the whole failure class here
    is silent reuse producing a plausible artifact.
    """
    fixed = tmp_path / "already-there.txt"
    fixed.write_text("a peer's content")

    with pytest.raises(FileExistsError):
        MOD.create_exclusive(fixed)

    assert fixed.read_text() == "a peer's content"


def test_cli_reports_a_collision_loudly_with_a_marker(tmp_path: Path) -> None:
    result = _run_cli(["--root", str(tmp_path), "--exact", "taken.txt"])
    assert result.returncode == 0, result.stderr
    again = _run_cli(["--root", str(tmp_path), "--exact", "taken.txt"])
    assert again.returncode == MOD.EXIT_PATH_TAKEN
    assert "lane_scratch:" in again.stderr


# --------------------------------------------------------------------------
# AC2(c) -- a forgotten label degrades to ISOLATED, never to shared
# --------------------------------------------------------------------------


def test_omitted_label_still_yields_an_isolated_path(tmp_path: Path) -> None:
    """OMN-15678 AC3: "scaffold so a forgotten scope defaults to isolated."

    The tempting failure is for a missing label to collapse to a bare, shared
    default name -- which is precisely the bug. It must still carry the
    per-process uniqueness.
    """
    a = MOD.mint_path(root=tmp_path, label=None, suffix=".txt")
    b = MOD.mint_path(root=tmp_path, label="", suffix=".txt")
    c = MOD.mint_path(root=tmp_path, label="   ", suffix=".txt")

    assert len({a, b, c}) == 3
    for path in (a, b, c):
        assert path.name not in ("scratch", "scratch.txt", ".txt", "")


def test_cli_without_a_label_succeeds_and_is_unique(tmp_path: Path) -> None:
    first = _run_cli(["--root", str(tmp_path)])
    second = _run_cli(["--root", str(tmp_path)])
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout.strip() != second.stdout.strip()


# --------------------------------------------------------------------------
# AC2(d) -- the label is recoverable, so a stray artifact is attributable
# --------------------------------------------------------------------------


def test_label_is_recoverable_from_the_minted_path(tmp_path: Path) -> None:
    path = MOD.mint_path(root=tmp_path, label="friction-pair", suffix=".log")
    assert MOD.label_of(path) == "friction-pair"
    assert "friction-pair" in path.name


def test_label_is_sanitised_but_still_recognisable(tmp_path: Path) -> None:
    """A label carries a ticket and spaces; the filename must stay sane."""
    path = MOD.mint_path(root=tmp_path, label="OMN-16842 heavy run/1", suffix=".log")
    assert "/" not in path.name
    assert " " not in path.name
    assert "OMN-16842" in path.name
    assert MOD.label_of(path).startswith("OMN-16842")


def test_pid_is_embedded_for_attribution(tmp_path: Path) -> None:
    path = MOD.mint_path(root=tmp_path, label="lane", suffix=".txt")
    assert str(os.getpid()) in path.name


# --------------------------------------------------------------------------
# Directory mode, and the root-resolution contract
# --------------------------------------------------------------------------


def test_dir_mode_mints_a_lane_private_directory(tmp_path: Path) -> None:
    result = _run_cli(["--root", str(tmp_path), "--label", "lane", "--dir"])
    assert result.returncode == 0, result.stderr
    minted = Path(result.stdout.strip())
    assert minted.is_dir()
    assert MOD.label_of(minted) == "lane"


def test_root_defaults_to_the_env_scratchpad_when_set(tmp_path: Path) -> None:
    env = dict(os.environ, CLAUDE_SCRATCHPAD_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--label", "lane"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).parent == tmp_path


def test_default_root_is_the_workspace_never_slash_tmp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`feedback_no_tmp_use_workspace`: /tmp is not the scratch root.

    Asserted against ``default_root()`` directly rather than by inspecting a
    minted path under ``tmp_path``: pytest's ``tmp_path`` root is itself under
    ``/tmp`` on Linux (and under ``/private/var/folders`` on macOS), so a
    "does the output start with /tmp" check on a ``tmp_path`` root tests the
    platform, not the helper -- it passed locally and failed on the CI runner.
    """
    monkeypatch.delenv("CLAUDE_SCRATCHPAD_DIR", raising=False)
    monkeypatch.delenv("OMNI_HOME", raising=False)
    assert not str(MOD.default_root()).startswith("/tmp")  # noqa: S108

    monkeypatch.setenv("OMNI_HOME", "/some/workspace")
    assert MOD.default_root() == Path("/some/workspace/.onex_state/lane_scratch")

    monkeypatch.setenv("CLAUDE_SCRATCHPAD_DIR", "/scratch/here")
    assert MOD.default_root() == Path("/scratch/here")
