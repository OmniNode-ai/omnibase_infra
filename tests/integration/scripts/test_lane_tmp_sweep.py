# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Linux integration of the shipped tmpfiles rules with systemd-tmpfiles.

This suite requires systemd-tmpfiles with --root support. It deliberately fails
if that integration dependency is absent; the portable rule tests live in the
unit suite and do not invoke the cleaner.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_CONF = (
    Path(__file__).resolve().parents[3] / "deploy/disk-gc/onex-lane-tmp.tmpfiles.conf"
)


def test_cleanup_preserves_live_and_unrelated_trees(tmp_path: Path) -> None:
    """Run the shipped rules against a private filesystem root, never the host temp."""
    root = tmp_path / "root"
    scratch = root / "tmp"
    fixtures = {
        "pytest-of-lane/pytest-0/input.db": "live fixture",
        "pytest-of-lane/pytest-0/.lock": "lock contents are not a liveness check",
        "pytest-of-lane/garbage-retired/nested/input.db": "retired fixture",
        "pytest-of-lane/garbage-empty/.keep": "retired marker",
        "pytest-of-retired-only/garbage-old/input.db": "retired fixture",
        "pytest-of-lane/stray.txt": "unrelated user file",
        "pytest-of-lane/.lock": "unrelated lock",
        "pytest-of-lane/.garbage-hidden/nested/input.db": "unrelated dot tree",
        "pytest-of-lane/custom-run/garbage-nested/input.db": "excluded subtree",
        "pytest-of-lane/custom-run/input.db": "unrelated tree",
        "claude-lane/input.db": "live lane",
        "service-worker/input.db": "service state",
    }
    for relative, contents in fixtures.items():
        path = scratch / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    parent = scratch / "pytest-of-lane"
    preserved_names = ["x"]
    for length in range(1, len("garbage-")):
        prefix = "garbage-"[:length]
        preserved_names.extend((prefix, prefix + "x", prefix + "x-tail"))
    for name in preserved_names:
        (parent / name).write_text("unrelated prefix")
    (parent / "pytest-current").symlink_to("pytest-0", target_is_directory=True)
    (parent / "garbage-already-empty").mkdir()
    (parent / "garbage-").mkdir()
    (scratch / "pytest-of-empty").mkdir()
    config = tmp_path / "cleanup.conf"
    # Exercise the actual 12h configuration before accelerating the old side.
    # New ctimes mean every fixture must survive this first pass unchanged.
    config.write_text(_CONF.read_text())
    fresh_pass = subprocess.run(
        ["systemd-tmpfiles", "--clean", f"--root={root}", str(config)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert fresh_pass.returncode == 0, fresh_pass.stdout + fresh_pass.stderr
    for relative, contents in fixtures.items():
        assert (scratch / relative).read_text() == contents
    for name in ("garbage-already-empty", "garbage-"):
        assert (parent / name).is_dir()
    # Scale only the age; retain the shipped rule types and path globs.
    config.write_text(
        "\n".join(
            " ".join([*rule[:5], "1s" if rule[0] == "e" else "-"])
            for rule in (
                line.split()
                for line in _CONF.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        )
        + "\n"
    )
    # Default aging includes file ctime, which utime cannot backdate. Wait for
    # the measured timestamps, with a full second margin for coarse clocks.
    newest = max(p.stat().st_ctime for p in scratch.rglob("*"))
    deadline = time.monotonic() + 10
    while time.time() <= newest + 2:
        assert time.monotonic() < deadline, (
            "clock did not advance past old fixture timestamps"
        )
        time.sleep(0.05)
    fresh = parent / "garbage-fresh" / "input.db"
    fresh.parent.mkdir()
    fresh.write_text("fresh retired fixture")
    # A future mtime keeps this side of the boundary stable even on a busy host.
    future = time.time() + 3600
    os.utime(fresh, (future, future))
    os.utime(fresh.parent, (future, future))
    cleaned = subprocess.run(
        ["systemd-tmpfiles", "--clean", f"--root={root}", str(config)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert cleaned.returncode == 0, cleaned.stdout + cleaned.stderr
    for relative, contents in fixtures.items():
        if relative.split("/")[1].startswith("garbage-"):
            assert not (scratch / relative).exists(), "retired scratch must be cleaned"
        else:
            assert (scratch / relative).read_text() == contents
    for retired in (
        "garbage-retired",
        "garbage-empty",
        "garbage-already-empty",
        "garbage-",
    ):
        assert not (scratch / "pytest-of-lane" / retired).exists(), (
            "retired directory must be removed, not left empty"
        )
    assert fresh.read_text() == "fresh retired fixture"
    assert parent.is_dir()
    assert (scratch / "pytest-of-empty").is_dir()
    emptied = scratch / "pytest-of-retired-only"
    assert emptied.is_dir()
    assert list(emptied.iterdir()) == []
    assert (parent / "pytest-current").is_symlink()
    assert (parent / "pytest-current").readlink() == Path("pytest-0")
    for name in preserved_names:
        assert (parent / name).read_text() == "unrelated prefix"
