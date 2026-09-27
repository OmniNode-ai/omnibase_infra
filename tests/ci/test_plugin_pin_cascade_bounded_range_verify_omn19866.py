# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""plugin-pin-cascade.yml's version-advance check must accept a bounded range.

OMN-18596 (``scripts/update-plugin-pins.py``'s ``_replace_pin``) rewrote a
range-pinned package to a bounded range that keeps its ceiling and advances
only its floor (``PKG>=FLOOR,<CEILING``), specifically so the Dockerfile
plugin pin validator does not reject the result. The "Verify pin advanced to
expected version" step never followed: it only ever grepped for an exact
``PKG==VERSION`` pin, so ``ACTUAL`` was always empty for a range-pinned
package and every cascade run failed at this step even though
``update-plugin-pins.py`` had correctly advanced the floor. Reproduced live:
https://github.com/OmniNode-ai/omnibase_infra/actions/runs/36321217998.

Pinned by EXECUTION, the same way
``test_plugin_pin_cascade_attributable_omn18596.py`` pins the input-validation
and base-resolver programs: the delimited verify script is lifted out of the
shipped YAML and run against a fixture ``docker/Dockerfile.runtime``.

Ticket: OMN-19866 (parent: OMN-18596)
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "plugin-pin-cascade.yml"

_VERIFY_OPEN = "# >>> OMN-19866 plugin pin verify >>>"
_VERIFY_CLOSE = "# <<< OMN-19866 plugin pin verify <<<"

# CI's self-hosted runners are Linux with GNU grep, where `grep -oP` (the
# production script's own tool) supports the lookbehind/lookahead this step
# needs. macOS ships BSD grep, which rejects `-P` outright, so a local run of
# this exact shipped script fails here for a reason unrelated to the
# behaviour under test. A tiny `grep` shim ahead of PATH gives `-oP` the same
# semantics on both platforms without changing the shipped step at all.
_GREP_SHIM_SOURCE = """#!/usr/bin/env python3
import re
import sys

args = sys.argv[1:]
opts = ""
positional: list[str] = []
for arg in args:
    if arg.startswith("-") and arg != "-":
        opts += arg[1:]
    else:
        positional.append(arg)
if "o" not in opts or "P" not in opts or len(positional) != 2:
    sys.exit(2)
pattern, path = positional
with open(path, encoding="utf-8") as fh:
    text = fh.read()
matches = re.findall(pattern, text)
if not matches:
    sys.exit(1)
for m in matches:
    print(m)
sys.exit(0)
"""
_GREP_SHIM_DIR = Path(tempfile.mkdtemp(prefix="omn19866-grep-shim-"))
_grep_shim_path = _GREP_SHIM_DIR / "grep"
_grep_shim_path.write_text(_GREP_SHIM_SOURCE, encoding="utf-8")
_grep_shim_path.chmod(_grep_shim_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)


def _text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _verify_program() -> str:
    body = _text()
    start = body.index(_VERIFY_OPEN)
    end = body.index(_VERIFY_CLOSE, start)
    lines = [
        line[10:] if line.startswith(" " * 10) else line
        for line in body[start:end].splitlines()
    ]
    program = "\n".join(lines)
    # No GitHub Actions expressions should remain inside the delimited
    # program: PKG and EXPECTED_VER are supplied via the step's own `env:`
    # block, never interpolated inline, so this program is directly
    # executable and directly testable.
    assert not re.search(r"\$\{\{", program), program
    return program


def _run(
    package: str, expected_version: str, dockerfile_contents: str, cwd: Path
) -> subprocess.CompletedProcess[str]:
    docker_dir = cwd / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    (docker_dir / "Dockerfile.runtime").write_text(dockerfile_contents)
    return subprocess.run(
        ["bash", "-c", _verify_program()],
        cwd=cwd,
        env={
            "PATH": os.pathsep.join(
                [str(_GREP_SHIM_DIR), "/usr/bin", "/bin", "/usr/local/bin"]
            ),
            "PKG": package,
            "EXPECTED_VER": expected_version,
        },
        capture_output=True,
        text=True,
        check=False,
    )


class TestVerifyStepIsDelimitedAndClean:
    def test_verify_step_is_lifted_out_of_a_run_block(self) -> None:
        # Sanity check that the markers actually bound a `run: |` script and
        # not some other part of the file.
        text = _text()
        start = text.index(_VERIFY_OPEN)
        preceding = text[:start]
        assert preceding.rstrip().endswith("run: |")


class TestExactPinStillWorks:
    """The pre-OMN-19866 behaviour for an exact `PKG==VERSION` pin is unchanged."""

    def test_exact_pin_matching_expected_version_passes(self, tmp_path: Path) -> None:
        result = _run(
            "omninode-claude",
            "0.15.0",
            'RUN pip install "omninode-claude==0.15.0"\n',
            tmp_path,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Pin confirmed" in result.stdout

    def test_exact_pin_at_stale_version_fails(self, tmp_path: Path) -> None:
        result = _run(
            "omninode-claude",
            "0.15.0",
            'RUN pip install "omninode-claude==0.14.9"\n',
            tmp_path,
        )
        assert result.returncode != 0
        assert "did not reach expected version" in result.stderr


class TestBoundedRangePositiveControl:
    """OMN-19866 AC1: a bounded range whose floor advanced to the dispatched
    version must pass — this is the exact shape update-plugin-pins.py writes
    for a range-pinned package, and the shape that made every cascade run
    fail before this fix.
    """

    def test_bounded_range_floor_matching_expected_version_passes(
        self, tmp_path: Path
    ) -> None:
        result = _run(
            "omninode-memory",
            "0.18.3",
            'RUN uv pip install --no-deps "omninode-memory>=0.18.3,<1.0.0"\n',
            tmp_path,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        assert "Pin confirmed" in result.stdout

    def test_v_prefixed_dispatch_version_matches_bare_floor(
        self, tmp_path: Path
    ) -> None:
        # The dispatched version may carry a release-tag 'v' prefix.
        result = _run(
            "omninode-memory",
            "v0.18.3",
            'RUN uv pip install --no-deps "omninode-memory>=0.18.3,<1.0.0"\n',
            tmp_path,
        )
        assert result.returncode == 0, result.stderr + result.stdout


class TestBoundedRangeNegativeControl:
    """OMN-19866 AC2: the fixed check must still refuse a genuine non-advance,
    so the fix is not a no-op that always passes a bounded range.
    """

    def test_unchanged_bounded_range_floor_still_fails(self, tmp_path: Path) -> None:
        # The floor never advanced (e.g. PyPI propagation lag left
        # update-plugin-pins.py fetching the same version as before) — the
        # Dockerfile still carries the pre-cascade floor while the dispatch
        # expected a newer one.
        result = _run(
            "omninode-memory",
            "0.18.3",
            'RUN uv pip install --no-deps "omninode-memory>=0.18.2,<1.0.0"\n',
            tmp_path,
        )
        assert result.returncode != 0
        assert "did not reach expected version" in result.stderr
        assert "Pin confirmed" not in result.stdout

    def test_no_pin_for_the_package_at_all_fails_closed(self, tmp_path: Path) -> None:
        result = _run(
            "omninode-memory",
            "0.18.3",
            'RUN uv pip install --no-deps "omninode-claude>=0.15.0,<1.0.0"\n',
            tmp_path,
        )
        assert result.returncode != 0
        assert "did not reach expected version" in result.stderr
