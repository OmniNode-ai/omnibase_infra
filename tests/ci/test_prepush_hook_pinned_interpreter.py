# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reject floating bare-`python3`/`python` invocations in lane-executed hooks
(OMN-14953, secondary finding F3).

Root cause: ``scripts/hooks/prepush_smart_tests.sh`` mixes two interpreter
resolution strategies in the same file -- ``uv run python`` (lines 111/118,
resolves through this repo's pinned ``.python-version`` venv on any host) and
a bare ``python3`` (line 135, resolves whatever ``python3`` happens to be
first on ``PATH`` -- unpinned and host-dependent, exactly the class of skew
OMN-14953 root-causes on the ``.200`` lane venv). A hook that runs on every
`git push` on every host (laptop, CI runner, the ``.200`` lane) must not
silently float between two different interpreters depending on which one
gets invoked.

This is a hermetic static scan: no live ``.200``/network access, no
subprocess execution of the scanned scripts.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / "scripts" / "hooks"

_PY_TOKEN_RE = re.compile(r"\bpython3?\b")


def _bare_python_invocations(path: Path) -> list[str]:
    """Return lines invoking python3/python NOT routed through `uv run`.

    A match is exempt only when the token is immediately preceded (ignoring
    whitespace) by ``uv run`` on the same line -- i.e. ``uv run python`` /
    ``uv run python3`` are pinned-venv invocations and pass; a bare
    ``python3 ...`` anywhere else in the file floats to PATH resolution and
    is flagged.
    """
    violations: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            continue  # comments (and shebangs) never execute
        for match in _PY_TOKEN_RE.finditer(raw_line):
            preceding = raw_line[: match.start()].rstrip()
            if preceding.endswith("uv run"):
                continue
            violations.append(raw_line.rstrip())
    return violations


def test_prepush_smart_tests_has_no_floating_python_invocation() -> None:
    """`scripts/hooks/prepush_smart_tests.sh` must route every interpreter
    invocation through `uv run` -- no bare `python3`/`python` on PATH.

    RED today: line 135 (`python3 - "$SELECTION_FILE" "$1" << 'PY'`) floats
    to whatever `python3` resolves first on PATH, independent of the
    `.python-version`-pinned uv venv the rest of the hook uses.
    """
    hook_path = HOOKS_DIR / "prepush_smart_tests.sh"
    assert hook_path.is_file(), f"expected hook script at {hook_path}"

    violations = _bare_python_invocations(hook_path)
    assert not violations, (
        f"{hook_path} invokes a bare python3/python interpreter not routed "
        f"through `uv run` (floats to unpinned PATH resolution): "
        f"{violations!r}"
    )


def test_no_hook_script_has_a_floating_python_invocation() -> None:
    """Every script under scripts/hooks/ must route python calls via `uv run`.

    Broader net-negative-surface guard: covers any future hook script added
    to this directory, not just the one flagged by the OMN-14953 canary.
    """
    assert HOOKS_DIR.is_dir(), f"expected hooks directory at {HOOKS_DIR}"

    hook_scripts = sorted(HOOKS_DIR.glob("*.sh"))
    assert hook_scripts, f"expected at least one *.sh hook under {HOOKS_DIR}"

    all_violations: dict[str, list[str]] = {}
    for script_path in hook_scripts:
        violations = _bare_python_invocations(script_path)
        if violations:
            all_violations[script_path.name] = violations

    assert not all_violations, (
        "the following scripts/hooks/*.sh files invoke a bare python3/python "
        f"interpreter not routed through `uv run`: {all_violations!r}"
    )
