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
_SSH_TOKEN_RE = re.compile(r"\bssh\b")


def _bare_python_invocations(path: Path) -> list[str]:
    """Return lines invoking python3/python NOT routed through `uv run`.

    A match is exempt when the token is immediately preceded (ignoring
    whitespace) by ``uv run`` on the same line -- i.e. ``uv run python`` /
    ``uv run python3`` are pinned-venv invocations and pass; a bare
    ``python3 ...`` anywhere else in the file floats to PATH resolution and
    is flagged.

    A match is also exempt when an ``ssh`` token appears earlier in the same
    *logical* line (OMN-16333). Such an invocation executes on a REMOTE host,
    where this repo's uv venv does not exist -- ``ssh .201 "uv run python3
    ..."`` fails with *command not found*. The invariant this scan protects
    is local PATH skew on the pushing host; it is unsatisfiable by
    construction for a remote command, so requiring ``uv run`` there would
    forbid the call rather than pin it.

    Continuations matter: the real ssh probe puts ``ssh`` and the remote
    ``python3`` on different physical lines joined by a trailing backslash,
    so tokens are matched against the accumulated logical line rather than
    the physical one. Violations are still reported as the physical line, so
    the failure message points at an editable location.
    """
    violations: list[str] = []
    logical_prefix = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            logical_prefix = ""
            continue  # comments (and shebangs) never execute
        for match in _PY_TOKEN_RE.finditer(raw_line):
            preceding = (logical_prefix + raw_line[: match.start()]).rstrip()
            if preceding.endswith("uv run"):
                continue
            if _SSH_TOKEN_RE.search(preceding):
                continue  # runs on a remote host; see docstring
            violations.append(raw_line.rstrip())
        # A trailing backslash continues this command onto the next line.
        logical_prefix = (
            logical_prefix + raw_line[:-1] + " "
            if raw_line.rstrip().endswith("\\")
            else ""
        )
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


def test_local_bare_python_is_still_flagged(tmp_path: Path) -> None:
    """The OMN-16333 ssh carve-out must not blunt the original guard.

    A bare local `python3` is the exact skew OMN-14953 exists to catch, and
    stays a violation whether or not the file also contains ssh calls
    elsewhere.
    """
    script = tmp_path / "hook.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'raw="$(python3 -c "$PROBE")"\n'
        'ssh "$target" "python3 -c \'$PROBE\'"\n',
        encoding="utf-8",
    )

    violations = _bare_python_invocations(script)

    assert len(violations) == 1, f"expected exactly the local line, got {violations!r}"
    assert "ssh" not in violations[0]
    assert "python3 -c" in violations[0]


def test_ssh_embedded_python_is_exempt(tmp_path: Path) -> None:
    """A python invocation inside an ssh remote command is out of scope.

    It runs on the remote host, which has no uv venv for this repo, so the
    pinned-interpreter invariant cannot apply to it.
    """
    script = tmp_path / "hook.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'raw="$(timeout 6 ssh -o BatchMode=yes "$target" "python3 -c \'$PROBE\'")"\n',
        encoding="utf-8",
    )

    assert _bare_python_invocations(script) == []


def test_ssh_exemption_spans_a_line_continuation(tmp_path: Path) -> None:
    """The real probe splits `ssh` and the remote `python3` across a `\\`.

    Scanning physical lines would miss the ssh context entirely and flag the
    continuation line -- which is exactly how this gate deadlocked the repo.
    """
    script = tmp_path / "hook.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'raw="$(timeout 6 ssh -o ConnectTimeout=3 -o BatchMode=yes \\\n'
        '  "$target" "python3 -c \'$PROBE\'" 2> /dev/null)" || return 1\n',
        encoding="utf-8",
    )

    assert _bare_python_invocations(script) == []


def test_continuation_does_not_leak_exemption_to_later_commands(tmp_path: Path) -> None:
    """An ssh command that has ended must not exempt a later local python3."""
    script = tmp_path / "hook.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'raw="$(ssh -o BatchMode=yes \\\n'
        '  "$target" "uptime")"\n'
        'local_raw="$(python3 -c "$PROBE")"\n',
        encoding="utf-8",
    )

    violations = _bare_python_invocations(script)

    assert len(violations) == 1, f"expected the local line only, got {violations!r}"
    assert "local_raw" in violations[0]


def test_uv_run_python_is_exempt(tmp_path: Path) -> None:
    """The original `uv run` exemption is unchanged."""
    script = tmp_path / "hook.sh"
    script.write_text(
        '#!/usr/bin/env bash\nraw="$(uv run python3 -c "$PROBE")"\n',
        encoding="utf-8",
    )

    assert _bare_python_invocations(script) == []
