# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The `.201` scheduler wrapper must forward argv, and must own the root (OMN-17336).

THE INCIDENT

``deploy/maintenance/omninode-workspace-reconcile.sh`` (OMN-17311, shipped in
omnibase_infra PR #3071) ended with a fixed exec that never forwarded ``"$@"``::

    exec env OMNI_HOME="$OMNI_HOME" bash "$RECONCILER" \\
      --omni-home "$OMNI_HOME" \\
      --branch "${RECONCILE_BRANCH:-dev}"

So every argument handed to the wrapper was discarded without a word. Reproduced
on `.201` on 2026-08-31 while verifying the OMN-17311 deployment. The intent was
a read-only probe::

    sudo -n /data/maintenance/bin/omninode-workspace-reconcile.sh --check

It ran in **repair** mode and fast-forwarded all five deploy-source clones to
``origin/dev``. The mutation was correct and sanctioned -- exactly what the
hourly cron does at :19, and no lane or container was touched. The defect is
that a caller asked for an observation and silently got a mutation.

WHY IT BELONGS TO THIS EPIC

OMN-17305 exists because reconciliation reported outcomes it had not verified.
"You asked for ``--check`` and got a repair" is the same family of surprise, and
``reconcile-host.sh --check`` is the read-only mode operators are pointed at by
the runbook. A cron wrapper that takes no arguments is defensible; one that
accepts them and throws them away is not.

WHY FORWARDING IS SAFE

``scripts/reconcile-host.sh`` rejects an unknown argument with
``EXIT_INDETERMINATE`` rather than ignoring it, so a typo forwarded through this
wrapper stays loud instead of becoming a surprise repair. That premise is what
makes blanket forwarding correct, so it is asserted here rather than assumed.

WHY ``--omni-home`` IS THE ONE ARGUMENT THAT IS REFUSED

Forwarding argv blindly would re-open OMN-17365 through a different door. The
wrapper execs the reconciler *from* the tree it resolved, so a caller-supplied
``--omni-home`` would make the reconciler run from one checkout while
reconciling another -- the precise split OMN-17365 closed, and whose invariant
``test_workspace_reconcile_wrapper_omn17365.py`` pins. Root resolution is this
wrapper's job (sourcing the env file that overrides it is most of what it does),
so an argument that contradicts it is refused with a typed message, not honoured
and not silently dropped.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER = _REPO_ROOT / "deploy" / "maintenance" / "omninode-workspace-reconcile.sh"
_RECONCILER = _REPO_ROOT / "scripts" / "reconcile-host.sh"

# `reconcile-host.sh`'s own exit code for a configuration it cannot resolve.
_EXIT_INDETERMINATE = 3


def _make_tree(root: Path) -> Path:
    """A tree shaped like `$OMNI_HOME`, whose reconciler echoes the argv it got."""
    scripts = root / "omnibase_infra" / "scripts"
    scripts.mkdir(parents=True)
    reconciler = scripts / "reconcile-host.sh"
    reconciler.write_text(
        "#!/usr/bin/env bash\n"
        # Newline-separated, so an assertion can pin an argument's exact
        # position rather than substring-matching a flattened "$*".
        'printf "ARG=%s\\n" "$@"\n',
        encoding="utf-8",
    )
    reconciler.chmod(0o755)
    return root


def _run(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    """Invoke the wrapper with an empty-ish environment, as cron does."""
    env_file = root / "alert.env"
    env_file.write_text("SLACK_CHANNEL_ID=C123\n", encoding="utf-8")
    env = {
        **os.environ,
        "OMNINODE_ALERT_ENV_FILE": str(env_file),
        "OMNI_HOME": str(root),
    }
    return subprocess.run(
        ["bash", str(_WRAPPER), *argv],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _args(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [
        line.removeprefix("ARG=")
        for line in result.stdout.splitlines()
        if line.startswith("ARG=")
    ]


# --------------------------------------------------------------------------- #
# The defect
# --------------------------------------------------------------------------- #
def test_check_reaches_the_reconciler(tmp_path: Path) -> None:
    """RED against the shipped wrapper: ``--check`` never arrived.

    This is the reported incident reduced to one assertion. Before the fix the
    forwarded argv was exactly ``--omni-home <root> --branch dev`` -- a full
    repair -- no matter what the caller typed.
    """
    root = _make_tree(tmp_path / "root")

    result = _run(root, "--check")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--check" in _args(result), (
        "the wrapper dropped --check on the floor: the caller asked for a "
        "read-only observation and the reconciler was handed a repair"
    )


def test_every_forwarded_argument_arrives_not_just_the_first(tmp_path: Path) -> None:
    """Forwarding is blanket, not a special case for one flag.

    A wrapper that pattern-matched ``--check`` alone would pass the test above
    while still swallowing ``--verbose`` -- the flag an operator reaches for when
    the reconciler is misbehaving, i.e. exactly when silence is most expensive.
    """
    root = _make_tree(tmp_path / "root")

    result = _run(root, "--check", "--verbose")

    assert result.returncode == 0, result.stdout + result.stderr
    forwarded = _args(result)
    assert "--check" in forwarded
    assert "--verbose" in forwarded


def test_caller_arguments_come_after_the_wrappers_defaults(tmp_path: Path) -> None:
    """Order is the precedence rule, so it is asserted rather than left to luck.

    ``reconcile-host.sh`` parses left to right and lets the last occurrence win.
    Forwarding argv AFTER the wrapper's own ``--branch`` therefore makes an
    explicit caller branch beat the ``RECONCILE_BRANCH`` default. Prepending
    would silently invert that and hand the caller the default instead.
    """
    root = _make_tree(tmp_path / "root")

    result = _run(root, "--branch", "main")

    assert result.returncode == 0, result.stdout + result.stderr
    forwarded = _args(result)
    assert forwarded.index("main") > forwarded.index("dev"), (
        f"caller's --branch must be parsed last so it wins; got {forwarded}"
    )


def test_the_no_argument_cron_invocation_is_unchanged(tmp_path: Path) -> None:
    """The scheduled path is the one that must not regress.

    /etc/cron.d/omninode-workspace-reconcile invokes this with no arguments at
    all. Forwarding an empty ``"$@"`` must add nothing -- under ``set -u`` a
    mis-written expansion here would abort the hourly reconcile entirely.
    """
    root = _make_tree(tmp_path / "root")

    result = _run(root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert _args(result) == ["--omni-home", str(root), "--branch", "dev"]


# --------------------------------------------------------------------------- #
# The one argument that must NOT be honoured
# --------------------------------------------------------------------------- #
def test_a_caller_supplied_omni_home_is_refused(tmp_path: Path) -> None:
    """Refused loudly -- neither honoured nor silently dropped.

    The wrapper execs the reconciler from the root it resolved. Honouring a
    caller's ``--omni-home`` would run one checkout against another tree, which
    is OMN-17365 exactly. Dropping it quietly would be OMN-17336 exactly. The
    only remaining answer is to refuse and say why.
    """
    root = _make_tree(tmp_path / "root")
    other = _make_tree(tmp_path / "other")

    result = _run(root, "--omni-home", str(other))

    assert result.returncode == _EXIT_INDETERMINATE, result.stdout + result.stderr
    assert "--omni-home" in result.stderr
    assert _args(result) == [], "the reconciler must not run at all on a refusal"


@pytest.mark.parametrize(
    "equals_form", [True, False], ids=["--omni-home=X", "--omni-home X"]
)
def test_both_spellings_of_omni_home_are_refused(
    tmp_path: Path, equals_form: bool
) -> None:
    """``--omni-home=X`` is the same argument as ``--omni-home X``.

    ``reconcile-host.sh`` accepts both spellings, so a refusal that only knew the
    space-separated one would leave the equals form as a silent way back into the
    split root.
    """
    root = _make_tree(tmp_path / "root")
    other = _make_tree(tmp_path / "other")

    argv = [f"--omni-home={other}"] if equals_form else ["--omni-home", str(other)]
    result = _run(root, *argv)

    assert result.returncode == _EXIT_INDETERMINATE, result.stdout + result.stderr
    assert _args(result) == []


# --------------------------------------------------------------------------- #
# The premise that makes blanket forwarding safe
# --------------------------------------------------------------------------- #
def test_the_reconciler_rejects_an_unknown_argument(tmp_path: Path) -> None:
    """Asserted, not assumed.

    Blanket forwarding is only safe because a typo cannot be ignored downstream.
    If ``reconcile-host.sh`` ever grew a permissive catch-all, forwarding would
    quietly turn ``--dry-run`` (a flag it does not have) into a full repair --
    the original incident wearing a different flag. This pins the premise so that
    change goes red here.
    """
    result = subprocess.run(
        ["bash", str(_RECONCILER), "--not-a-real-flag"],
        capture_output=True,
        text=True,
        env={**os.environ, "OMNI_HOME": str(tmp_path)},
        check=False,
    )

    assert result.returncode == _EXIT_INDETERMINATE
    assert "unknown argument" in result.stderr


# --------------------------------------------------------------------------- #
# The gate (CLAUDE.md rule 5): the forwarding is asserted structurally too
# --------------------------------------------------------------------------- #
def test_the_exec_line_forwards_argv() -> None:
    """Static ratchet on the shipped wrapper.

    The behavioural tests above prove today's wrapper forwards. This one names
    the specific regression: an edit that tidies the exec back into a fixed
    argument list. That is what shipped in OMN-17311 and it read as perfectly
    reasonable -- a cron entry point with a fixed invocation -- which is why the
    property needs a check rather than a comment.
    """
    source = _WRAPPER.read_text(encoding="utf-8")

    exec_line = re.search(r"^exec env OMNI_HOME=.*?$", source, re.MULTILINE | re.DOTALL)
    assert exec_line is not None, "the wrapper no longer execs the reconciler"

    tail = source[exec_line.start() :]
    assert '"$@"' in tail, (
        'the exec does not forward "$@". Every argument handed to this wrapper '
        "would be discarded without a word, so --check would perform a repair "
        "(OMN-17336)."
    )
