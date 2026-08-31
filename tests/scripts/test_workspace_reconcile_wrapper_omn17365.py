# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The `.201` wrapper must run the reconciler from the tree it reconciles (OMN-17365).

THE INCIDENT

`deploy/maintenance/omninode-workspace-reconcile.sh` resolved the reconciler
path from ``OMNI_HOME`` **before** sourcing the alert env file -- and that file,
on `.201`, assigns ``OMNI_HOME``. Under ``set -a`` the assignment took effect for
the ``--omni-home`` argument but not for the already-computed path, so the run
split in two:

    executed:    /data/omninode/omni_home/omnibase_infra/scripts/   (default)
    reconciled:  /data/omninode/omnibase_infra/                     (sourced)

Observed live on 2026-08-31 running the cron command as root::

    [reconcile-host]   clone:omnibase_infra: MOVED (53ff3bbb9f1a -> 620160848118)
    [reconcile-host] venv surface: delegating to
        /data/omninode/omni_home/omnibase_infra/scripts/reconcile-workspace-venvs.sh

WHY IT MATTERED MORE THAN IT LOOKED

The wrapper's own header accepts running from a clone on one stated ground:

    "a stale clone runs a stale reconciler. It is bounded -- the reconciler's
     first act is to advance the clones, so the next tick runs the current code."

That bound holds only when the tree it runs from is a tree it advances. It was
not: nothing advances `/data/omninode/omni_home`, so every tick re-ran identical
stale code and no merged fix could ever reach the host. OMN-17335 merged at
`6201608`, was present in the reconciled tree, and still would never have
executed.

There is no symptom. Clones advance, surfaces get verdicts, a receipt is
written, and the log looks completely normal -- which is why it survived from
OMN-17311 until a merged fix visibly failed to take effect.
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


def _make_tree(root: Path, marker: str) -> Path:
    """A tree shaped like `$OMNI_HOME`, whose reconciler announces which one it is."""
    scripts = root / "omnibase_infra" / "scripts"
    scripts.mkdir(parents=True)
    reconciler = scripts / "reconcile-host.sh"
    reconciler.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "RAN_FROM={marker}\\n"\n'
        'printf "ARGV=%s\\n" "$*"\n'
        'printf "ENV_OMNI_HOME=%s\\n" "${OMNI_HOME:-<unset>}"\n',
        encoding="utf-8",
    )
    reconciler.chmod(0o755)
    return root


def _run(env_file: Path, ambient_omni_home: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "OMNINODE_ALERT_ENV_FILE": str(env_file),
        "OMNI_HOME": str(ambient_omni_home),
    }
    return subprocess.run(
        ["bash", str(_WRAPPER)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_the_sourced_env_file_decides_which_tree_the_reconciler_runs_from(
    tmp_path: Path,
) -> None:
    """The `.201` shape exactly: ambient says one root, the env file says another.

    RED against the shipped wrapper, which printed ``RAN_FROM=ambient`` while
    passing ``--omni-home <sourced>`` -- running one checkout against another's
    tree, which is the entire defect.
    """
    ambient = _make_tree(tmp_path / "ambient", "ambient")
    sourced = _make_tree(tmp_path / "sourced", "sourced")

    env_file = tmp_path / "alert.env"
    env_file.write_text(f"OMNI_HOME={sourced}\n", encoding="utf-8")

    result = _run(env_file, ambient)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RAN_FROM=sourced" in result.stdout, (
        "the wrapper executed the reconciler from the ambient root while the "
        "sourced env file had overridden OMNI_HOME -- the OMN-17365 split"
    )
    assert "RAN_FROM=ambient" not in result.stdout


def test_the_executed_tree_and_the_omni_home_argument_always_agree(
    tmp_path: Path,
) -> None:
    """The invariant, stated directly: one resolved root, used for both.

    Asserting only "it ran from the sourced tree" would still pass a wrapper that
    handed the child a *third* root. What must hold is that the tree it runs from
    and the tree it is told to reconcile are the same one.
    """
    ambient = _make_tree(tmp_path / "ambient", "ambient")
    sourced = _make_tree(tmp_path / "sourced", "sourced")

    env_file = tmp_path / "alert.env"
    env_file.write_text(f"OMNI_HOME={sourced}\n", encoding="utf-8")

    result = _run(env_file, ambient)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"--omni-home {sourced}" in result.stdout
    assert f"ENV_OMNI_HOME={sourced}" in result.stdout
    assert "RAN_FROM=sourced" in result.stdout


def test_an_env_file_that_sets_no_omni_home_leaves_the_ambient_root_in_force(
    tmp_path: Path,
) -> None:
    """The common case must not regress: no override means no change.

    Most hosts' alert env files carry only alert credentials. Reordering the
    resolution must not make those hosts start reconciling somewhere else.
    """
    ambient = _make_tree(tmp_path / "ambient", "ambient")

    env_file = tmp_path / "alert.env"
    env_file.write_text("SLACK_CHANNEL_ID=C123\n", encoding="utf-8")

    result = _run(env_file, ambient)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "RAN_FROM=ambient" in result.stdout
    assert f"--omni-home {ambient}" in result.stdout


def test_a_missing_reconciler_still_fails_closed(tmp_path: Path) -> None:
    """Moving the resolution must not weaken the existing refusal."""
    empty = tmp_path / "empty"
    empty.mkdir()
    env_file = tmp_path / "alert.env"
    env_file.write_text("SLACK_CHANNEL_ID=C123\n", encoding="utf-8")

    result = _run(env_file, empty)

    assert result.returncode == 3
    assert "no reconciler at" in result.stderr


# --------------------------------------------------------------------------- #
# The gate (CLAUDE.md rule 5): the ordering itself is asserted, not just its effect
# --------------------------------------------------------------------------- #
def test_reconciler_is_assigned_after_the_env_file_is_sourced() -> None:
    """Static ratchet on the shipped wrapper.

    The behavioural tests above prove the wrapper works today. This one prevents
    the specific edit that broke it: hoisting the ``RECONCILER=`` assignment back
    above the sourcing block, where it silently reads a pre-override
    ``OMNI_HOME``. That edit looks like harmless tidying -- grouping the variable
    definitions at the top is what most people would call a cleanup -- which is
    precisely why it needs a check rather than a comment.
    """
    source = _WRAPPER.read_text(encoding="utf-8")

    assign = re.search(r"^RECONCILER=", source, re.MULTILINE)
    assert assign is not None, "the wrapper no longer assigns RECONCILER"

    sourcing = re.search(r'^\s*\.\s+"\$ALERT_ENV_FILE"', source, re.MULTILINE)
    assert sourcing is not None, "the wrapper no longer sources the alert env file"

    assert assign.start() > sourcing.end(), (
        "RECONCILER is assigned BEFORE the alert env file is sourced. That file "
        "may assign OMNI_HOME (it does on .201), so the reconciler path would be "
        "taken from the pre-override root while --omni-home takes the "
        "post-override one -- running one checkout against another tree, which "
        "is OMN-17365."
    )


def test_omni_home_default_is_applied_after_the_sourcing_too() -> None:
    """The default and the path must be resolved from the same point.

    Splitting them -- default early, path late -- reintroduces the same class of
    bug in the other direction.
    """
    source = _WRAPPER.read_text(encoding="utf-8")

    default = re.search(r'^OMNI_HOME="\$\{OMNI_HOME:-', source, re.MULTILINE)
    assert default is not None, "the wrapper no longer applies an OMNI_HOME default"

    sourcing = re.search(r'^\s*\.\s+"\$ALERT_ENV_FILE"', source, re.MULTILINE)
    assert sourcing is not None

    assert default.start() > sourcing.end(), (
        "the OMNI_HOME default is applied before the env file is sourced, so a "
        "host relying on the env file's value would be overridden by the default"
    )
