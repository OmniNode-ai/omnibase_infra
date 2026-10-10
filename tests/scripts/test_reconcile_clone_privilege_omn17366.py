# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The clone surface writes as its owner, or refuses (OMN-17366).

THE INCIDENT

`/etc/cron.d/omninode-workspace-reconcile` runs as **root**. Every file under
`.201`'s `/data/omninode` is owned by the operator. `reconcile-host.sh` fetched
each clone and ran the clone delegate in-process, as root, into those
operator-owned trees -- so every hourly tick deposited more root-owned objects.
Counted live on 2026-09-01::

    omnibase_infra 572   omnimarket 261   omnibase_compat 150
    omnibase_core  119   omnibase_spi  16          (1118 total)

The resulting failure is intermittent, which is what makes it expensive: a plain
operator `git fetch` breaks only when it needs to write near an object root
owns::

    error: insufficient permission for adding an object to repository database
    .git/objects
    fatal: failed to write object

So the clone looks healthy until it suddenly does not, and the cause is an hour
of cron ticks in the past rather than anything the operator just did.

THIS IS OMN-17335 ONE SURFACE OVER

OMN-17335 established the rule for the **venv** surface: a mutation runs as the
owner of the surface it writes, via `as_owner`, or it refuses. That fix was
still hypothetical when it landed. This one had already materialised.

The rule is therefore shared, not re-implemented: `scripts/reconcile_privilege_lib.sh`
holds the mechanics and both reconcilers source it. Two copies of a privilege
rule drift, and the half that drifts is the half nobody is watching.

WHY `--check` REFUSES HERE, WHILE THE VENV RECONCILER LETS IT THROUGH

The venv reconciler deliberately exempts `--check` from the ownership rule,
reasoning that a read-only probe writes nothing and that "a read-only probe that
refuses teaches people to stop running it."

That reasoning does not transfer, because **this script's check mode is not
read-only**. It calls `fetch_all` before verdicting -- it must, since a verifier
that takes its target from the thing under verification is not a verifier -- and
`git fetch` writes objects, refs and reflogs. A `--check` that deposits
root-owned objects is the very hazard this ticket is about, so here the guard
covers both modes. `test_check_mode_is_also_refused_because_it_fetches` pins
that divergence so it cannot be "tidied" into agreement with the venv script.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB = _REPO_ROOT / "scripts" / "reconcile_privilege_lib.sh"
_GATE = _REPO_ROOT / "scripts" / "check_reconciler_privilege.py"

_FOREIGN = "someone-else"


def test_both_reconcilers_source_the_one_privilege_library() -> None:
    """OMN-17366's central requirement, asserted structurally.

    The ticket is explicit: route the clone surface through the *same* guard
    rather than inventing a second one. Two implementations of a privilege rule
    drift, and nobody is watching the copy that drifts. This fails if either
    script grows its own ``as_owner``.
    """
    lib_name = _LIB.name
    for script in ("reconcile-workspace-venvs.sh",):
        source = (_REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert lib_name in source, f"{script} does not source {lib_name}"
        assert "as_owner() {" not in source, (
            f"{script} defines its own as_owner instead of using the shared "
            f"{lib_name} -- that is the second implementation OMN-17366 forbids"
        )


# --------------------------------------------------------------------------- #
# AC4 -- the gate is extended, not duplicated
# --------------------------------------------------------------------------- #
def _gate(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(_GATE), "--repo-root", str(repo_root)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_gate_passes_on_the_real_repository() -> None:
    result = _gate(_REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def _fixture_repo(tmp_path: Path, host_body: str) -> Path:
    """A minimal repo the gate can scan, carrying a stand-in reconcile-host."""
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(
        _REPO_ROOT / "scripts" / "reconcile-workspace-venvs.sh",
        scripts / "reconcile-workspace-venvs.sh",
    )
    shutil.copy2(_LIB, scripts / _LIB.name)
    (scripts / "reconcile-host.sh").write_text(host_body, encoding="utf-8")
    return root


def test_the_gate_rejects_a_clone_fetch_that_skips_the_owner_helper(
    tmp_path: Path,
) -> None:
    """The exact pre-fix line, rejected.

    This is the invocation that deposited 1118 root-owned paths on `.201`.
    """
    root = _fixture_repo(
        tmp_path,
        "#!/usr/bin/env bash\n"
        'source "$SCRIPT_DIR/reconcile_privilege_lib.sh"\n'
        'git -C "$OMNI_HOME/$repo" fetch --quiet --prune origin "$BRANCH"\n',
    )

    result = _gate(root)

    assert result.returncode == 1
    assert "fetch" in result.stderr
    assert "as_owner" in result.stderr


def test_the_gate_rejects_a_clone_delegate_that_skips_the_owner_helper(
    tmp_path: Path,
) -> None:
    """Fetching as the owner while the delegate still runs as root fixes nothing.

    The delegate does its own fetch and checkout, so it is the larger of the two
    write paths. A gate that only covered the in-process fetch would pass the
    script while the damage continued.
    """
    root = _fixture_repo(
        tmp_path,
        "#!/usr/bin/env bash\n"
        'source "$SCRIPT_DIR/reconcile_privilege_lib.sh"\n'
        'as_owner git -C "$OMNI_HOME/$repo" fetch --prune origin "$BRANCH"\n'
        'env OMNI_HOME="$OMNI_HOME" bash "$CLONE_DELEGATE"\n',
    )

    result = _gate(root)

    assert result.returncode == 1
    assert "CLONE_DELEGATE" in result.stderr or "delegate" in result.stderr


def test_the_gate_does_not_flag_a_git_command_quoted_in_a_message(
    tmp_path: Path,
) -> None:
    """Refusals print the command to run by hand; documentation is not invocation.

    Every script in this family names the exact command in its error text, so a
    gate that matched on content alone would flag its own help output -- and a
    gate that cries wolf gets an allowlist bolted on, which is how enforcement
    dies.
    """
    root = _fixture_repo(
        tmp_path,
        "#!/usr/bin/env bash\n"
        'source "$SCRIPT_DIR/reconcile_privilege_lib.sh"\n'
        'as_owner git -C "$c" fetch --prune origin "$BRANCH"\n'
        'say "  run: git -C /data/omninode/omnibase_infra fetch origin dev"\n',
    )

    result = _gate(root)

    assert result.returncode == 0, result.stdout + result.stderr
