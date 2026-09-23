# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The packaged CLI's workspace-root parameter is ``OMNIBASE_PATH`` (OMN-16852).

## What this pins and why it is a test rather than a convention

Three packaged ``onex`` commands -- ``delegate``, ``node`` and ``skill`` --
each declare a workspace-root option bound to an environment variable. That
binding is the single parameter a person who is not the maintainer exports to
point the omnimarket drift guard at a workspace root, and it is load-bearing
by construction: no caller passes the flag explicitly, so an unbound option
hands the guard ``None`` and the check never fires on the canonical-clone
path (the OMN-14531 / OMN-14560 regression, pinned separately in
``test_cli_skill.py`` and ``test_cli_node.py``).

OMN-16855 shipped the product name for that root -- ``OMNIBASE_PATH``,
fail-fast, no default -- in ``omniclaude``'s ``plugins/onex/ENVIRONMENT_VARIABLES.md``,
and recorded there that these three bindings still read the older
``OMNI_HOME`` spelling "until OMN-16852 lands", instructing readers to export
both names for one directory in the interim. This module is what ends that
interim: it refuses the old spelling on the CLI boundary and proves the new
one actually reaches the guard.

## Scope fence -- this is a PARAMETER rename, not a repository-wide one

Only the environment variable a caller sets moves. The option's own spelling
(since renamed ``--omnibase-path`` by OMN-19197), the ``omni_home`` keyword threaded through the guard, and
every read that resolves the maintainer's own multi-repo registry checkout
(the workspace reconciler, dispatch-venv purity, lane declarations, the
machine-registry export) keep ``OMNI_HOME`` by the 2026-08-28 boundary ruling
under OMN-16849. A test that swept those in would be enforcing the opposite
of the ruling.

## Why each test drives behaviour rather than reading source

``test_no_command_binds_the_retired_spelling`` reads click's own parsed
parameter table, so it cannot be satisfied by a comment or missed by one.
The three per-command tests then set ONLY the new variable, with the old one
deliberately removed from the environment, and assert the guard's verdict
flips -- an option that merely *exists* is exactly the failure OMN-14531
found, so existence is not the property worth pinning.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_delegate, cli_node, cli_skill
from omnibase_infra.cli.cli_delegate import delegate_command
from omnibase_infra.cli.cli_node import run_node_by_name
from omnibase_infra.cli.cli_skill import run_skill_by_name
from omnibase_infra.cli.omnimarket_drift_guard import check_omnimarket_drift

#: The product name for the workspace root (OMN-16855, OMN-16849).
WORKSPACE_ROOT_ENVVAR = "OMNIBASE_PATH"

#: The maintainer-workspace name this parameter must no longer read.
RETIRED_ENVVAR = "OMNI_HOME"

#: An arbitrary commit the stubbed canonical clone reports, so the installed
#: side (stubbed to absent) cannot accidentally agree with it.
_DRIFT_SEAM_SHA = "cccccccccccccccccccccccccccccccccccccccc"

_COMMANDS: tuple[tuple[str, click.Command], ...] = (
    ("delegate", delegate_command),
    ("node", run_node_by_name),
    ("skill", run_skill_by_name),
)


def _option_envvars(command: click.Command) -> dict[str, object]:
    """Map each option's primary flag to whatever env var click bound to it."""
    return {
        param.opts[0]: param.envvar
        for param in command.params
        if isinstance(param, click.Option) and param.envvar is not None
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "command"), list(_COMMANDS), ids=[n for n, _ in _COMMANDS]
)
def test_no_command_binds_the_retired_spelling(
    name: str, command: click.Command
) -> None:
    """No option on any of the three commands may bind ``OMNI_HOME``.

    Enumerated from click's parsed parameter table rather than from the
    source text, so a rename that leaves the old name in a help string
    passes and a rename that leaves it in a binding fails.
    """
    bound = _option_envvars(command)
    offenders = sorted(
        flag for flag, envvar in bound.items() if envvar == RETIRED_ENVVAR
    )
    assert offenders == [], (
        f"`onex {name}` still binds the retired {RETIRED_ENVVAR} spelling on "
        f"{offenders}. The workspace-root parameter is {WORKSPACE_ROOT_ENVVAR} "
        "(OMN-16855); only maintainer-registry reads keep the old name."
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "command"), list(_COMMANDS), ids=[n for n, _ in _COMMANDS]
)
def test_workspace_root_option_binds_the_product_name(
    name: str, command: click.Command
) -> None:
    """``--omnibase-path`` binds ``OMNIBASE_PATH`` on all three commands."""
    bound = _option_envvars(command)
    assert bound.get("--omnibase-path") == WORKSPACE_ROOT_ENVVAR, (
        f"`onex {name} --omnibase-path` binds {bound.get('--omnibase-path')!r}; "
        f"expected {WORKSPACE_ROOT_ENVVAR!r}."
    )


def _force_drift(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    """Restore the real guard (conftest stubs it) in a drifted state.

    ``installed_omnimarket_commit`` returning ``None`` with a canonical clone
    present is the "omnimarket is not installed from git" refusal, which is a
    determinable state on the canonical-clone path since OMN-14531. The
    canonical side is stubbed to a fixed sha so the branch is reached only
    when the guard actually received a workspace root.
    """
    monkeypatch.setattr(module, "check_omnimarket_drift", check_omnimarket_drift)
    monkeypatch.setattr(
        "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
        lambda: None,
    )
    monkeypatch.setattr(
        "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
        lambda omni_home=None: _DRIFT_SEAM_SHA if omni_home else None,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "command", "module", "args"),
    [
        ("node", run_node_by_name, cli_node, ["definitely_not_a_real_node"]),
        ("skill", run_skill_by_name, cli_skill, ["definitely_not_a_real_skill"]),
    ],
    ids=["node", "skill"],
)
def test_product_name_alone_reaches_the_guard(
    name: str,
    command: click.Command,
    module: object,
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ONLY ``OMNIBASE_PATH`` must make the guard fire.

    The old name is removed from the environment first, so a dual-read would
    not rescue this and a passing result means the new binding carried the
    value on its own.
    """
    monkeypatch.delenv(RETIRED_ENVVAR, raising=False)
    monkeypatch.setenv(WORKSPACE_ROOT_ENVVAR, "/fake/workspace-root")
    _force_drift(monkeypatch, module)

    result = CliRunner().invoke(command, args)

    combined = result.output + str(result.exception or "")
    assert "NOT INSTALLED" in combined, (
        f"`onex {name}` did not reach the drift guard with only "
        f"{WORKSPACE_ROOT_ENVVAR} set, so the binding does not carry the "
        f"value. Output: {combined[:400]}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "command", "module", "args"),
    [
        ("node", run_node_by_name, cli_node, ["definitely_not_a_real_node"]),
        ("skill", run_skill_by_name, cli_skill, ["definitely_not_a_real_skill"]),
    ],
    ids=["node", "skill"],
)
def test_retired_name_alone_no_longer_reaches_the_guard(
    name: str,
    command: click.Command,
    module: object,
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ONLY the retired name must NOT feed the guard a workspace root.

    This is the clean break stated rather than assumed. Off registry the
    guard reports and never refuses (OMN-17255), so the observable is that the
    canonical-clone refusal does not appear -- the run proceeds to its own
    downstream error instead.
    """
    monkeypatch.delenv(WORKSPACE_ROOT_ENVVAR, raising=False)
    monkeypatch.setenv(RETIRED_ENVVAR, "/fake/workspace-root")
    _force_drift(monkeypatch, module)

    result = CliRunner().invoke(command, args)

    combined = result.output + str(result.exception or "")
    assert "NOT INSTALLED" not in combined, (
        f"`onex {name}` still resolved a workspace root from the retired "
        f"{RETIRED_ENVVAR} spelling, so the break is not clean. "
        f"Output: {combined[:400]}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "command", "module", "args"),
    [
        ("node", run_node_by_name, cli_node, ["definitely_not_a_real_node"]),
        ("skill", run_skill_by_name, cli_skill, ["definitely_not_a_real_skill"]),
    ],
    ids=["node", "skill"],
)
def test_unset_workspace_root_takes_the_off_registry_path(
    name: str,
    command: click.Command,
    module: object,
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With neither name set the guard reports off registry and never refuses.

    This is the third arm of the set / unset / wrong-path trio. It is the
    behaviour OMN-17255 installed and it is what makes the clean break above
    defensible: a machine that exports nothing is not silently unguarded, it
    is compared against the pins packaged inside the installed artifacts.
    """
    monkeypatch.delenv(WORKSPACE_ROOT_ENVVAR, raising=False)
    monkeypatch.delenv(RETIRED_ENVVAR, raising=False)
    _force_drift(monkeypatch, module)

    result = CliRunner().invoke(command, args)

    combined = result.output + str(result.exception or "")
    assert "NOT INSTALLED" not in combined, (
        f"`onex {name}` refused on the canonical-clone path with no "
        f"workspace root set; that path needs a clone. Output: {combined[:400]}"
    )


@pytest.mark.unit
def test_repair_command_carries_the_resolved_root_not_a_variable_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The co-install repair command must not depend on the reader's exports.

    ``install-node-skill-package.sh`` still reads ``OMNI_HOME`` -- its only
    invokers are the workspace reconciler and the drift-check script, both
    maintainer-registry tooling that keeps the older spelling by the
    OMN-16849 boundary ruling, and renaming it would break a reconciler that
    passes the old name explicitly. So on a machine that exports only
    ``OMNIBASE_PATH`` the two names disagree, and a repair command naming a
    variable would be wrong for exactly the reader who most needs it. The
    refusal therefore spells the resolved path out as an assignment.
    """
    from omnibase_infra.cli.omnimarket_drift_guard import OmnimarketDriftError

    workspace = "/fake/workspace-root"
    monkeypatch.setattr(
        "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
        lambda: None,
    )
    monkeypatch.setattr(
        "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
        lambda omni_home=None: _DRIFT_SEAM_SHA,
    )

    with pytest.raises(OmnimarketDriftError) as exc_info:
        check_omnimarket_drift(omni_home=workspace)

    message = str(exc_info.value)
    assert f"OMNI_HOME={workspace} " in message, (
        "the co-install repair command must carry the resolved workspace root "
        f"as an explicit assignment. Message: {message[:600]}"
    )
