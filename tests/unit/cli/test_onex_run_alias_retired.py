# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The legacy ``onex run`` alias must stay retired (OMN-16761).

Background. OMN-8938 renamed this repo's node runner ``onex run`` -> ``onex node``
and left ``run`` behind as an alias pointing at the *identical* callable. The
alias was never retired. omnibase_core 0.46.13 (OMN-16677) then legitimately
reused the freed name for a different command -- the tier-0 local workflow
harness -- and core's built-in wins the entry-point race:

    onex.cli extension 'run' conflicts with an existing command, skipping

That warning printed on every single CLI invocation in any environment with both
packages installed, and ``onex run <node>`` silently stopped running nodes.

``run`` now belongs to omnibase_core. This repo owns ``node``. These tests pin
that boundary from the side this repo actually controls: they assert what
``omnibase_infra`` *declares*, not how core's harness behaves. Asserting core's
flag surface from here would couple this repo to core's CLI internals and break
on every unrelated harness change -- core tests core's command.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODE_RUNNER_TARGET = "omnibase_infra.cli.cli_node:run_node_by_name"


def _onex_cli_entry_points() -> dict[str, str]:
    """Return this repo's declared ``onex.cli`` entry-point table."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry_points = data["project"]["entry-points"]["onex.cli"]
    assert isinstance(entry_points, dict)
    return entry_points


def test_infra_does_not_declare_an_onex_cli_run_entry_point() -> None:
    """OMN-16761 AC1: no ``run`` entry point, so no collision warning.

    This is the whole defect in one assertion. omnibase_core owns ``onex run``;
    a ``run`` key here is shadowed by core's built-in and only produces noise.
    """
    entry_points = _onex_cli_entry_points()
    assert "run" not in entry_points, (
        "omnibase_infra must not declare an 'onex.cli' entry point named 'run' -- "
        "omnibase_core 0.46.13+ ships a built-in 'onex run' (the tier-0 workflow "
        "harness) that always wins the entry-point race, so this declaration is "
        "dead weight that emits "
        "\"onex.cli extension 'run' conflicts with an existing command, skipping\" "
        f"on every CLI invocation. Use 'onex node'. Got: {entry_points!r}"
    )


def test_infra_still_declares_the_node_entry_point() -> None:
    """OMN-16761 AC3: retiring the alias must not disturb ``onex node``.

    ``node`` is the canonical name from OMN-8938 and the only surviving route to
    this repo's node runner. core ships no built-in ``node``, so nothing shadows it.
    """
    entry_points = _onex_cli_entry_points()
    assert entry_points.get("node") == _NODE_RUNNER_TARGET, (
        f"'onex node' must resolve to {_NODE_RUNNER_TARGET!r}; got "
        f"{entry_points.get('node')!r}"
    )


def test_node_runner_is_reachable_under_exactly_one_name() -> None:
    """No second alias may reintroduce the collision under a different key.

    Guards the *class* of defect, not just the one instance: any future entry
    point re-pointed at the node runner would recreate an OMN-8938-style alias.
    """
    aliases = sorted(
        name
        for name, target in _onex_cli_entry_points().items()
        if target == _NODE_RUNNER_TARGET
    )
    assert aliases == ["node"], (
        "the node runner must be reachable under exactly one 'onex.cli' name "
        f"('node'); found {aliases!r}. Aliasing one callable under two CLI names "
        "is what produced the OMN-16761 collision in the first place."
    )
