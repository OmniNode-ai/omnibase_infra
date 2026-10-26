# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21436: ``onex node`` guards omnimarket drift only for nodes omnimarket could own.

The drift guard exists so a market-provided node is not dispatched from a stale
or missing omnimarket. ``onex node`` ran it before EVERY node, so the
``ci-live-contact`` pre-commit hook -- which starts ``node_ci_live_contact_effect``,
a node omnibase_infra itself ships -- was refused in every worktree venv, which
correctly carries no omnimarket. These cases pin the scope: an infra-owned node
runs past a drifted canonical clone; a market-provided node and a name that
resolves from no package are still refused.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest

from omnibase_infra.cli import contract_registry
from omnibase_infra.cli.cli_node import run_node_by_name

pytestmark = pytest.mark.unit

_CANONICAL_SHA = "a" * 40
_REACHED_RESOLUTION = "the node reached contract resolution"


def _entry(name: str, distribution: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        value=f"{distribution.replace('-', '_')}.nodes.{name}",
        dist=SimpleNamespace(name=distribution, metadata={"Name": distribution}),
    )


def _run(node_name: str, tmp_path: Path) -> None:
    """Invoke ``onex node`` with omnimarket absent and a canonical clone present."""
    entries = (
        _entry("node_ci_live_contact_effect", "omnibase-infra"),
        _entry("node_aislop_sweep", "omnimarket"),
    )
    with (
        patch.object(contract_registry, "_entry_points", return_value=entries),
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.installed_omnimarket_commit",
            return_value=None,
        ),
        patch(
            "omnibase_infra.cli.omnimarket_drift_guard.canonical_local_omnimarket_commit",
            return_value=_CANONICAL_SHA,
        ),
        patch(
            "omnibase_infra.cli.cli_node.make_workspace_reconciler",
            return_value=None,
        ),
        patch(
            "omnibase_infra.cli.cli_node._resolve_packaged_contract",
            side_effect=click.ClickException(_REACHED_RESOLUTION),
        ),
    ):
        run_node_by_name.callback(
            node_name,
            contract_path=None,
            input_path=None,
            state_root=tmp_path,
            backend=(),
            timeout=1,
            verbose=False,
            output_mode="default",
            emit_socket=None,
            omnibase_path=tmp_path,
            allow_omnimarket_drift=False,
        )


def test_an_infra_owned_node_runs_past_omnimarket_drift(tmp_path: Path) -> None:
    """AC1: the guard is not consulted; the run reaches contract resolution."""
    with pytest.raises(click.ClickException) as raised:
        _run("node_ci_live_contact_effect", tmp_path)

    assert _REACHED_RESOLUTION in str(raised.value)
    assert "NOT INSTALLED" not in str(raised.value)


def test_a_market_node_is_still_refused_on_drift(tmp_path: Path) -> None:
    """AC2: a node omnimarket provides still meets the guard."""
    with pytest.raises(click.ClickException) as raised:
        _run("node_aislop_sweep", tmp_path)

    assert _REACHED_RESOLUTION not in str(raised.value)
    assert "omnimarket" in str(raised.value)


def test_an_unresolvable_node_is_still_refused_on_drift(tmp_path: Path) -> None:
    """AC2: a name no package advertises may be a market node a stale venv lacks."""
    with pytest.raises(click.ClickException) as raised:
        _run("node_not_installed_anywhere", tmp_path)

    assert _REACHED_RESOLUTION not in str(raised.value)
    assert "omnimarket" in str(raised.value)
