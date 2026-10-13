# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The two GitHub source-control nodes omnimarket owns are not advertised here — OMN-20871.

omnimarket#3734 moved ``node_github_pr_poller_effect`` and
``node_github_webhook_ingress_effect`` into omnimarket and registered both as
omnimarket ``onex.nodes`` entry points. The runtime image installs omnimarket
beside this package, so while this ``pyproject.toml`` still advertised the same
two names, auto-wiring discovery saw each contract twice:

    DUPLICATE_REGISTRATION prevented: contract='node_github_pr_poller_effect'
    first_owner='omnimarket' duplicate_package='omnibase_infra'

Each refusal is a discovery error, the runtime health monitor reports its
``discovery_errors`` dimension ``DEGRADED`` while that count is above zero, and
the laptop profile's ``omninode-runtime`` healthcheck runs with
``--degraded-policy fail``. The "Boot the laptop profile (make up-local)" step
failed merge-group run 38390505398.

This repository's own test venv does not install omnimarket, so no other test
here can see the collision. The modules stay importable: the released
omnimarket still imports the PR-state observation model from this package's
webhook node, so deleting them waits for an omnimarket release that carries
the move. Only the entry points that make this package claim ownership
are removed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[4] / "pyproject.toml"

_OWNED_BY_OMNIMARKET = (
    "node_github_pr_poller_effect",
    "node_github_webhook_ingress_effect",
)


def _advertised_nodes() -> dict[str, str]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    nodes: dict[str, str] = data["project"]["entry-points"]["onex.nodes"]
    return nodes


@pytest.mark.unit
def test_pyproject_advertises_onex_nodes() -> None:
    """Positive control: the table this file reads is the one discovery scans."""
    nodes = _advertised_nodes()
    assert nodes["node_db_error_linear_effect"] == (
        "omnibase_infra.nodes.node_db_error_linear_effect"
    )


@pytest.mark.unit
@pytest.mark.parametrize("contract_name", _OWNED_BY_OMNIMARKET)
def test_moved_scm_node_is_not_an_omnibase_infra_entry_point(
    contract_name: str,
) -> None:
    """A second owner of the same contract name degrades the runtime's /health."""
    assert contract_name not in _advertised_nodes()
