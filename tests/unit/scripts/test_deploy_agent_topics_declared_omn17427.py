# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Every topic the deploy agent publishes must be declared where topics are created.

The broker has auto-create off. The only thing that creates a topic is the
contract-first ``TopicProvisioner``, and it creates only a topic that a contract
or a ``topics.yaml`` manifest declares. The deploy agent is a standalone uv
sub-project with no node and no contract, so it cannot declare its own topics.

On an operator deploy-agent job the agent logged
``Topic onex.evt.deploy.dev-tag-heartbeat.v1 not found in cluster metadata`` and
the publish timed out: ``TOPIC_DEV_TAG_HEARTBEAT`` (OMN-19998) was added to
``deploy_agent.events`` and declared nowhere, so it was never created.

This test reads the constants out of ``events.py`` by AST (the sub-project is not
importable from the root environment) and asks the same extractor the kernel
uses, over the same infra roots the kernel scans, whether each one is declared.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_EVENTS = _REPO / "scripts" / "deploy-agent" / "deploy_agent" / "events.py"
_INFRA_SRC = _REPO / "src" / "omnibase_infra"

# Declared by the counterparty, not here: omnimarket's
# node_redeploy_deploy_effect contract lists these three under
# event_bus.publish_topics / subscribe_topics (rebuild-rejected is also named in
# its externally_produced_topics). omnimarket is not a build dependency of
# omnibase_infra, so its contract cannot be read in this repository's CI.
_DECLARED_BY_OMNIMARKET = frozenset(
    {
        "TOPIC_REBUILD_REQUESTED",
        "TOPIC_REBUILD_COMPLETED",
        "TOPIC_REBUILD_REJECTED",
    }
)


def _agent_topic_constants() -> dict[str, str]:
    """Every module-level ``TOPIC_* = "<string>"`` in the deploy agent's events.py."""
    tree = ast.parse(_EVENTS.read_text(encoding="utf-8"), filename=str(_EVENTS))
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in targets:
            if target.id.startswith("TOPIC_"):
                constants[target.id] = value.value
    return constants


def _provisioned_topics() -> set[str]:
    """Topics the provisioner is handed from this repository's own declarations.

    Mirrors ``service_kernel``: node contracts, plus the ``cli`` and ``services``
    topics.yaml manifest roots.
    """
    entries = ContractTopicExtractor().extract_all(
        contracts_root=_INFRA_SRC / "nodes",
        skill_manifests_roots=[_INFRA_SRC / "cli", _INFRA_SRC / "services"],
    )
    return {entry.topic for entry in entries}


def test_the_constant_derivation_is_not_vacuous() -> None:
    """Positive control: the sweep finds the agent's topics, or it checks nothing."""
    constants = _agent_topic_constants()
    assert "TOPIC_DEV_TAG_HEARTBEAT" in constants, constants
    assert constants.keys() >= _DECLARED_BY_OMNIMARKET, (
        f"{sorted(_DECLARED_BY_OMNIMARKET - constants.keys())} no longer exist in "
        f"{_EVENTS}; drop them from _DECLARED_BY_OMNIMARKET"
    )
    assert _provisioned_topics(), "the provisioner roots yielded no topics at all"


def test_every_deploy_agent_topic_is_declared_for_provisioning() -> None:
    declared = _provisioned_topics()
    undeclared = {
        name: topic
        for name, topic in _agent_topic_constants().items()
        if name not in _DECLARED_BY_OMNIMARKET and topic not in declared
    }
    assert not undeclared, (
        "deploy agent topics declared in no contract and no topics.yaml manifest, "
        "so the TopicProvisioner never creates them and the first publish fails "
        "with 'not found in cluster metadata': "
        f"{undeclared}. Declare them in src/omnibase_infra/services/topics.yaml."
    )
