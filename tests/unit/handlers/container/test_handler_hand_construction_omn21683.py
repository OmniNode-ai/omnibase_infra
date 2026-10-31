# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handlers and nodes are built at a binding site, never by hand (OMN-21683).

Runs core's contracts-and-DI validator over the modules that used to build a
handler by hand, with a planted construction as the positive control, checks that
each binding site builds the handler it names, and checks that the node packages
whose registry factories nothing called no longer hold one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)
from omnibase_infra.handlers.container.container_config_materializer import (
    bind_workspace_runtime_config_materializer,
)
from omnibase_infra.handlers.container.container_config_root import (
    bind_root_resolve,
)
from omnibase_infra.handlers.container.container_handler_consumer_coordinator_rejoin import (
    bind_handler_consumer_coordinator_rejoin,
)
from omnibase_infra.handlers.container.container_qdrant import (
    bind_handler_qdrant,
)
from omnibase_infra.handlers.container.container_slack import (
    bind_slack_webhook,
)
from omnibase_infra.handlers.handler_consumer_coordinator_rejoin import (
    HandlerConsumerCoordinatorRejoin,
)
from omnibase_infra.handlers.handler_qdrant import HandlerQdrant
from omnibase_infra.handlers.handler_slack_webhook import HandlerSlackWebhook
from omnibase_infra.handlers.handler_workspace_config_root_resolve import (
    HandlerWorkspaceConfigRootResolve,
)
from omnibase_infra.handlers.handler_workspace_runtime_config_materializer import (
    HandlerWorkspaceRuntimeConfigMaterializer,
)
from omnibase_infra.nodes.node_migration_append_only_check_compute.container.container_node_migration_append_only_check import (
    bind_node_migration_append_only_check,
)
from omnibase_infra.nodes.node_migration_append_only_check_compute.handler import (
    NodeMigrationAppendOnlyCheckCompute,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute.container.container_node_migration_freeze_check import (
    bind_node_migration_freeze_check,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute.handler import (
    NodeMigrationFreezeCheckCompute,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute.container.container_node_migration_sequence_check import (
    bind_node_migration_sequence_check,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute.handler import (
    NodeMigrationSequenceCheckCompute,
)
from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.container.container_node_sibling_sha_pin_ratchet import (
    bind_node_sibling_sha_pin_ratchet,
)
from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.handler import (
    NodeSiblingShaPinRatchetCompute,
)

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra"

_RULE = "handler_hand_construction"

_MODULES_WITHOUT_HAND_CONSTRUCTION: tuple[str, ...] = (
    "cli/cli_delegate.py",
    "cli/cli_ledger.py",
    "cli/delegate_lane.py",
    "event_bus/consumer_rejoin_supervisor.py",
    "handlers/handler_workspace_runtime_config_materializer.py",
    "nodes/node_migration_append_only_check_compute/runtime_migration_append_only_check.py",
    "nodes/node_migration_freeze_check_compute/runtime_migration_freeze_check.py",
    "nodes/node_migration_sequence_check_compute/runtime_migration_sequence_check.py",
    "nodes/node_sibling_sha_pin_ratchet_compute/runtime_sibling_sha_pin_ratchet.py",
    "observability/runner_health/cli_runner_health.py",
    "services/waitlist_signup_notifier/consumer.py",
    "topics/contract_topic_extractor.py",
)

_NODES_WITHOUT_REGISTRY_FACTORY: tuple[str, ...] = (
    "node_auth_gate_compute",
    "node_contract_resolver_bridge",
    "node_ledger_projection_compute",
    "node_llm_completion_effect",
    "node_registry_api_effect",
    "node_session_lifecycle_reducer",
    "node_session_state_effect",
    "node_validation_adjudicator",
    "node_validation_executor",
)

_BINDING_SITES: tuple[str, ...] = (
    "handlers/container/container_handler_consumer_coordinator_rejoin.py",
    "handlers/container/container_qdrant.py",
    "handlers/container/container_slack.py",
    "handlers/container/container_ledger_query.py",
    "handlers/container/container_config_root.py",
    "handlers/container/container_config_materializer.py",
    "nodes/node_migration_append_only_check_compute/container/container_node_migration_append_only_check.py",
    "nodes/node_migration_freeze_check_compute/container/container_node_migration_freeze_check.py",
    "nodes/node_migration_sequence_check_compute/container/container_node_migration_sequence_check.py",
    "nodes/node_sibling_sha_pin_ratchet_compute/container/container_node_sibling_sha_pin_ratchet.py",
)


def _rules_reported(path: Path) -> list[str]:
    result = ValidatorContractsAndDi().validate([path])
    return [issue.rule_name or "" for issue in result.issues]


def _constructions_reported(path: Path) -> list[str]:
    result = ValidatorContractsAndDi().validate([path])
    return [issue.message for issue in result.issues if issue.rule_name == _RULE]


def test_the_check_reports_a_planted_construction(tmp_path: Path) -> None:
    planted = tmp_path / "planted_module.py"
    planted.write_text(
        "from somewhere import HandlerPlanted\n\nplanted = HandlerPlanted()\n",
        encoding="utf-8",
    )
    assert _RULE in _rules_reported(planted)


@pytest.mark.parametrize("module", _BINDING_SITES)
def test_binding_site_is_not_reported(module: str) -> None:
    assert (_SRC / module).is_file()
    assert _RULE not in _rules_reported(_SRC / module)


@pytest.mark.parametrize("module", _MODULES_WITHOUT_HAND_CONSTRUCTION)
def test_module_builds_nothing_by_hand(module: str) -> None:
    assert _RULE not in _rules_reported(_SRC / module)


def test_similarity_enrichment_adapter_builds_no_qdrant_handler_by_hand() -> None:
    reported = _constructions_reported(
        _SRC / "adapters/llm/adapter_similarity_enrichment.py"
    )
    assert not any("HandlerQdrant" in message for message in reported)


@pytest.mark.parametrize("node", _NODES_WITHOUT_REGISTRY_FACTORY)
def test_node_package_has_no_dead_registry_factory(node: str) -> None:
    assert not (_SRC / "nodes" / node / "registry").exists()


@pytest.mark.parametrize(
    ("bind", "expected"),
    [
        (bind_handler_consumer_coordinator_rejoin, HandlerConsumerCoordinatorRejoin),
        (bind_slack_webhook, HandlerSlackWebhook),
        (
            bind_root_resolve,
            HandlerWorkspaceConfigRootResolve,
        ),
        (
            bind_workspace_runtime_config_materializer,
            HandlerWorkspaceRuntimeConfigMaterializer,
        ),
        (bind_node_migration_append_only_check, NodeMigrationAppendOnlyCheckCompute),
        (bind_node_migration_freeze_check, NodeMigrationFreezeCheckCompute),
        (bind_node_migration_sequence_check, NodeMigrationSequenceCheckCompute),
        (bind_node_sibling_sha_pin_ratchet, NodeSiblingShaPinRatchetCompute),
    ],
)
def test_binding_site_builds_the_handler_it_names(
    bind: Callable[[], object], expected: type[object]
) -> None:
    assert isinstance(bind(), expected)


def test_slack_binding_site_passes_its_arguments_through() -> None:
    handler = bind_slack_webhook(bot_token="token", default_channel="channel")
    assert isinstance(handler, HandlerSlackWebhook)


def test_qdrant_binding_site_builds_on_the_given_container() -> None:
    assert isinstance(bind_handler_qdrant(MagicMock()), HandlerQdrant)
