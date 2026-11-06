# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Behavioral evidence for the handler request review dispositions."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from omnibase_core.enums import EnumInjectionScope
from omnibase_core.errors.error_duplicate_yaml_mapping_key import (
    DuplicateYamlMappingKeyError,
)
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.utils.util_safe_yaml_loader import load_yaml_mapping_no_duplicates
from omnibase_infra.models.projection.model_projection_derived_write_request import (
    ModelProjectionDerivedWriteRequest,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
    HandlerOnboarding,
    OnboardingHandlerError,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_input import (
    ModelOnboardingInput,
)
from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput
from omnibase_infra.onboarding.protocol_input_adapter import ProtocolInputAdapter
from omnibase_infra.protocols.protocol_projection_write_cursor import (
    ProtocolProjectionWriteCursor,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("policy", "message"),
    [("new_employee", "expected 'interactive'"), ("missing_policy", "not found")],
)
async def test_named_policy_never_falls_back_to_dag(policy: str, message: str) -> None:
    container = ModelONEXContainer()
    await container.service_registry.register_instance(
        interface=ProtocolInputAdapter,
        instance=AdapterFakeInput(responses={}),
        scope=EnumInjectionScope.GLOBAL,
    )
    with pytest.raises(OnboardingHandlerError, match=message):
        await HandlerOnboarding(container).handle(
            ModelOnboardingInput(policy_name=policy)
        )


def _projection_inputs() -> dict[str, object]:
    return {
        "cursor": MagicMock(spec=ProtocolProjectionWriteCursor),
        "physical_schema": "public",
        "table": "events",
        "stored_rows": [{"key": "original"}],
        "conflict_columns": ("key",),
        "input_columns": ("payload",),
        "derived_columns": ("summary",),
        "returning": (),
        "derive": lambda row: {"summary": "derived"},
        "adapt": lambda row: row,
    }


@pytest.mark.parametrize("callback", ["derive", "adapt"])
def test_projection_request_rejects_noncallable_callbacks(callback: str) -> None:
    with pytest.raises(ValidationError, match="callable_type"):
        ModelProjectionDerivedWriteRequest.model_validate(
            _projection_inputs() | {callback: object()}
        )


def test_projection_request_copies_caller_rows_and_keeps_bound_cursor() -> None:
    inputs = _projection_inputs()
    row = {"key": "original"}
    caller_rows = [row]
    request = ModelProjectionDerivedWriteRequest.model_validate(
        inputs | {"stored_rows": caller_rows}
    )
    row["key"] = "changed"
    caller_rows.clear()
    assert request.stored_rows == ({"key": "original"},)
    assert request.stored_rows[0] is not row
    assert request.cursor is inputs["cursor"]


@pytest.mark.parametrize(
    "node",
    [
        "node_validation_executor",
        "node_validation_ledger_write_effect",
        "node_validation_orchestrator",
        "node_scope_file_read_effect",
    ],
)
def test_contract_descriptors_have_unique_keys(node: str) -> None:
    path = ROOT / "src/omnibase_infra/nodes" / node / "contract.yaml"
    text = path.read_text()
    contract = load_yaml_mapping_no_duplicates(text, source=node)
    assert contract["descriptor"]["node_archetype"] == (
        "orchestrator" if node == "node_validation_orchestrator" else "effect"
    )
    if node == "node_validation_executor":
        terminal = contract["terminal_event"]
        assert terminal in contract["event_bus"]["publish_topics"]
        assert terminal in {event["topic"] for event in contract["published_events"]}
    with pytest.raises(
        DuplicateYamlMappingKeyError, match="duplicate key 'descriptor'"
    ):
        load_yaml_mapping_no_duplicates(
            text + "\ndescriptor:\n  node_archetype: compute\n", source=node
        )
