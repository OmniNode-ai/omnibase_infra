# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for handler class existence validation (OMN-12408).

Verifies that broken handler_routing entries — ones that reference a handler
class name or module that does not actually exist — produce a loud
ProtocolConfigurationError rather than being silently skipped.

Part of OMN-12408: Runtime auto-wiring silently skips entire contract on one
broken handler entry — should fail loud.

Test Categories:
    - TestLoadHandlerClassInfoFailLoud: load_handler_class_info_from_contract
      must raise on missing/broken handler name/module fields.
    - TestValidateHandlerClassExists: standalone validator raises when the
      named class is absent from the declared module.
    - TestCIGateAllContractsHaveValidHandlerClasses: CI gate scans every
      real contract.yaml in the repo and asserts every named handler class
      actually exists in the declared module.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.contract_loaders.handler_routing_loader import (
    load_handler_class_info_from_contract,
    validate_handler_class_exists,
)

# ---------------------------------------------------------------------------
# YAML fixtures
# ---------------------------------------------------------------------------

_CONTRACT_MISSING_HANDLER_NAME = """
name: "test_node"
node_type: "ORCHESTRATOR_GENERIC"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "ModelSomeEvent"
        module: "omnibase_infra.models.registration.model_node_introspection_event"
      handler:
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected"
"""

_CONTRACT_MISSING_HANDLER_MODULE = """
name: "test_node"
node_type: "ORCHESTRATOR_GENERIC"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "ModelSomeEvent"
        module: "omnibase_infra.models.registration.model_node_introspection_event"
      handler:
        name: "HandlerNodeIntrospected"
"""

_CONTRACT_NONEXISTENT_CLASS = """
name: "test_node"
node_type: "ORCHESTRATOR_GENERIC"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "ModelSomeEvent"
        module: "omnibase_infra.models.registration.model_node_introspection_event"
      handler:
        name: "HandlerThatNeverExisted"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected"
"""

_CONTRACT_NONEXISTENT_MODULE = """
name: "test_node"
node_type: "ORCHESTRATOR_GENERIC"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "ModelSomeEvent"
        module: "omnibase_infra.models.registration.model_node_introspection_event"
      handler:
        name: "HandlerNodeIntrospected"
        module: "omnibase_infra.totally.nonexistent.module"
"""

_CONTRACT_VALID = """
name: "test_node"
node_type: "ORCHESTRATOR_GENERIC"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "ModelNodeIntrospectionEvent"
        module: "omnibase_infra.models.registration.model_node_introspection_event"
      handler:
        name: "HandlerNodeIntrospected"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected"
"""


# ---------------------------------------------------------------------------
# TestLoadHandlerClassInfoFailLoud
# ---------------------------------------------------------------------------


class TestLoadHandlerClassInfoFailLoud:
    """load_handler_class_info_from_contract must raise on broken entries.

    Before OMN-12408 the function silently logged a warning and skipped
    entries with missing handler.name or handler.module.  The fix must
    raise ProtocolConfigurationError so the startup defect is surfaced
    immediately rather than hiding behind an empty consumer group.
    """

    def test_missing_handler_name_raises(self, tmp_path: Path) -> None:
        """Entry missing handler.name must raise ProtocolConfigurationError."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(_CONTRACT_MISSING_HANDLER_NAME)
        with pytest.raises(ProtocolConfigurationError, match=r"handler\.name"):
            load_handler_class_info_from_contract(contract)

    def test_missing_handler_module_raises(self, tmp_path: Path) -> None:
        """Entry missing handler.module must raise ProtocolConfigurationError."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(_CONTRACT_MISSING_HANDLER_MODULE)
        with pytest.raises(ProtocolConfigurationError, match=r"handler\.module"):
            load_handler_class_info_from_contract(contract)

    def test_nonexistent_class_raises(self, tmp_path: Path) -> None:
        """Entry naming a class that does not exist in the module must raise."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(_CONTRACT_NONEXISTENT_CLASS)
        with pytest.raises(ProtocolConfigurationError, match="CLASS_NOT_FOUND"):
            load_handler_class_info_from_contract(contract)

    def test_nonexistent_module_raises(self, tmp_path: Path) -> None:
        """Entry naming a module that cannot be imported must raise."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(_CONTRACT_NONEXISTENT_MODULE)
        with pytest.raises(ProtocolConfigurationError, match="MODULE_NOT_FOUND"):
            load_handler_class_info_from_contract(contract)

    def test_valid_entry_succeeds(self, tmp_path: Path) -> None:
        """A fully correct handler entry must succeed and return class info."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(_CONTRACT_VALID)
        result = load_handler_class_info_from_contract(contract)
        assert len(result) == 1
        assert result[0]["handler_class"] == "HandlerNodeIntrospected"
        assert "handler_node_introspected" in result[0]["handler_module"]


# ---------------------------------------------------------------------------
# TestValidateHandlerClassExists
# ---------------------------------------------------------------------------


class TestValidateHandlerClassExists:
    """validate_handler_class_exists raises on broken module/class pairs.

    This is the standalone validator that CI uses to audit contracts without
    starting the runtime.
    """

    def test_existing_class_does_not_raise(self) -> None:
        """Known-good class must not raise."""
        validate_handler_class_exists(
            handler_class="HandlerNodeIntrospected",
            handler_module="omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected",
            contract_path=Path("synthetic"),
        )

    def test_missing_class_raises(self) -> None:
        """Class absent from an otherwise-importable module must raise."""
        with pytest.raises(ProtocolConfigurationError, match="CLASS_NOT_FOUND"):
            validate_handler_class_exists(
                handler_class="HandlerThatNeverExisted",
                handler_module="omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected",
                contract_path=Path("synthetic"),
            )

    def test_missing_module_raises(self) -> None:
        """Non-importable module must raise MODULE_NOT_FOUND."""
        with pytest.raises(ProtocolConfigurationError, match="MODULE_NOT_FOUND"):
            validate_handler_class_exists(
                handler_class="HandlerFoo",
                handler_module="omnibase_infra.totally.nonexistent.module",
                contract_path=Path("synthetic"),
            )

    def test_error_includes_contract_path(self, tmp_path: Path) -> None:
        """Error message must name the contract that declared the broken entry."""
        contract = tmp_path / "some_node" / "contract.yaml"
        contract.parent.mkdir(parents=True, exist_ok=True)
        contract.write_text("")
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_handler_class_exists(
                handler_class="HandlerMissing",
                handler_module="omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected",
                contract_path=contract,
            )
        assert "some_node" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TestCIGateAllContractsHaveValidHandlerClasses
# ---------------------------------------------------------------------------


class TestCIGateAllContractsHaveValidHandlerClasses:
    """CI gate: every handler_routing entry in every repo contract resolves.

    Scans src/omnibase_infra/nodes/**/contract.yaml and asserts that every
    handler.name / handler.module pair can be imported and the class exists.
    This is the canonical contract-validation check required by OMN-12408 DoD.
    """

    def test_all_deployed_handler_classes_exist(self) -> None:
        """All handler_routing entries in shipped contracts must resolve."""
        import yaml

        repo_root = Path(__file__).parents[
            4
        ]  # contract_loaders/runtime/unit/tests/../.. = repo root
        nodes_dir = repo_root / "src" / "omnibase_infra" / "nodes"

        if not nodes_dir.exists():
            pytest.skip(f"Nodes directory not found: {nodes_dir}")

        failures: list[str] = []

        for contract_path in sorted(nodes_dir.glob("**/contract.yaml")):
            try:
                raw_text = contract_path.read_text(encoding="utf-8")
            except OSError:
                continue

            try:
                contract = yaml.safe_load(raw_text)
            except yaml.YAMLError:
                continue

            if not isinstance(contract, dict):
                continue

            handler_routing = contract.get("handler_routing")
            if not isinstance(handler_routing, dict):
                continue

            handlers = handler_routing.get("handlers", [])
            if not isinstance(handlers, list):
                continue

            for entry in handlers:
                if not isinstance(entry, dict):
                    continue

                # Two handler_routing schemas coexist in the repo:
                #   payload_type_match: handler.name + handler.module (nested)
                #   operation_match:    handler_class + handler_module (flat)
                # Both must be validated; entries that use neither schema are skipped
                # (they have no class to verify and are not the broken-class pattern).
                handler_info = entry.get("handler", {})
                if isinstance(handler_info, dict):
                    # payload_type_match nested schema
                    class_name = handler_info.get("name")
                    module_name = handler_info.get("module")
                else:
                    class_name = None
                    module_name = None

                # operation_match flat schema overrides if not found in nested
                if not class_name:
                    class_name = entry.get("handler_class")
                if not module_name:
                    module_name = entry.get("handler_module")

                if not class_name or not module_name:
                    # Entry has neither schema populated — skip (no class to check).
                    continue

                try:
                    module = importlib.import_module(module_name)
                except ImportError as e:
                    failures.append(
                        f"{contract_path}: cannot import module {module_name!r}: {e}"
                    )
                    continue

                if not hasattr(module, class_name):
                    failures.append(
                        f"{contract_path}: class {class_name!r} not found"
                        f" in module {module_name!r}"
                    )

        assert not failures, (
            f"{len(failures)} handler_routing entries reference broken "
            f"handler classes:\n" + "\n".join(f"  - {f}" for f in failures)
        )
