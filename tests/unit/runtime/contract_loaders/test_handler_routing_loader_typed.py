# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for typed contract loading via load_and_validate_contract_yaml().

OMN-9746: Wires model_validate() into handler_routing_loader as the spine
of systemic contract validation. Validates that the new public function
returns a typed ModelContractNodeType rather than a raw dict, ensuring all
426 node contracts are validated at load time.
"""

from __future__ import annotations

import pytest
import yaml

from omnibase_core.enums.enum_node_type import EnumNodeType
from omnibase_infra.runtime.contract_loaders.handler_routing_loader import (
    ModelContractNodeType,
    _dispatch_contract_model,
    load_and_validate_contract_yaml,
)


def _write_contract(tmp_path, data: dict) -> object:
    yaml_path = tmp_path / "contract.yaml"
    yaml_path.write_text(yaml.dump(data))
    return yaml_path


class TestLoadAndValidateContractYaml:
    """load_and_validate_contract_yaml() returns ModelContractNodeType, not raw dict."""

    def test_effect_generic_returns_typed_model(self, tmp_path):
        yaml_path = _write_contract(
            tmp_path,
            {
                "name": "test_effect_node",
                "node_type": "EFFECT_GENERIC",
                "contract_version": {"major": 1, "minor": 0, "patch": 0},
                "description": "Test effect node",
                "input_model": "some.InputModel",
                "output_model": "some.OutputModel",
                "handler_routing": {
                    "routing_strategy": "payload_type_match",
                    "handlers": [],
                },
            },
        )
        result = load_and_validate_contract_yaml(yaml_path)
        assert isinstance(result, ModelContractNodeType)
        assert result.node_type == EnumNodeType.EFFECT_GENERIC

    def test_compute_generic_returns_typed_model(self, tmp_path):
        yaml_path = _write_contract(
            tmp_path,
            {
                "name": "test_compute_node",
                "node_type": "COMPUTE_GENERIC",
                "contract_version": {"major": 1, "minor": 0, "patch": 0},
                "description": "Test compute node",
                "input_model": "some.InputModel",
                "output_model": "some.OutputModel",
                "handler_routing": {
                    "routing_strategy": "payload_type_match",
                    "handlers": [],
                },
            },
        )
        result = load_and_validate_contract_yaml(yaml_path)
        assert isinstance(result, ModelContractNodeType)
        assert result.node_type == EnumNodeType.COMPUTE_GENERIC

    def test_reducer_generic_returns_typed_model(self, tmp_path):
        yaml_path = _write_contract(
            tmp_path,
            {
                "name": "test_reducer_node",
                "node_type": "REDUCER_GENERIC",
                "contract_version": {"major": 1, "minor": 0, "patch": 0},
                "description": "Test reducer node",
                "input_model": "some.InputModel",
                "output_model": "some.OutputModel",
                "handler_routing": {
                    "routing_strategy": "payload_type_match",
                    "handlers": [],
                },
            },
        )
        result = load_and_validate_contract_yaml(yaml_path)
        assert isinstance(result, ModelContractNodeType)
        assert result.node_type == EnumNodeType.REDUCER_GENERIC

    def test_orchestrator_generic_returns_typed_model(self, tmp_path):
        yaml_path = _write_contract(
            tmp_path,
            {
                "name": "test_orchestrator_node",
                "node_type": "ORCHESTRATOR_GENERIC",
                "contract_version": {"major": 1, "minor": 0, "patch": 0},
                "description": "Test orchestrator node",
                "input_model": "some.InputModel",
                "output_model": "some.OutputModel",
                "handler_routing": {
                    "routing_strategy": "payload_type_match",
                    "handlers": [],
                },
            },
        )
        result = load_and_validate_contract_yaml(yaml_path)
        assert isinstance(result, ModelContractNodeType)
        assert result.node_type == EnumNodeType.ORCHESTRATOR_GENERIC

    def test_lowercase_effect_alias_accepted(self, tmp_path):
        yaml_path = _write_contract(
            tmp_path,
            {
                "name": "test_node",
                "node_type": "effect",
                "contract_version": {"major": 1, "minor": 0, "patch": 0},
                "description": "Test",
                "input_model": "some.Input",
                "output_model": "some.Output",
                "handler_routing": {
                    "routing_strategy": "payload_type_match",
                    "handlers": [],
                },
            },
        )
        result = load_and_validate_contract_yaml(yaml_path)
        assert result.node_type == EnumNodeType.EFFECT_GENERIC

    def test_result_is_not_raw_dict(self, tmp_path):
        yaml_path = _write_contract(
            tmp_path,
            {
                "name": "test_node",
                "node_type": "EFFECT_GENERIC",
                "contract_version": {"major": 1, "minor": 0, "patch": 0},
                "description": "Test",
                "input_model": "some.Input",
                "output_model": "some.Output",
                "handler_routing": {
                    "routing_strategy": "payload_type_match",
                    "handlers": [],
                },
            },
        )
        result = load_and_validate_contract_yaml(yaml_path)
        assert not isinstance(result, dict), "Expected typed model, not raw dict"

    def test_result_carries_raw_contract_for_downstream(self, tmp_path):
        contract = {
            "name": "test_node",
            "node_type": "COMPUTE_GENERIC",
            "contract_version": {"major": 1, "minor": 0, "patch": 0},
            "description": "Test",
            "input_model": "some.Input",
            "output_model": "some.Output",
            "handler_routing": {
                "routing_strategy": "payload_type_match",
                "handlers": [],
            },
        }
        yaml_path = _write_contract(tmp_path, contract)
        result = load_and_validate_contract_yaml(yaml_path)
        assert result.raw == contract


class TestDispatchContractModel:
    """_dispatch_contract_model() validates node_type and raises on unknown."""

    def test_dispatch_effect_generic(self):
        raw = {"node_type": "EFFECT_GENERIC", "name": "test"}
        result = _dispatch_contract_model(raw)
        assert result.node_type == EnumNodeType.EFFECT_GENERIC

    def test_dispatch_compute_generic(self):
        raw = {"node_type": "COMPUTE_GENERIC", "name": "test"}
        result = _dispatch_contract_model(raw)
        assert result.node_type == EnumNodeType.COMPUTE_GENERIC

    def test_dispatch_reducer_generic(self):
        raw = {"node_type": "REDUCER_GENERIC", "name": "test"}
        result = _dispatch_contract_model(raw)
        assert result.node_type == EnumNodeType.REDUCER_GENERIC

    def test_dispatch_orchestrator_generic(self):
        raw = {"node_type": "ORCHESTRATOR_GENERIC", "name": "test"}
        result = _dispatch_contract_model(raw)
        assert result.node_type == EnumNodeType.ORCHESTRATOR_GENERIC

    def test_dispatch_lowercase_alias(self):
        raw = {"node_type": "compute", "name": "test"}
        result = _dispatch_contract_model(raw)
        assert result.node_type == EnumNodeType.COMPUTE_GENERIC

    def test_dispatch_unknown_node_type_raises(self):
        raw = {"node_type": "UNKNOWN_TYPE", "name": "test"}
        with pytest.raises(ValueError, match="Unknown node_type"):
            _dispatch_contract_model(raw)

    def test_dispatch_missing_node_type_raises(self):
        raw = {"name": "test"}
        with pytest.raises(ValueError, match="node_type"):
            _dispatch_contract_model(raw)

    def test_returns_typed_model_not_dict(self):
        raw = {"node_type": "EFFECT_GENERIC"}
        result = _dispatch_contract_model(raw)
        assert isinstance(result, ModelContractNodeType)
        assert not isinstance(result, dict)
