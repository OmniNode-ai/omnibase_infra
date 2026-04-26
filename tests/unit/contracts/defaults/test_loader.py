# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the infra concrete loader for SPI default handler contract templates (OMN-9755)."""

from __future__ import annotations

import pytest

from omnibase_core.models.contracts.model_handler_contract import ModelHandlerContract
from omnibase_infra.contracts.defaults import load_default_handler_contract
from omnibase_spi.exceptions import TemplateNotFoundError


@pytest.mark.unit
class TestDefaultHandlerContractLoader:
    def test_default_effect_contract_is_typed(self) -> None:
        contract = load_default_handler_contract("default_effect_handler.yaml")
        assert isinstance(contract, ModelHandlerContract)

    def test_default_compute_contract_is_typed(self) -> None:
        contract = load_default_handler_contract("default_compute_handler.yaml")
        assert isinstance(contract, ModelHandlerContract)

    def test_default_nondeterministic_compute_contract_is_typed(self) -> None:
        contract = load_default_handler_contract(
            "default_nondeterministic_compute_handler.yaml"
        )
        assert isinstance(contract, ModelHandlerContract)

    def test_missing_template_raises_template_not_found_error(self) -> None:
        with pytest.raises(TemplateNotFoundError):
            load_default_handler_contract("nonexistent_template.yaml")

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(TemplateNotFoundError):
            load_default_handler_contract("../../etc/passwd")

    def test_absolute_path_rejected(self) -> None:
        with pytest.raises(TemplateNotFoundError):
            load_default_handler_contract("/etc/passwd")

    def test_effect_contract_has_expected_handler_id(self) -> None:
        contract = load_default_handler_contract("default_effect_handler.yaml")
        assert contract.handler_id == "template.effect.default"

    def test_compute_contract_has_expected_handler_id(self) -> None:
        contract = load_default_handler_contract("default_compute_handler.yaml")
        assert contract.handler_id == "template.compute.default"
