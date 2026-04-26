# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration test for the SPI default handler-contract loader (OMN-9755).

Verifies that the concrete loader in omnibase_infra reads real YAML data from
the installed omnibase_spi package via importlib.resources and returns a fully
validated ModelHandlerContract — no mocking, no stubs.

Integration Test Coverage gate: OMN-7005 (hard gate since 2026-04-13).
"""

from __future__ import annotations

import pytest

from omnibase_core.models.contracts.model_handler_contract import ModelHandlerContract
from omnibase_infra.contracts.defaults import load_default_handler_contract
from omnibase_spi.exceptions import TemplateNotFoundError


@pytest.mark.integration
class TestSpiDefaultContractLoaderIntegration:
    """End-to-end loader tests — exercises real importlib.resources package reads."""

    def test_effect_template_loads_and_validates(self) -> None:
        contract = load_default_handler_contract("default_effect_handler.yaml")
        assert isinstance(contract, ModelHandlerContract)
        assert contract.handler_id == "template.effect.default"

    def test_compute_template_loads_and_validates(self) -> None:
        contract = load_default_handler_contract("default_compute_handler.yaml")
        assert isinstance(contract, ModelHandlerContract)
        assert contract.handler_id == "template.compute.default"

    def test_nondeterministic_compute_template_loads_and_validates(self) -> None:
        contract = load_default_handler_contract(
            "default_nondeterministic_compute_handler.yaml"
        )
        assert isinstance(contract, ModelHandlerContract)

    def test_unknown_template_raises_template_not_found_error(self) -> None:
        with pytest.raises(TemplateNotFoundError):
            load_default_handler_contract("nonexistent.yaml")

    def test_path_traversal_raises_template_not_found_error(self) -> None:
        with pytest.raises(TemplateNotFoundError):
            load_default_handler_contract("../../etc/passwd")
