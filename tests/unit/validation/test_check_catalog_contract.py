# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""The check registry is built from the declared catalog (OMN-21623)."""

from __future__ import annotations

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.validation.model_check_catalog_entry import (
    ModelCheckCatalogEntry,
)
from omnibase_infra.validation.checks import registry_infra_check as registry

pytestmark = pytest.mark.unit


def test_registry_holds_one_executor_per_declared_entry() -> None:
    declared = tuple(
        entry.check_code for entry in ModelCheckCatalogEntry.load_catalog()
    )
    assert declared == registry.CHECK_CATALOG_ORDER
    assert set(registry.CHECK_REGISTRY) == set(declared)


def test_subprocess_entry_reaches_the_executor() -> None:
    entry = next(e for e in ModelCheckCatalogEntry.load_catalog() if e.command)
    executor = registry.CHECK_REGISTRY[entry.check_code]
    assert executor.label == entry.label
    assert executor.severity == entry.severity


def test_handler_outside_the_checks_namespace_is_refused() -> None:
    entry = ModelCheckCatalogEntry(
        check_code="CHECK-X-001", handler="os.path.join", command="true"
    )
    with pytest.raises(ProtocolConfigurationError):
        entry.build_executor()


def test_handler_that_is_not_an_executor_is_refused() -> None:
    entry = ModelCheckCatalogEntry(
        check_code="CHECK-X-002",
        handler="omnibase_infra.validation.checks.registry_infra_check.MappingProxyType",
    )
    with pytest.raises(ProtocolConfigurationError):
        entry.build_executor()


def test_entry_whose_code_disagrees_with_its_executor_is_refused() -> None:
    entry = ModelCheckCatalogEntry(
        check_code="CHECK-RISK-009",
        handler="omnibase_infra.validation.checks.handler_risk.HandlerRiskDiffSize",
    )
    with pytest.raises(ProtocolConfigurationError):
        entry.build_executor()
