# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""The check registry is built from the declared catalog (OMN-21623)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.validation import model_check_catalog_entry as catalog
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


@pytest.mark.parametrize(
    "rows", [[], [{"check_code": "duplicate", "handler": "x.y"}] * 2]
)
def test_invalid_catalog_is_refused(
    rows: list[dict[str, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "check_catalog.yaml"
    path.write_text(yaml.safe_dump({"checks": rows}), encoding="utf-8")
    monkeypatch.setattr(catalog, "_CATALOG_PATH", path)
    ModelCheckCatalogEntry.load_catalog.cache_clear()
    try:
        with pytest.raises(ProtocolConfigurationError):
            ModelCheckCatalogEntry.load_catalog()
    finally:
        ModelCheckCatalogEntry.load_catalog.cache_clear()


def test_catalog_cache_is_reused_until_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "check_catalog.yaml"
    path.write_text(
        yaml.safe_dump({"checks": [{"check_code": "one", "handler": "x.y"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(catalog, "_CATALOG_PATH", path)
    ModelCheckCatalogEntry.load_catalog.cache_clear()
    try:
        entries = ModelCheckCatalogEntry.load_catalog()
        path.write_text(yaml.safe_dump({"checks": []}), encoding="utf-8")
        assert ModelCheckCatalogEntry.load_catalog() is entries
        ModelCheckCatalogEntry.load_catalog.cache_clear()
        with pytest.raises(ProtocolConfigurationError):
            ModelCheckCatalogEntry.load_catalog()
    finally:
        ModelCheckCatalogEntry.load_catalog.cache_clear()
