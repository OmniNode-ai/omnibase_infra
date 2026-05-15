# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for OverlayConfigResolver end-to-end flow (OMN-11069).

Validates that OverlayConfigResolver wires correctly with
ModelConfigRequirements and ModelOverlayFile without requiring external
services. Skips cleanly when the Wave-1 omnibase_core overlay models are
not yet installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip(
    "omnibase_core.models.overlay.model_overlay_file",
    reason="omnibase_core overlay models not yet released; Wave-1 worktree required",
)

from omnibase_core.models.overlay.model_overlay_file import (
    ModelOverlayFile,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.runtime.config_discovery.models.model_config_requirement import (
    ModelConfigRequirement,
)
from omnibase_infra.runtime.config_discovery.models.model_config_requirements import (
    ModelConfigRequirements,
)
from omnibase_infra.runtime.overlay.errors import (
    RequiredConfigMissingError,
)
from omnibase_infra.runtime.overlay.overlay_config_resolver import (
    OverlayConfigResolver,
)

pytestmark = [pytest.mark.integration]


def _make_overlay(**transports: dict[str, str]) -> ModelOverlayFile:
    return ModelOverlayFile.model_validate(
        {
            "overlay_version": "1.0.0",
            "environment": "integration-test",
            "scope": "env",
            "transports": transports,
        }
    )


def _make_requirements(
    keys: list[tuple[str, EnumInfraTransportType, bool]],
) -> ModelConfigRequirements:
    return ModelConfigRequirements(
        requirements=tuple(
            ModelConfigRequirement(
                key=k,
                transport_type=t,
                source_contract=Path("contract.yaml"),
                required=r,
            )
            for k, t, r in keys
        ),
        transport_types=tuple({t for _, t, _ in keys}),
        contract_paths=(Path("contract.yaml"),),
    )


class TestOverlayConfigResolverIntegration:
    def test_resolver_roundtrip_resolves_and_injects(self) -> None:
        """Full round-trip: resolve overlay → inject to env → verify injected."""
        overlay = _make_overlay(
            database={"INTEG_DB_HOST": "db.integration", "INTEG_DB_PORT": "5432"}
        )
        reqs = _make_requirements(
            [
                ("INTEG_DB_HOST", EnumInfraTransportType.DATABASE, True),
                ("INTEG_DB_PORT", EnumInfraTransportType.DATABASE, True),
            ]
        )
        resolver = OverlayConfigResolver()
        result = resolver.resolve(overlay, reqs)

        assert result.resolved["INTEG_DB_HOST"] == "db.integration"
        assert result.resolved["INTEG_DB_PORT"] == "5432"
        assert result.manifest.config_source == "overlay"
        assert result.manifest.contract_requirements_hash.startswith("sha256:")

        os.environ.pop("INTEG_DB_HOST", None)
        os.environ.pop("INTEG_DB_PORT", None)
        try:
            inj = result.apply_to_environment()
            assert "INTEG_DB_HOST" in inj.injected_keys
            assert "INTEG_DB_PORT" in inj.injected_keys
            assert os.environ["INTEG_DB_HOST"] == "db.integration"
            assert os.environ["INTEG_DB_PORT"] == "5432"
        finally:
            os.environ.pop("INTEG_DB_HOST", None)
            os.environ.pop("INTEG_DB_PORT", None)

    def test_resolver_raises_on_missing_required_keys(self) -> None:
        """RequiredConfigMissingError lists all missing required keys."""
        overlay = _make_overlay()
        reqs = _make_requirements(
            [
                ("INTEG_MISSING_A", EnumInfraTransportType.DATABASE, True),
                ("INTEG_MISSING_B", EnumInfraTransportType.KAFKA, True),
            ]
        )
        with pytest.raises(RequiredConfigMissingError) as exc_info:
            OverlayConfigResolver().resolve(overlay, reqs)
        msg = str(exc_info.value)
        assert "INTEG_MISSING_A" in msg
        assert "INTEG_MISSING_B" in msg

    def test_manifest_stable_identity_hash_is_deterministic(self) -> None:
        """Same inputs produce the same manifest hash."""
        overlay = _make_overlay(database={"INTEG_STABLE_KEY": "value"})
        reqs = _make_requirements(
            [("INTEG_STABLE_KEY", EnumInfraTransportType.DATABASE, True)]
        )
        r1 = OverlayConfigResolver().resolve(overlay, reqs)
        r2 = OverlayConfigResolver().resolve(overlay, reqs)
        assert r1.manifest.stable_identity_hash() == r2.manifest.stable_identity_hash()
