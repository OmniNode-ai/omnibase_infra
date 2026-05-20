# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for OverlayConfigResolver (OMN-11069)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip(
    "omnibase_core.models.overlay.model_overlay_file",
    reason="omnibase_core overlay models not yet released; requires Wave-1 worktree in PYTHONPATH",
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
from omnibase_infra.runtime.overlay.model_overlay_env_injection_result import (
    ModelOverlayEnvInjectionResult,
)
from omnibase_infra.runtime.overlay.overlay_config_resolver import (
    ModelOverlayResolutionResult,
    OverlayConfigResolver,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _overlay(**transports: dict[str, str]) -> ModelOverlayFile:
    return ModelOverlayFile.model_validate(
        {
            "overlay_version": "1.0.0",
            "environment": "dev",
            "scope": "env",
            "transports": transports,
        }
    )


def _reqs(
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOverlayConfigResolver:
    def test_resolves_matching_keys(self) -> None:
        overlay = _overlay(database={"POSTGRES_HOST": "db.local"})
        reqs = _reqs([("POSTGRES_HOST", EnumInfraTransportType.DATABASE, True)])
        result = OverlayConfigResolver().resolve(overlay, reqs)
        assert result.resolved["POSTGRES_HOST"] == "db.local"
        assert len(result.missing) == 0

    def test_missing_required_raises_with_key_name(self) -> None:
        overlay = _overlay()
        reqs = _reqs([("POSTGRES_HOST", EnumInfraTransportType.DATABASE, True)])
        with pytest.raises(RequiredConfigMissingError, match="POSTGRES_HOST"):
            OverlayConfigResolver().resolve(overlay, reqs)

    def test_missing_required_lists_all_missing(self) -> None:
        overlay = _overlay()
        reqs = _reqs(
            [
                ("POSTGRES_HOST", EnumInfraTransportType.DATABASE, True),
                ("POSTGRES_PORT", EnumInfraTransportType.DATABASE, True),
            ]
        )
        with pytest.raises(RequiredConfigMissingError) as exc_info:
            OverlayConfigResolver().resolve(overlay, reqs)
        msg = str(exc_info.value)
        assert "POSTGRES_HOST" in msg
        assert "POSTGRES_PORT" in msg

    def test_missing_optional_does_not_raise(self) -> None:
        overlay = _overlay()
        reqs = _reqs([("OPT_KEY", EnumInfraTransportType.HTTP, False)])
        result = OverlayConfigResolver().resolve(overlay, reqs)
        assert "OPT_KEY" in result.missing
        assert "OPT_KEY" not in result.resolved

    def test_resolved_transports_reflects_actual_resolution(self) -> None:
        overlay = _overlay(database={"POSTGRES_HOST": "h"})
        reqs = _reqs(
            [
                ("POSTGRES_HOST", EnumInfraTransportType.DATABASE, True),
                ("KAFKA_GROUP_ID", EnumInfraTransportType.KAFKA, False),
            ]
        )
        result = OverlayConfigResolver().resolve(overlay, reqs)
        assert "db" in result.manifest.resolved_transports
        assert "kafka" not in result.manifest.resolved_transports
        assert "kafka" in result.manifest.required_transports

    def test_manifest_contract_requirements_hash_starts_with_sha256(self) -> None:
        overlay = _overlay(database={"POSTGRES_HOST": "h"})
        reqs = _reqs([("POSTGRES_HOST", EnumInfraTransportType.DATABASE, True)])
        result = OverlayConfigResolver().resolve(overlay, reqs)
        assert result.manifest.contract_requirements_hash.startswith("sha256:")

    def test_manifest_config_source_is_overlay(self) -> None:
        overlay = _overlay(database={"POSTGRES_HOST": "h"})
        reqs = _reqs([("POSTGRES_HOST", EnumInfraTransportType.DATABASE, True)])
        result = OverlayConfigResolver().resolve(overlay, reqs)
        assert result.manifest.config_source == "overlay"

    def test_env_injection_injects_and_reports(self) -> None:
        overlay = _overlay(database={"OVERLAY_HOST": "h", "OVERLAY_PORT": "5432"})
        reqs = _reqs(
            [
                ("OVERLAY_HOST", EnumInfraTransportType.DATABASE, True),
                ("OVERLAY_PORT", EnumInfraTransportType.DATABASE, True),
            ]
        )
        result = OverlayConfigResolver().resolve(overlay, reqs)
        # Ensure a clean slate, then pre-set only HOST to simulate already-existing
        os.environ.pop("OVERLAY_HOST", None)
        os.environ.pop("OVERLAY_PORT", None)
        os.environ["OVERLAY_HOST"] = "already-set"
        try:
            inj: ModelOverlayEnvInjectionResult = result.apply_to_environment()
            assert "OVERLAY_PORT" in inj.injected_keys
            assert "OVERLAY_HOST" in inj.skipped_existing_keys
            assert os.environ["OVERLAY_HOST"] == "already-set"
            assert os.environ["OVERLAY_PORT"] == "5432"
        finally:
            os.environ.pop("OVERLAY_HOST", None)
            os.environ.pop("OVERLAY_PORT", None)

    def test_existing_env_vars_not_overwritten(self) -> None:
        overlay = _overlay(database={"MY_VAR": "overlay-value"})
        reqs = _reqs([("MY_VAR", EnumInfraTransportType.DATABASE, True)])
        result = OverlayConfigResolver().resolve(overlay, reqs)
        os.environ["MY_VAR"] = "original"
        try:
            inj = result.apply_to_environment()
            assert os.environ["MY_VAR"] == "original"
            assert "MY_VAR" in inj.skipped_existing_keys
            assert "MY_VAR" not in inj.injected_keys
        finally:
            os.environ.pop("MY_VAR", None)

    def test_result_is_frozen(self) -> None:
        from pydantic import ValidationError

        overlay = _overlay(database={"POSTGRES_HOST": "h"})
        reqs = _reqs([("POSTGRES_HOST", EnumInfraTransportType.DATABASE, True)])
        result = OverlayConfigResolver().resolve(overlay, reqs)
        with pytest.raises((ValidationError, AttributeError)):
            result.resolved = {}  # type: ignore[misc]

    def test_injection_result_is_frozen(self) -> None:
        from pydantic import ValidationError

        overlay = _overlay(database={"OVERLAY_FREEZE_KEY": "h"})
        reqs = _reqs([("OVERLAY_FREEZE_KEY", EnumInfraTransportType.DATABASE, True)])
        result = OverlayConfigResolver().resolve(overlay, reqs)
        os.environ.pop("OVERLAY_FREEZE_KEY", None)
        try:
            inj = result.apply_to_environment()
            with pytest.raises((ValidationError, AttributeError, TypeError)):
                inj.injected_keys = ()  # type: ignore[misc]
        finally:
            os.environ.pop("OVERLAY_FREEZE_KEY", None)

    def test_model_overlay_resolution_result_importable_from_resolver_module(
        self,
    ) -> None:
        assert ModelOverlayResolutionResult is not None
