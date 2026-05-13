# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for pricing_manifest_version on ModelTaskDelegatedEvent.

OMN-10949: Projection rows must include pricing_manifest_version so
savings can be audited and recomputed against any manifest revision.
Events without the field default to version 0.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_delegation_orchestrator.models.model_task_delegated_event import (
    ModelTaskDelegatedEvent,
)

pytestmark = [pytest.mark.unit]


def _base_event(**overrides: object) -> ModelTaskDelegatedEvent:
    defaults: dict[str, object] = {
        "timestamp": "2026-05-13T10:00:00Z",
        "correlation_id": uuid4(),
        "task_type": "code_generation",
        "delegated_to": "local-qwen3",
        "quality_gate_passed": True,
    }
    defaults.update(overrides)
    return ModelTaskDelegatedEvent(**defaults)  # type: ignore[arg-type]


class TestPricingManifestVersion:
    def test_default_is_zero(self) -> None:
        event = _base_event()
        assert event.pricing_manifest_version == 0

    def test_explicit_version_accepted(self) -> None:
        event = _base_event(pricing_manifest_version=3)
        assert event.pricing_manifest_version == 3

    def test_serialization_roundtrip(self) -> None:
        event = _base_event(pricing_manifest_version=7)
        data = event.model_dump(mode="json")
        assert data["pricing_manifest_version"] == 7
        restored = ModelTaskDelegatedEvent.model_validate(data)
        assert restored.pricing_manifest_version == 7

    def test_extra_field_still_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _base_event(unknown_field="bad")
