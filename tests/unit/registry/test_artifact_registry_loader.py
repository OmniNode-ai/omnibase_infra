# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from uuid import UUID

import pytest

from omnibase_infra.registry.loader import load_artifact_registry
from omnibase_infra.registry.models.model_artifact_registry_entry import (
    ModelArtifactRegistryEntry,
)
from omnibase_infra.registry.models.model_source_trigger import ModelSourceTrigger

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.unit
class TestArtifactRegistryModels:
    def test_source_trigger_valid(self):
        trigger = ModelSourceTrigger(
            pattern="src/nodes/*/contract.yaml", change_scope="structural"
        )
        assert trigger.pattern == "src/nodes/*/contract.yaml"
        assert trigger.change_scope == "structural"

    def test_source_trigger_rejects_invalid_scope(self):
        with pytest.raises(ValueError):
            ModelSourceTrigger(pattern="*.yaml", change_scope="invalid")

    def test_match_fields_reserved_and_documented(self):
        trigger = ModelSourceTrigger(
            pattern="src/nodes/*/contract.yaml",
            match_fields=["event_bus.publish_topics"],
        )
        assert trigger.match_fields == ["event_bus.publish_topics"]

    def test_registry_entry_minimal(self):
        entry = ModelArtifactRegistryEntry(
            artifact_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            artifact_type="doc",
            title="Test Doc",
            path="docs/test.md",
            repo="omnibase_infra",
        )
        assert entry.update_policy == "warn"
        assert entry.owner_hint is None
        assert entry.last_verified is None
        assert isinstance(entry.artifact_id, UUID)

    def test_registry_entry_rejects_extra_fields(self):
        with pytest.raises(ValueError):
            ModelArtifactRegistryEntry(
                artifact_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                artifact_type="doc",
                title="Test",
                path="docs/test.md",
                repo="test",
                bogus_field="nope",
            )


@pytest.mark.unit
class TestArtifactRegistryLoader:
    def test_load_valid_registry(self):
        registry = load_artifact_registry(FIXTURE_DIR / "valid_registry.yaml")
        assert registry.version == "1.0.0"
        assert len(registry.artifacts) == 2

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_artifact_registry(Path("/nonexistent.yaml"))

    def test_load_invalid_yaml_raises_domain_error(self):
        with pytest.raises(ValueError, match="Invalid artifact registry"):
            load_artifact_registry(FIXTURE_DIR / "invalid_registry.yaml")

    def test_load_malformed_yaml_raises_domain_error(self):
        with pytest.raises(ValueError, match="Failed to parse"):
            load_artifact_registry(FIXTURE_DIR / "malformed_registry.yaml")
