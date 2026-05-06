# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for bifrost_delegation.yaml config loading and schema validation.

Covers:
- Schema validation of the canonical bifrost_delegation.yaml
- Required provenance fields: rule_id, config_version, shadow_policy_id
- Loader raises on missing file or invalid YAML
- All three task classes (code_generation, documentation, research) are present
- All backends have the required env-var or model_name fields

Related:
    - OMN-10637: Bifrost routing rules for delegation task classes
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from uuid import UUID

import pytest
import yaml

from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config_loader_bifrost_delegation import (
    load_bifrost_delegation_config,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_delegation_config import (
    ModelBifrostDelegationConfig,
    ModelDelegationRoutingRule,
)

# Path to the canonical config relative to this test file.
_CONFIGS_DIR = (
    Path(__file__).parent.parent.parent.parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "configs"
)
_CANONICAL_CONFIG = _CONFIGS_DIR / "bifrost_delegation.yaml"


@pytest.mark.unit
class TestBifrostDelegationConfigLoad:
    """Tests for load_bifrost_delegation_config loading the canonical YAML."""

    def test_canonical_config_exists(self) -> None:
        """bifrost_delegation.yaml must exist at the canonical path."""
        assert _CANONICAL_CONFIG.exists(), (
            f"bifrost_delegation.yaml not found at {_CANONICAL_CONFIG}"
        )

    def test_canonical_config_loads_and_validates(self) -> None:
        """Canonical config loads without validation errors."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        assert isinstance(config, ModelBifrostDelegationConfig)

    def test_config_version_present(self) -> None:
        """config_version must be set — it is recorded in every gateway response."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        assert config.config_version, "config_version must not be empty"

    def test_schema_version_present(self) -> None:
        """schema_version must be set for format identification."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        assert config.schema_version, "schema_version must not be empty"

    def test_at_least_one_backend(self) -> None:
        """Config must define at least one backend."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        assert len(config.backends) >= 1

    def test_at_least_one_routing_rule(self) -> None:
        """Config must define at least one routing rule."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        assert len(config.routing_rules) >= 1

    def test_default_backends_populated(self) -> None:
        """default_backends must name at least one backend for fallback."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        assert len(config.default_backends) >= 1

    def test_default_backends_reference_known_ids(self) -> None:
        """Every default_backend must reference a declared backend_id."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        backend_ids = {b.backend_id for b in config.backends}
        for default_id in config.default_backends:
            assert default_id in backend_ids, (
                f"default_backend '{default_id}' not in backends"
            )


@pytest.mark.unit
class TestBifrostDelegationRoutingRuleProvenance:
    """Tests that every rule carries the required provenance fields (OMN-10637 R1)."""

    def test_all_rules_have_rule_id(self) -> None:
        """Every routing rule must have a non-empty rule_id."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        for rule in config.routing_rules:
            assert rule.rule_id, (
                f"rule for task_class={rule.task_class} missing rule_id"
            )

    def test_all_rules_have_task_class_contract_version(self) -> None:
        """Every routing rule must declare the task class contract version it targets."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        for rule in config.routing_rules:
            assert rule.task_class_contract_version, (
                f"rule {rule.rule_id} missing task_class_contract_version"
            )

    def test_all_rules_have_backend_policy_version(self) -> None:
        """Every routing rule must declare a backend_policy_version."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        for rule in config.routing_rules:
            assert rule.backend_policy_version, (
                f"rule {rule.rule_id} missing backend_policy_version"
            )

    def test_all_rules_have_shadow_policy_id(self) -> None:
        """Every routing rule must declare a shadow_policy_id for A/B evaluation."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        for rule in config.routing_rules:
            assert rule.shadow_policy_id, (
                f"rule {rule.rule_id} missing shadow_policy_id"
            )

    def test_all_rules_have_fallback_policy(self) -> None:
        """Every routing rule must declare a fallback_policy."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        for rule in config.routing_rules:
            assert rule.fallback_policy is not None, (
                f"rule {rule.rule_id} missing fallback_policy"
            )

    def test_rule_ids_are_unique(self) -> None:
        """rule_id must be unique across all routing rules."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        ids = [rule.rule_id for rule in config.routing_rules]
        assert len(ids) == len(set(ids)), "Duplicate rule_ids found in routing_rules"

    def test_all_rules_reference_declared_backends(self) -> None:
        """Every backend_id in routing rules must reference a declared backend."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        backend_ids = {b.backend_id for b in config.backends}
        for rule in config.routing_rules:
            for bid in rule.backend_ids:
                assert bid in backend_ids, (
                    f"rule {rule.rule_id} references unknown backend '{bid}'"
                )


@pytest.mark.unit
class TestBifrostDelegationTaskClasses:
    """Tests that the three required delegation task classes are covered."""

    def _rules_for_task(
        self, config: ModelBifrostDelegationConfig, task_class: str
    ) -> list[ModelDelegationRoutingRule]:
        return [r for r in config.routing_rules if r.task_class == task_class]

    def test_code_generation_rule_exists(self) -> None:
        """A routing rule for code_generation must exist."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        rules = self._rules_for_task(config, "code_generation")
        assert rules, "No routing rule found for task_class=code_generation"

    def test_documentation_rule_exists(self) -> None:
        """A routing rule for documentation must exist."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        rules = self._rules_for_task(config, "documentation")
        assert rules, "No routing rule found for task_class=documentation"

    def test_research_rule_exists(self) -> None:
        """A routing rule for research must exist."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        rules = self._rules_for_task(config, "research")
        assert rules, "No routing rule found for task_class=research"

    def test_code_generation_includes_local_backend(self) -> None:
        """code_generation rule must try a local backend first (cost optimization)."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        rules = self._rules_for_task(config, "code_generation")
        assert rules
        rule = rules[0]
        local_backends = {b.backend_id for b in config.backends if b.tier == "local"}
        assert any(bid in local_backends for bid in rule.backend_ids), (
            "code_generation rule does not route to any local backend"
        )

    def test_documentation_has_cost_ceiling_or_haiku_backend(self) -> None:
        """documentation rule must enforce a cost ceiling or use haiku (cheap cloud)."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        rules = self._rules_for_task(config, "documentation")
        assert rules
        rule = rules[0]
        has_cost_ceiling = rule.cost_ceiling_usd_per_1k_tokens is not None
        has_haiku = any("haiku" in bid for bid in rule.backend_ids)
        assert has_cost_ceiling or has_haiku, (
            "documentation rule must have a cost ceiling or use haiku backend"
        )

    def test_research_includes_reasoning_backend(self) -> None:
        """research rule must route to a reasoning-capable backend."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        rules = self._rules_for_task(config, "research")
        assert rules
        rule = rules[0]
        reasoning_backends = {
            b.backend_id
            for b in config.backends
            if "deep_reasoning" in b.capabilities or "reasoning" in b.capabilities
        }
        assert any(bid in reasoning_backends for bid in rule.backend_ids), (
            "research rule does not route to a reasoning-capable backend"
        )


@pytest.mark.unit
class TestBifrostDelegationConfigLoader:
    """Tests for config loader error handling."""

    def test_raises_file_not_found_for_missing_config(self, tmp_path: Path) -> None:
        """Loader raises FileNotFoundError for a non-existent path."""
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_bifrost_delegation_config(missing)

    def test_raises_value_error_for_non_mapping_yaml(self, tmp_path: Path) -> None:
        """Loader raises ValueError when YAML root is not a mapping."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected YAML mapping"):
            load_bifrost_delegation_config(bad_yaml)

    def test_raises_on_schema_validation_failure(self, tmp_path: Path) -> None:
        """Loader raises when YAML fails Pydantic schema validation."""
        invalid = tmp_path / "invalid.yaml"
        invalid.write_text(
            textwrap.dedent("""\
                config_version: "1.0.0"
                schema_version: "bifrost_delegation.v1"
                backends: []
                routing_rules: []
            """),
            encoding="utf-8",
        )
        with pytest.raises(Exception):
            load_bifrost_delegation_config(invalid)

    def test_loads_minimal_valid_config(self, tmp_path: Path) -> None:
        """Loader succeeds with a minimal but valid config."""
        minimal = {
            "config_version": "0.1.0",
            "schema_version": "bifrost_delegation.v1",
            "backends": [
                {
                    "backend_id": "test-backend",
                    "model_name": "test-model",
                    "tier": "local",
                }
            ],
            "routing_rules": [
                {
                    "rule_id": "aaaaaaaa-0001-4000-8000-000000000001",
                    "task_class": "code_generation",
                    "task_class_contract_version": "1.0.0",
                    "backend_policy_version": "1.0.0",
                    "backend_ids": ["test-backend"],
                    "fallback_policy": {
                        "action": "return_error",
                    },
                    "shadow_policy_id": "bbbbbbbb-0001-4000-8000-000000000001",
                }
            ],
        }
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text(yaml.dump(minimal), encoding="utf-8")
        config = load_bifrost_delegation_config(config_file)
        assert config.config_version == "0.1.0"
        assert len(config.routing_rules) == 1
        assert config.routing_rules[0].rule_id == UUID(
            "aaaaaaaa-0001-4000-8000-000000000001"
        )


@pytest.mark.unit
class TestBifrostResponseProvenanceFields:
    """Tests that the config fields needed for response provenance are accessible.

    The gateway must include rule_id and config_version in every response.
    These tests verify those fields are present and non-empty on a loaded config,
    as a proxy for the gateway's ability to populate response metadata.
    """

    def test_config_version_accessible_for_response_metadata(self) -> None:
        """config_version is accessible and non-empty for injection into responses."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        assert config.config_version
        # Ensure it's a string that can be embedded in a response
        assert isinstance(config.config_version, str)

    def test_rule_id_accessible_on_matched_rule(self) -> None:
        """rule_id on a matched rule is accessible and a valid UUID."""
        config = load_bifrost_delegation_config(_CANONICAL_CONFIG)
        # Simulate matching the first code_generation rule
        matched = next(
            (r for r in config.routing_rules if r.task_class == "code_generation"),
            None,
        )
        assert matched is not None
        assert matched.rule_id
        assert isinstance(matched.rule_id, UUID)
