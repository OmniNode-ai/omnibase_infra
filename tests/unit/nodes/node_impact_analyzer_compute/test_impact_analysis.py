# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerImpactAnalysis scoring logic.

Tests the deterministic scoring table from OMN-3925:
- fnmatch pattern matching against changed files
- scope multiplier (1.0 structural / 0.7 PR)
- policy floor enforcement
- required_action threshold assignment
- highest_merge_policy derivation
- manual_plan_request with empty files matches all repo triggers
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_artifact_change_detector_effect.models.model_update_trigger import (
    ModelUpdateTrigger,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.handlers.constants import (
    ACTION_THRESHOLD_REGENERATE,
    ACTION_THRESHOLD_REVIEW,
    MERGE_POLICY_ORDER,
    POLICY_FLOORS,
    REASON_CODES,
    SCOPE_MULTIPLIER_PR,
    SCOPE_MULTIPLIER_STRUCTURAL,
    STRUCTURAL_TRIGGER_TYPES,
)
from omnibase_infra.nodes.node_impact_analyzer_compute.handlers.handler_impact_analysis import (
    HandlerImpactAnalysis,
)
from omnibase_infra.registry.models.model_artifact_registry import ModelArtifactRegistry
from omnibase_infra.registry.models.model_artifact_registry_entry import (
    ModelArtifactRegistryEntry,
)
from omnibase_infra.registry.models.model_source_trigger import ModelSourceTrigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trigger(
    trigger_type: str = "contract_changed",
    changed_files: list[str] | None = None,
    source_repo: str = "omnibase_infra",
) -> ModelUpdateTrigger:
    return ModelUpdateTrigger(
        trigger_id=uuid4(),
        trigger_type=trigger_type,  # type: ignore[arg-type]
        source_repo=source_repo,
        changed_files=changed_files or [],
        timestamp=datetime.now(UTC),
    )


def _make_entry(
    update_policy: str = "warn",
    patterns: list[str] | None = None,
    repo: str = "omnibase_infra",
) -> ModelArtifactRegistryEntry:
    triggers = [
        ModelSourceTrigger(pattern=p, change_scope="structural")
        for p in (patterns or ["src/omnibase_infra/nodes/*/contract.yaml"])
    ]
    return ModelArtifactRegistryEntry(
        artifact_id=uuid4(),
        artifact_type="doc",
        title="Test Doc",
        path="docs/test.md",
        repo=repo,
        update_policy=update_policy,  # type: ignore[arg-type]
        source_triggers=triggers,
    )


def _make_registry(*entries: ModelArtifactRegistryEntry) -> ModelArtifactRegistry:
    return ModelArtifactRegistry(
        version="1.0.0",
        description="Test registry",
        artifacts=list(entries),
    )


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConstants:
    def test_merge_policy_order_is_total_order(self) -> None:
        assert MERGE_POLICY_ORDER == {"none": 0, "warn": 1, "require": 2, "strict": 3}

    def test_scope_multipliers(self) -> None:
        assert SCOPE_MULTIPLIER_STRUCTURAL == 1.0
        assert SCOPE_MULTIPLIER_PR == 0.7

    def test_policy_floors(self) -> None:
        assert POLICY_FLOORS["none"] == 0.0
        assert POLICY_FLOORS["warn"] == 0.0
        assert POLICY_FLOORS["require"] == 0.3
        assert POLICY_FLOORS["strict"] == 0.5

    def test_action_thresholds(self) -> None:
        assert ACTION_THRESHOLD_REGENERATE == 0.8
        assert ACTION_THRESHOLD_REVIEW == 0.5

    def test_structural_trigger_types(self) -> None:
        assert "contract_changed" in STRUCTURAL_TRIGGER_TYPES
        assert "schema_changed" in STRUCTURAL_TRIGGER_TYPES
        assert "pr_opened" not in STRUCTURAL_TRIGGER_TYPES

    def test_reason_codes_frozenset(self) -> None:
        assert "contract_yaml_changed" in REASON_CODES
        assert "manual_reconciliation" in REASON_CODES
        assert "full_repo_reconciliation" in REASON_CODES


# ---------------------------------------------------------------------------
# HandlerImpactAnalysis tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHandlerImpactAnalysis:
    def setup_method(self) -> None:
        self.handler = HandlerImpactAnalysis()

    def test_contract_change_matches_structural_trigger(self) -> None:
        """contract_changed trigger with matching pattern produces an impacted artifact."""
        entry = _make_entry(
            update_policy="warn",
            patterns=["src/omnibase_infra/nodes/*/contract.yaml"],
        )
        registry = _make_registry(entry)
        trigger = _make_trigger(
            trigger_type="contract_changed",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        result = self.handler.analyze(trigger, registry)

        assert len(result.impacted_artifacts) == 1
        artifact = result.impacted_artifacts[0]
        assert artifact.artifact_id == entry.artifact_id
        # 1 of 1 triggers matched, scope_multiplier=1.0 → base=1.0, impact=1.0
        assert artifact.impact_strength == 1.0
        assert artifact.required_action == "regenerate"

    def test_unrelated_file_no_impact(self) -> None:
        """Changed file that does not match any trigger produces no impact."""
        entry = _make_entry(
            update_policy="warn",
            patterns=["src/omnibase_infra/nodes/*/contract.yaml"],
        )
        registry = _make_registry(entry)
        trigger = _make_trigger(
            trigger_type="pr_opened",
            changed_files=["tests/unit/test_smoke.py"],
        )
        result = self.handler.analyze(trigger, registry)
        assert len(result.impacted_artifacts) == 0
        assert result.highest_merge_policy == "none"

    def test_multiple_triggers_increase_score(self) -> None:
        """Artifact with 2 triggers where 1 matches gets base=0.5."""
        entry = _make_entry(
            update_policy="warn",
            patterns=[
                "src/omnibase_infra/nodes/*/contract.yaml",
                "docs/architecture/*.md",
            ],
        )
        registry = _make_registry(entry)
        trigger = _make_trigger(
            trigger_type="contract_changed",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        result = self.handler.analyze(trigger, registry)

        assert len(result.impacted_artifacts) == 1
        artifact = result.impacted_artifacts[0]
        # 1 of 2 triggers matched, structural → scope=1.0 → base=0.5
        assert abs(artifact.impact_strength - 0.5) < 1e-9
        # >= 0.5 → review
        assert artifact.required_action == "review"

    def test_strict_policy_enforces_floor(self) -> None:
        """Strict policy floor (0.5) applies even when computed score is lower."""
        entry = _make_entry(
            update_policy="strict",
            patterns=[
                "src/omnibase_infra/nodes/*/contract.yaml",
                "docs/architecture/*.md",
                "src/omnibase_infra/scripts/*.sh",
                "src/omnibase_infra/config/*.yaml",
            ],
        )
        registry = _make_registry(entry)
        trigger = _make_trigger(
            trigger_type="pr_opened",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        result = self.handler.analyze(trigger, registry)

        assert len(result.impacted_artifacts) == 1
        artifact = result.impacted_artifacts[0]
        # 1 of 4 triggers matched, pr trigger → scope=0.7 → raw=0.175
        # policy_floor for strict = 0.5 → impact_strength = 0.5
        assert abs(artifact.impact_strength - 0.5) < 1e-9

    def test_action_assignment_regenerate(self) -> None:
        """impact_strength >= 0.8 maps to required_action='regenerate'."""
        entry = _make_entry(
            update_policy="require",
            patterns=["src/omnibase_infra/nodes/*/contract.yaml"],
        )
        registry = _make_registry(entry)
        trigger = _make_trigger(
            trigger_type="contract_changed",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        result = self.handler.analyze(trigger, registry)

        assert len(result.impacted_artifacts) == 1
        assert result.impacted_artifacts[0].required_action == "regenerate"

    def test_reason_codes_are_from_fixed_set(self) -> None:
        """All reason codes produced by the handler are in the REASON_CODES constant."""
        entry = _make_entry(
            update_policy="warn",
            patterns=["src/omnibase_infra/nodes/*/contract.yaml"],
        )
        registry = _make_registry(entry)
        trigger = _make_trigger(
            trigger_type="contract_changed",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        result = self.handler.analyze(trigger, registry)

        for artifact in result.impacted_artifacts:
            for code in artifact.reason_codes:
                assert code in REASON_CODES, f"Unexpected reason code: {code}"

    def test_highest_merge_policy_is_strictest(self) -> None:
        """highest_merge_policy is the max policy across all impacted artifacts."""
        entry_warn = _make_entry(
            update_policy="warn",
            patterns=["src/omnibase_infra/nodes/*/contract.yaml"],
        )
        entry_strict = _make_entry(
            update_policy="strict",
            patterns=["src/omnibase_infra/nodes/*/contract.yaml"],
        )
        registry = _make_registry(entry_warn, entry_strict)
        trigger = _make_trigger(
            trigger_type="contract_changed",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        result = self.handler.analyze(trigger, registry)

        assert result.highest_merge_policy == "strict"

    def test_manual_trigger_empty_files_matches_all_repo_triggers(self) -> None:
        """manual_plan_request with empty changed_files matches all artifacts in source_repo."""
        entry_infra = _make_entry(
            update_policy="warn",
            patterns=["src/omnibase_infra/nodes/*/contract.yaml"],
            repo="omnibase_infra",
        )
        entry_other = _make_entry(
            update_policy="strict",
            patterns=["src/other/contract.yaml"],
            repo="other_repo",
        )
        registry = _make_registry(entry_infra, entry_other)
        trigger = _make_trigger(
            trigger_type="manual_plan_request",
            changed_files=[],
            source_repo="omnibase_infra",
        )
        result = self.handler.analyze(trigger, registry)

        # Only entry_infra matches (same repo); entry_other is in a different repo
        assert len(result.impacted_artifacts) == 1
        artifact = result.impacted_artifacts[0]
        assert artifact.artifact_id == entry_infra.artifact_id
        assert "manual_reconciliation" in artifact.reason_codes

    def test_pr_trigger_applies_pr_scope_multiplier(self) -> None:
        """pr_opened trigger uses scope_multiplier=0.7, not 1.0."""
        entry = _make_entry(
            update_policy="warn",
            patterns=["src/omnibase_infra/nodes/*/contract.yaml"],
        )
        registry = _make_registry(entry)
        trigger_pr = _make_trigger(
            trigger_type="pr_opened",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        trigger_contract = _make_trigger(
            trigger_type="contract_changed",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )

        result_pr = self.handler.analyze(trigger_pr, registry)
        result_contract = self.handler.analyze(trigger_contract, registry)

        # PR: base=1.0, scope=0.7 → 0.7
        # contract_changed: base=1.0, scope=1.0 → 1.0
        pr_strength = result_pr.impacted_artifacts[0].impact_strength
        contract_strength = result_contract.impacted_artifacts[0].impact_strength
        assert abs(pr_strength - 0.7) < 1e-9
        assert abs(contract_strength - 1.0) < 1e-9

    def test_none_policy_artifact_with_low_strength_no_impact(self) -> None:
        """Artifact with update_policy='none' and low impact returns required_action='none' → filtered out."""
        # PR trigger → scope=0.7, one of many triggers matched → low base
        entry = _make_entry(
            update_policy="none",
            patterns=[
                "src/omnibase_infra/nodes/*/contract.yaml",
                "docs/architecture/*.md",
                "src/scripts/*.sh",
                "src/config/*.yaml",
            ],
        )
        registry = _make_registry(entry)
        # Only 1 of 4 triggers matches, PR trigger → 0.25 * 0.7 = 0.175
        # policy floor for 'none' = 0.0, impact=0.175
        # action: 0.175 < 0.5, policy='none' → 'none' → filtered out
        trigger = _make_trigger(
            trigger_type="pr_opened",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        result = self.handler.analyze(trigger, registry)
        assert len(result.impacted_artifacts) == 0

    def test_handler_classification(self) -> None:
        """Handler exposes correct type and category."""
        from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

        assert self.handler.handler_type == EnumHandlerType.INFRA_HANDLER
        assert self.handler.handler_category == EnumHandlerTypeCategory.COMPUTE
        assert self.handler.handler_id == "handler-impact-analysis"

    def test_empty_registry_produces_no_impact(self) -> None:
        """Empty registry always returns empty impacted_artifacts."""
        registry = _make_registry()
        trigger = _make_trigger(
            trigger_type="contract_changed",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        result = self.handler.analyze(trigger, registry)
        assert len(result.impacted_artifacts) == 0
        assert result.highest_merge_policy == "none"

    def test_impact_strength_capped_at_1(self) -> None:
        """impact_strength is always <= 1.0 even with policy floors."""
        entry = _make_entry(
            update_policy="strict",
            patterns=["src/omnibase_infra/nodes/*/contract.yaml"],
        )
        registry = _make_registry(entry)
        trigger = _make_trigger(
            trigger_type="contract_changed",
            changed_files=["src/omnibase_infra/nodes/node_foo/contract.yaml"],
        )
        result = self.handler.analyze(trigger, registry)
        for artifact in result.impacted_artifacts:
            assert artifact.impact_strength <= 1.0
