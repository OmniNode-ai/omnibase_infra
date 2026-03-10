# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Quality tests for the seeded artifact_registry.yaml.

These tests verify structural integrity and data quality of the production
artifact registry — not the loader mechanics (those live in
test_artifact_registry_loader.py).
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path
from uuid import UUID

import pytest

from omnibase_infra.registry.loader import load_artifact_registry
from omnibase_infra.registry.models.model_artifact_registry import ModelArtifactRegistry
from omnibase_infra.registry.models.model_artifact_registry_entry import (
    ModelArtifactRegistryEntry,
)

REGISTRY_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "registry"
    / "artifact_registry.yaml"
)

# Glob patterns that represent contract.yaml files — at least one strict-policy
# artifact must declare a trigger matching these.
CONTRACT_YAML_PATTERNS = [
    "src/omnibase_infra/nodes/*/contract.yaml",
    "src/omnibase_infra/nodes/node_artifact_change_detector_effect/contract.yaml",
    "src/omnibase_infra/nodes/node_registration_reducer/contract.yaml",
]

# Files that should NEVER match any strict-policy artifact trigger.
UNRELATED_FILES = [
    "README.md",
    "docs/README.md",
    "tests/unit/registry/test_registry_quality.py",
    ".github/CODEOWNERS",
    "pyproject.toml",
    "uv.lock",
]

# Real contract.yaml paths from the codebase — used to verify trigger patterns
# actually match the intended files.
REAL_CONTRACT_PATHS = [
    "src/omnibase_infra/nodes/node_session_lifecycle_reducer/contract.yaml",
    "src/omnibase_infra/nodes/node_validation_orchestrator/contract.yaml",
    "src/omnibase_infra/nodes/node_github_pr_poller_effect/contract.yaml",
    "src/omnibase_infra/nodes/node_llm_embedding_effect/contract.yaml",
    "src/omnibase_infra/nodes/node_registration_reducer/contract.yaml",
]


@pytest.fixture(scope="module")
def registry() -> ModelArtifactRegistry:
    """Load the production artifact registry once per module."""
    return load_artifact_registry(REGISTRY_PATH)


@pytest.fixture(scope="module")
def artifacts(registry: ModelArtifactRegistry) -> list[ModelArtifactRegistryEntry]:
    return list(registry.artifacts)


@pytest.mark.unit
class TestRegistryExists:
    def test_registry_file_exists(self) -> None:
        assert REGISTRY_PATH.exists(), f"Registry file not found at {REGISTRY_PATH}"

    def test_registry_loads_without_error(self) -> None:
        registry = load_artifact_registry(REGISTRY_PATH)
        assert registry is not None

    def test_registry_has_minimum_artifact_count(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        assert len(artifacts) >= 10, (
            f"Registry must have at least 10 artifacts; found {len(artifacts)}"
        )

    def test_registry_has_at_most_expected_artifact_count(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        assert len(artifacts) <= 20, (
            f"Registry unexpectedly large: {len(artifacts)} artifacts "
            "(keep seed focused; add more in follow-up tickets)"
        )


@pytest.mark.unit
class TestArtifactIDUniqueness:
    def test_all_artifact_ids_unique(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        ids = [str(a.artifact_id) for a in artifacts]
        duplicates = {aid for aid in ids if ids.count(aid) > 1}
        assert not duplicates, f"Duplicate artifact IDs detected: {duplicates}"

    def test_all_artifact_ids_are_valid_uuids(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        for a in artifacts:
            assert isinstance(a.artifact_id, UUID), (
                f"artifact_id for '{a.title}' is not a UUID: {a.artifact_id!r}"
            )


@pytest.mark.unit
class TestArtifactPathConventions:
    """Paths must be relative (no leading slash) and end with a file extension."""

    def test_all_paths_are_relative(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        for a in artifacts:
            assert not a.path.startswith("/"), (
                f"Artifact '{a.title}' has absolute path: {a.path!r}"
            )

    def test_all_paths_have_extension(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        for a in artifacts:
            assert "." in Path(a.path).name, (
                f"Artifact '{a.title}' path has no file extension: {a.path!r}"
            )

    def test_all_titles_non_empty(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        for a in artifacts:
            assert a.title.strip(), f"Artifact {a.artifact_id} has empty title"

    def test_all_repos_non_empty(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        for a in artifacts:
            assert a.repo.strip(), f"Artifact '{a.title}' has empty repo"

    def test_known_repos_only(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        allowed_repos = {
            "omnibase_infra",
            "omnibase_core",
            "omnibase_spi",
            "omnidash",
            "omniintelligence",
            "omnimemory",
            "omniclaude",
        }
        for a in artifacts:
            assert a.repo in allowed_repos, (
                f"Artifact '{a.title}' references unknown repo: {a.repo!r}"
            )


@pytest.mark.unit
class TestSourceTriggers:
    def test_every_artifact_has_at_least_one_source_trigger(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        for a in artifacts:
            assert a.source_triggers, (
                f"Artifact '{a.title}' ({a.artifact_id}) has no source_triggers"
            )

    def test_trigger_patterns_are_non_empty_strings(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        for a in artifacts:
            for trigger in a.source_triggers:
                assert trigger.pattern.strip(), (
                    f"Artifact '{a.title}' has a trigger with empty pattern"
                )

    def test_trigger_change_scope_valid(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        valid_scopes = {"structural", "semantic", "any"}
        for a in artifacts:
            for trigger in a.source_triggers:
                assert trigger.change_scope in valid_scopes, (
                    f"Artifact '{a.title}' trigger has invalid change_scope: "
                    f"{trigger.change_scope!r}"
                )

    def test_contract_yaml_triggers_match_real_paths(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        """Every real contract.yaml path in the codebase should be matched by
        at least one trigger across the registry."""
        for real_path in REAL_CONTRACT_PATHS:
            matched_by_any = False
            for a in artifacts:
                for trigger in a.source_triggers:
                    if fnmatch(real_path, trigger.pattern):
                        matched_by_any = True
                        break
                if matched_by_any:
                    break
            assert matched_by_any, (
                f"Real file {real_path!r} is not matched by any trigger in the registry. "
                "Add a trigger pattern covering this file."
            )


@pytest.mark.unit
class TestUpdatePolicy:
    def test_all_update_policies_valid(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        valid_policies = {"none", "warn", "require", "strict"}
        for a in artifacts:
            assert a.update_policy in valid_policies, (
                f"Artifact '{a.title}' has invalid update_policy: {a.update_policy!r}"
            )

    def test_at_least_one_strict_or_require_policy(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        """Registry must have at least some meaningful enforcement."""
        blocking_policies = {a.update_policy for a in artifacts}
        assert blocking_policies & {"strict", "require"}, (
            "Registry has no 'strict' or 'require' policy artifacts — "
            "add at least one to ensure enforcement is tested."
        )

    def test_at_least_one_strict_policy_artifact(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        strict_artifacts = [a for a in artifacts if a.update_policy == "strict"]
        assert strict_artifacts, (
            "Registry has no 'strict' policy artifact. "
            "Topic catalog or ADRs should be strict."
        )


@pytest.mark.unit
class TestUnrelatedFilesDoNotMatchStrictArtifacts:
    """Strict-policy artifacts must not accidentally match unrelated files."""

    def test_unrelated_files_do_not_match_strict_artifacts(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        strict_artifacts = [a for a in artifacts if a.update_policy == "strict"]
        for unrelated in UNRELATED_FILES:
            for a in strict_artifacts:
                for trigger in a.source_triggers:
                    matched = fnmatch(unrelated, trigger.pattern)
                    assert not matched, (
                        f"Strict artifact '{a.title}' trigger pattern {trigger.pattern!r} "
                        f"incorrectly matches unrelated file {unrelated!r}"
                    )


@pytest.mark.unit
class TestRegistryContractYamlCoverage:
    """Artifacts that cover contract.yaml changes should declare structural scope."""

    def test_contract_yaml_triggers_use_structural_scope(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        contract_yaml_re = re.compile(r".*contract\.yaml$")
        for a in artifacts:
            for trigger in a.source_triggers:
                if contract_yaml_re.match(trigger.pattern):
                    assert trigger.change_scope in ("structural", "any"), (
                        f"Artifact '{a.title}' has a contract.yaml trigger with "
                        f"non-structural scope: {trigger.change_scope!r}. "
                        "contract.yaml triggers should use 'structural' or 'any'."
                    )


@pytest.mark.unit
class TestRegistryCoveredArtifactTypes:
    """The seeded registry must include diverse artifact types."""

    def test_includes_doc_type(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        assert any(a.artifact_type == "doc" for a in artifacts), (
            "Registry must include at least one 'doc' artifact"
        )

    def test_includes_reference_type(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        assert any(a.artifact_type == "reference" for a in artifacts), (
            "Registry must include at least one 'reference' artifact"
        )

    def test_includes_design_spec_type(
        self, artifacts: list[ModelArtifactRegistryEntry]
    ) -> None:
        assert any(a.artifact_type == "design_spec" for a in artifacts), (
            "Registry must include at least one 'design_spec' artifact (e.g. ADR)"
        )
