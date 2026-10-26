# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Workflow exemption boundaries across registry loading, classification and CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_infra.lab_proof.enum_lab_proof_kind import EnumLabProofKind
from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    classify_exemption,
    load_lab_proof_profile_registry,
    validate_steps_against_repo,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = REPO_ROOT / "config/lab_proof_profiles.yaml"
VALIDATOR = REPO_ROOT / "scripts/ci/validate_lab_proof_profiles.py"


def test_registry_rows_classify_workflows_without_exempting_documentation_repos() -> (
    None
):
    registry = load_lab_proof_profile_registry(REGISTRY)
    assert {profile.repo for profile in registry.profiles} == set(
        registry.registry_repos
    )
    assert len(registry.profiles) == len(registry.registry_repos)
    assert validate_steps_against_repo(registry, REPO_ROOT) == []
    exempt_repos = {
        profile.repo
        for profile in registry.profiles
        if all(
            variant.proof_kind is EnumLabProofKind.EXEMPT
            for variant in profile.variants
        )
    }
    assert len(exempt_repos) == 2
    for repo in exempt_repos:
        assert classify_exemption(registry, repo, ["README.md"], "human-author").exempt
    for repo in registry.registry_repos:
        decision = classify_exemption(
            registry, repo, [".github/workflows/ci.yml"], "human-author"
        )
        assert not decision.exempt, repo
        assert decision.exempt_class is None, repo
        assert decision.proof_kind is EnumLabProofKind.SCRIPT_REPLAY, repo


@pytest.mark.parametrize("invalid", ["workflow_exempt", "missing_step", "enforce"])
def test_validator_refuses_workflow_exempt_rows_and_existing_invalid_rows(
    tmp_path: Path, invalid: str
) -> None:
    accepted = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    if invalid == "workflow_exempt":
        profile = next(
            row
            for row in data["profiles"]
            if row["repo"] == "OmniNode-ai/knowledge_base"
        )
        profile["variants"][0]["match"] = {
            "base_branches": ["main"],
            "include_globs": [".github/**"],
        }
        diagnostic = ".github/"
    elif invalid == "missing_step":
        profile = next(
            row
            for row in data["profiles"]
            if row["repo"] == "OmniNode-ai/omnibase_core"
        )
        profile["variants"][0]["steps"][0] = (
            "node_lab_proof_missing_effect:lab_proof.run"
        )
        diagnostic = "node_lab_proof_missing_effect"
    else:
        data["profiles"][0]["enforce"] = True
        diagnostic = "no bar record"
    candidate = tmp_path / "registry.yaml"
    candidate.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    refused = subprocess.run(
        [sys.executable, str(VALIDATOR), "--registry", str(candidate)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert diagnostic in refused.stdout + refused.stderr


def test_classifier_serializes_exemptions_and_refuses_mixed_workflow_companions() -> (
    None
):
    registry = load_lab_proof_profile_registry(REGISTRY)
    companion = registry.change_control_companion
    bot = companion.authors[0]
    companion_paths = ["contracts/OMN-19565.yaml"]
    docs = classify_exemption(
        registry, "OmniNode-ai/knowledge_base", ["docs/guide.md"], "human-author"
    )
    bot_companion = classify_exemption(registry, companion.repo, companion_paths, bot)
    for decision in (docs, bot_companion):
        assert decision.exempt
        assert decision.model_dump(mode="json")["proof_kind"] == "exempt"
        assert decision.exempt_class is not None
    human_companion = classify_exemption(
        registry, companion.repo, companion_paths, "human-author"
    )
    assert not human_companion.exempt
    assert human_companion.proof_kind is None
    for repo, paths, author in (
        (
            "OmniNode-ai/omnibase_core",
            ["docs/guide.md", "src/module.py"],
            "human-author",
        ),
        (
            "OmniNode-ai/knowledge_base",
            ["docs/guide.md", ".github/PULL_REQUEST_TEMPLATE.md"],
            "human-author",
        ),
        (companion.repo, [*companion_paths, ".github/workflows/ci.yml"], bot),
    ):
        decision = classify_exemption(registry, repo, paths, author)
        assert not decision.exempt
        assert decision.exempt_class is None
        assert decision.proof_kind is None
