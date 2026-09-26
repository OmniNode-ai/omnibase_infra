# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lab proof registry: one row per repository, rows that can run, derived exemptions.

These bind the committed registry, the validator script and the exemption
classifier to the rulings behind them (RULING 2026-09-25T13:07:40Z, rolling
ledger): every code PR is proved through a profile declared as data;
documentation-only PRs and bot change-control companions are exempt, derived
from the diff and the author, never from a PR body.

Ticket: OMN-19565
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.lab_proof.enum_lab_proof_exempt_class import (
    EnumLabProofExemptClass,
)
from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    classify_exemption,
    glob_to_regex,
    load_lab_proof_profile_registry,
    validate_steps_against_repo,
)
from omnibase_infra.lab_proof.model_lab_proof_profile_registry import (
    ModelLabProofProfileRegistry,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
REGISTRY = REPO_ROOT / "config" / "lab_proof_profiles.yaml"
SCRIPT = REPO_ROOT / "scripts" / "ci" / "validate_lab_proof_profiles.py"
OCC = "OmniNode-ai/onex_change_control"
OCC_BOT = "onexbot-occ-writer[bot]"


def _raw() -> dict[str, Any]:
    loaded = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _run_script(registry: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--registry", str(registry)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "lab_proof_profiles.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _core_variant(data: dict[str, Any]) -> dict[str, Any]:
    for profile in data["profiles"]:
        if profile["repo"] == "OmniNode-ai/omnibase_core":
            variant = profile["variants"][0]
            assert isinstance(variant, dict)
            return variant
    raise AssertionError("no omnibase_core row")


# --- AC1: one row per registry repository -----------------------------------


def test_committed_registry_validates_and_its_steps_resolve() -> None:
    registry = load_lab_proof_profile_registry(REGISTRY)
    assert validate_steps_against_repo(registry, REPO_ROOT) == []
    repos = {profile.repo for profile in registry.profiles}
    assert repos == set(registry.registry_repos)
    assert len(registry.profiles) == len(registry.registry_repos)


def test_committed_registry_has_the_two_exempt_documentation_repos() -> None:
    registry = load_lab_proof_profile_registry(REGISTRY)
    exempt = sorted(
        profile.repo
        for profile in registry.profiles
        if all(variant.proof_kind == "exempt" for variant in profile.variants)
    )
    assert exempt == [
        "OmniNode-ai/knowledge-base",
        "OmniNode-ai/knowledge-base-internal",
    ]


def test_script_exits_zero_on_the_committed_registry() -> None:
    result = _run_script(REGISTRY)
    assert result.returncode == 0, result.stdout + result.stderr


def test_script_refuses_a_copy_with_the_omnidash_row_removed(tmp_path: Path) -> None:
    data = _raw()
    data["profiles"] = [
        profile
        for profile in data["profiles"]
        if profile["repo"] != "OmniNode-ai/omnidash"
    ]
    result = _run_script(_write(tmp_path, data))
    assert result.returncode != 0
    assert "OmniNode-ai/omnidash" in result.stdout + result.stderr


def test_a_row_for_an_unlisted_repository_is_refused() -> None:
    data = _raw()
    extra = copy.deepcopy(data["profiles"][-1])
    extra["repo"] = "OmniNode-ai/not_a_registry_repo"
    extra["profile_key"] = "not_a_registry_repo.exempt"
    data["profiles"].append(extra)
    with pytest.raises(ValidationError, match="not_a_registry_repo"):
        ModelLabProofProfileRegistry.model_validate(data)


# --- AC2: steps must name real nodes; enforce needs a bar record -------------


def test_a_step_naming_no_existing_node_is_refused(tmp_path: Path) -> None:
    data = _raw()
    _core_variant(data)["steps"][1] = (
        "node_lab_proof_does_not_exist_effect:lab_proof.run"
    )
    registry = ModelLabProofProfileRegistry.model_validate(data)
    errors = validate_steps_against_repo(registry, REPO_ROOT)
    assert len(errors) == 1
    assert "node_lab_proof_does_not_exist_effect" in errors[0]
    result = _run_script(_write(tmp_path, data))
    assert result.returncode != 0
    assert "node_lab_proof_does_not_exist_effect" in result.stdout + result.stderr


def test_a_step_naming_an_operation_the_node_does_not_route_is_refused() -> None:
    data = _raw()
    _core_variant(data)["steps"][0] = "node_lab_proof_plan_compute:lab_proof.bogus"
    registry = ModelLabProofProfileRegistry.model_validate(data)
    errors = validate_steps_against_repo(registry, REPO_ROOT)
    assert errors and "lab_proof.bogus" in errors[0]


def test_a_step_written_as_shell_text_is_refused() -> None:
    data = _raw()
    _core_variant(data)["steps"][0] = "docker compose up -d"
    with pytest.raises(ValidationError, match="node_name"):
        ModelLabProofProfileRegistry.model_validate(data)


def test_enforce_true_is_refused_without_a_bar_record() -> None:
    data = _raw()
    data["profiles"][0]["enforce"] = True
    with pytest.raises(ValidationError, match="no bar record"):
        ModelLabProofProfileRegistry.model_validate(data)


def test_a_runnable_row_without_mandatory_checks_is_refused() -> None:
    data = _raw()
    _core_variant(data)["mandatory_checks"] = []
    with pytest.raises(ValidationError, match="mandatory_checks"):
        ModelLabProofProfileRegistry.model_validate(data)


def test_an_unknown_check_name_is_refused() -> None:
    data = _raw()
    _core_variant(data)["mandatory_checks"].append("looks_fine_to_me")
    with pytest.raises(ValidationError):
        ModelLabProofProfileRegistry.model_validate(data)


def test_a_measured_budget_without_its_source_row_is_refused() -> None:
    data = _raw()
    variant = _core_variant(data)
    variant["budget_basis"] = "measured"
    variant.pop("budget_source", None)
    with pytest.raises(ValidationError, match="budget_source"):
        ModelLabProofProfileRegistry.model_validate(data)


def test_a_not_safe_row_that_runs_something_is_refused() -> None:
    data = _raw()
    variant = _core_variant(data)
    variant["status"] = "not_safe"
    with pytest.raises(ValidationError, match="execution must be none"):
        ModelLabProofProfileRegistry.model_validate(data)


def test_an_exemption_class_outside_the_ruled_two_is_refused() -> None:
    data = _raw()
    data["profiles"][0]["exempt_classes"] = ["docs_only", "workflow_only"]
    with pytest.raises(ValidationError):
        ModelLabProofProfileRegistry.model_validate(data)


def test_the_companion_exemption_is_refused_outside_the_change_control_repo() -> None:
    data = _raw()
    data["profiles"][0]["exempt_classes"] = [
        "docs_only",
        "bot_change_control_companion",
    ]
    with pytest.raises(ValidationError, match="only valid on"):
        ModelLabProofProfileRegistry.model_validate(data)


# --- AC3: the exemption classifier ---------------------------------------------


@pytest.fixture(scope="module")
def registry() -> ModelLabProofProfileRegistry:
    return load_lab_proof_profile_registry(REGISTRY)


def test_documentation_only_diff_is_exempt(
    registry: ModelLabProofProfileRegistry,
) -> None:
    decision = classify_exemption(
        registry,
        "OmniNode-ai/omnibase_core",
        ["README.md", "docs/guides/walker.md"],
        "jonahgabriel",
    )
    assert decision.exempt
    assert decision.exempt_class is EnumLabProofExemptClass.DOCS_ONLY


def test_documentation_diff_plus_one_source_file_is_not_exempt(
    registry: ModelLabProofProfileRegistry,
) -> None:
    decision = classify_exemption(
        registry,
        "OmniNode-ai/omnibase_core",
        ["README.md", "docs/guides/walker.md", "src/omnibase_core/topics.py"],
        "jonahgabriel",
    )
    assert not decision.exempt
    assert decision.exempt_class is None


def test_bot_change_control_companion_is_exempt(
    registry: ModelLabProofProfileRegistry,
) -> None:
    decision = classify_exemption(
        registry,
        OCC,
        [
            "contracts/OMN-19543.yaml",
            "drift/dod_receipts/OMN-19543/dod-pr-4118/command.yaml",
        ],
        OCC_BOT,
    )
    assert decision.exempt
    assert decision.exempt_class is EnumLabProofExemptClass.BOT_CHANGE_CONTROL_COMPANION


def test_human_authored_change_control_contract_is_not_exempt(
    registry: ModelLabProofProfileRegistry,
) -> None:
    decision = classify_exemption(
        registry,
        OCC,
        [
            "contracts/OMN-19543.yaml",
            "drift/dod_receipts/OMN-19543/dod-pr-4118/command.yaml",
        ],
        "jonahgabriel",
    )
    assert not decision.exempt


def test_bot_pr_that_also_touches_a_prod_grant_is_not_exempt(
    registry: ModelLabProofProfileRegistry,
) -> None:
    decision = classify_exemption(
        registry,
        OCC,
        ["contracts/OMN-16753.yaml", "grants/prod_promotion_grants.yaml"],
        OCC_BOT,
    )
    assert not decision.exempt


def test_empty_diff_is_never_exempt(registry: ModelLabProofProfileRegistry) -> None:
    decision = classify_exemption(registry, "OmniNode-ai/omnibase_core", [], "x")
    assert not decision.exempt


def test_unknown_repository_raises(registry: ModelLabProofProfileRegistry) -> None:
    with pytest.raises(KeyError):
        classify_exemption(registry, "OmniNode-ai/nope", ["README.md"], "x")


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("**/*.md", "README.md", True),
        ("**/*.md", "docs/a/b.md", True),
        ("docs/**", "docs/a/b.py", True),
        ("docs/**", "src/docs.py", False),
        ("contracts/OMN-*.yaml", "contracts/OMN-1.yaml", True),
        ("contracts/OMN-*.yaml", "contracts/sub/OMN-1.yaml", False),
        ("**", "anything/at/all.py", True),
    ],
)
def test_glob_semantics(pattern: str, path: str, expected: bool) -> None:
    assert bool(glob_to_regex(pattern).match(path)) is expected
