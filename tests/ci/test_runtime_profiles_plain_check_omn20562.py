# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime_profiles validator is a plain blocking check with no allowlist (OMN-20562).

OMN-20562 declared ``runtime_profiles`` on the ten contracts the allowlist held and
deleted the allowlist. These tests keep it deleted: no allowlist file may exist at
either path the validator would read, the CI job and the pre-commit hook invoke the
validator with no allowlist argument, the whole ``src`` tree passes, a planted
violation fails, and CI Summary requires the job.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.validation.validator_runtime_profiles import (
    ValidatorRuntimeProfiles,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MANUAL_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "validator-runtime-profiles.yml"
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
PLAIN_CALL = "ValidatorRuntimeProfiles().validate(Path("

# The path this repo used, and the repo-root path the core validator discovers on
# its own by walking up from a contract. Either one would reintroduce exceptions.
ALLOWLIST_PATHS = (
    REPO_ROOT / "config" / "validation" / "runtime_profiles_allowlist.yaml",
    REPO_ROOT / "validation" / "runtime_profiles_allowlist.yaml",
)

PLANTED_CONTRACT = """\
name: "node_planted_effect"
node_type: "EFFECT_GENERIC"
contract_version:
  major: 1
  minor: 0
  patch: 0
event_bus:
  subscribe_topics:
    - "onex.cmd.omnibase-infra.planted-command.v1"
"""


@pytest.mark.parametrize(
    "path", ALLOWLIST_PATHS, ids=lambda p: p.name + ":" + p.parent.name
)
def test_no_runtime_profiles_allowlist_exists(path: Path) -> None:
    assert not path.exists(), (
        f"{path.relative_to(REPO_ROOT)} reintroduces an exception list for the "
        "runtime_profiles validator; fix the contract instead (OMN-20562)"
    )


def test_ci_job_runs_the_validator_with_no_allowlist() -> None:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    job = workflow["jobs"]["runtime-profiles-validator"]
    assert "if" not in job
    assert "needs" not in job
    script = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert PLAIN_CALL in script
    assert "allowlist" not in script
    assert "runtime-profiles-allowlist-oneway" not in workflow["jobs"]


def test_manual_workflow_runs_the_validator_with_no_allowlist() -> None:
    text = MANUAL_WORKFLOW.read_text()
    assert PLAIN_CALL in text
    assert "allowlist_path" not in text


def test_precommit_hook_runs_the_validator_with_no_allowlist() -> None:
    repos = yaml.safe_load(PRECOMMIT_CONFIG.read_text())["repos"]
    hooks = [hook for repo in repos for hook in repo.get("hooks", [])]
    (hook,) = [h for h in hooks if h.get("id") == "onex-validate-runtime-profiles"]
    assert PLAIN_CALL in hook["entry"]
    assert "allowlist" not in hook["entry"]
    assert not [
        h for h in hooks if h.get("alias") == "anti-growth-runtime-profiles-allowlist"
    ]


def test_ci_summary_requires_the_job() -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))
    import ci_summary_gate as gate

    assert "Runtime Profiles / validate" in gate.STRICT_GATE_JOBS


def test_every_contract_under_src_passes() -> None:
    result = ValidatorRuntimeProfiles().validate(REPO_ROOT / "src")
    assert result.is_valid, [issue.message for issue in result.issues]


def test_planted_violation_fails(tmp_path: Path) -> None:
    contract = tmp_path / "node_planted_effect" / "contract.yaml"
    contract.parent.mkdir()
    contract.write_text(PLANTED_CONTRACT)
    result = ValidatorRuntimeProfiles().validate(tmp_path)
    assert not result.is_valid
    assert any("runtime_profiles missing" in issue.message for issue in result.issues)
