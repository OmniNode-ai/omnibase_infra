# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "audit-runner-routing.py"
POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_runner_routing", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_workflow_audit_rejects_unallowlisted_hosted_runner(
    tmp_path: Path,
) -> None:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "bad.yml").write_text(
        "name: bad\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    policy = {"hosted_runner_allowlist": []}

    findings = module.audit_local_workflows(policy, tmp_path)

    assert len(findings) == 1
    assert findings[0].scope == ".github/workflows/bad.yml"
    assert "OMNI_RUNNER_SELECTOR_V1" in findings[0].message


def test_local_workflow_audit_honors_explicit_allowlist(tmp_path: Path) -> None:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "fork-only.yml").write_text(
        "name: fork-only\njobs:\n  verify:\n    runs-on: ubuntu-latest\n",
        encoding="utf-8",
    )
    policy = {
        "hosted_runner_allowlist": [
            {"path": ".github/workflows/fork-only.yml", "reason": "fork-only"}
        ]
    }

    assert module.audit_local_workflows(policy, tmp_path) == []


def test_policy_tracks_repos_that_drifted_to_hosted_minutes() -> None:
    import yaml

    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))

    assert policy["trusted_runner_variable"]["name"] == "OMNI_TRUSTED_CI_RUNS_ON_JSON"
    assert policy["trusted_runner_variable"]["expected_json"] == (
        '["self-hosted","omnibase-ci"]'
    )
    assert {
        "omnibase_core",
        "omnibase_infra",
        "omniclaude",
        "omnimarket",
        "onex_change_control",
    }.issubset(set(policy["repositories"]))


def test_github_variable_audit_allows_repo_to_inherit_org_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    def fake_variables(args: list[str]) -> list[dict[str, str]]:
        if args == ["--org", "OmniNode-ai"]:
            return [
                {
                    "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
                    "value": '["self-hosted","omnibase-ci"]',
                }
            ]
        return []

    monkeypatch.setattr(module, "_variables", fake_variables)

    policy = {
        "trusted_runner_variable": {
            "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
            "expected_json": '["self-hosted","omnibase-ci"]',
        },
        "repositories": ["omnibase_infra"],
    }

    assert module.audit_github_variables(policy) == []


def test_github_variable_audit_rejects_repo_hosted_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()

    def fake_variables(args: list[str]) -> list[dict[str, str]]:
        if args == ["--org", "OmniNode-ai"]:
            return [
                {
                    "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
                    "value": '["self-hosted","omnibase-ci"]',
                }
            ]
        return [
            {
                "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
                "value": '["ubuntu-latest"]',
            }
        ]

    monkeypatch.setattr(module, "_variables", fake_variables)

    policy = {
        "trusted_runner_variable": {
            "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
            "expected_json": '["self-hosted","omnibase-ci"]',
        },
        "repositories": ["omnibase_core"],
    }

    findings = module.audit_github_variables(policy)

    assert len(findings) == 1
    assert findings[0].scope == "omnibase_core"
    assert "ubuntu-latest" in findings[0].message
