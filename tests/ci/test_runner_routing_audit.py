# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
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
    # OMN-18031: the scope gained the job name when this check moved from a
    # whole-file regex to a per-job read of the parsed runs-on value. A file
    # can pin several jobs and only some of them wrongly, so the file alone was
    # never enough to act on.
    assert findings[0].scope == ".github/workflows/bad.yml:test"
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


def test_local_workflow_audit_rejects_dev_base_shortcut(tmp_path: Path) -> None:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "bad.yml").write_text(
        """name: bad
jobs:
  test:
    runs-on: >-
      ${{
        (github.event_name == 'pull_request' && github.base_ref == 'dev')
        && fromJSON(vars.OMNI_PUBLIC_PR_RUNS_ON_JSON)
        || fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON)
      }}
""",
        encoding="utf-8",
    )

    findings = module.audit_local_workflows({"hosted_runner_allowlist": []}, tmp_path)

    assert len(findings) == 2
    assert "dev-base shortcut" in findings[0].message
    assert "head repository differs" in findings[1].message


def test_local_workflow_audit_rejects_public_runner_for_every_pr(
    tmp_path: Path,
) -> None:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "bad.yml").write_text(
        """name: bad
jobs:
  test:
    runs-on: >-
      ${{
        github.event_name == 'pull_request'
        && fromJSON(vars.OMNI_PUBLIC_PR_RUNS_ON_JSON)
        || fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON)
      }}
""",
        encoding="utf-8",
    )

    findings = module.audit_local_workflows({"hosted_runner_allowlist": []}, tmp_path)

    assert len(findings) == 1
    assert "head repository differs" in findings[0].message


def test_runner_variable_selection_is_fork_aware() -> None:
    module = _load_script()

    assert (
        module.runner_variable_for_event(
            "pull_request", "OmniNode-ai/omnibase_infra", "OmniNode-ai/omnibase_infra"
        )
        == "OMNI_TRUSTED_CI_RUNS_ON_JSON"
    )
    assert (
        module.runner_variable_for_event(
            "pull_request", "contributor/omnibase_infra", "OmniNode-ai/omnibase_infra"
        )
        == "OMNI_PUBLIC_PR_RUNS_ON_JSON"
    )
    assert (
        module.runner_variable_for_event("push", None, "OmniNode-ai/omnibase_infra")
        == "OMNI_TRUSTED_CI_RUNS_ON_JSON"
    )
    assert (
        module.runner_variable_for_event(
            "merge_group",
            None,
            "OmniNode-ai/omnibase_infra",
            merge_group_variable="OMNI_REQUIRED_CI_RUNS_ON_JSON",
        )
        == "OMNI_REQUIRED_CI_RUNS_ON_JSON"
    )


def test_repository_workflows_follow_fork_aware_runner_policy() -> None:
    module = _load_script()
    import yaml

    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))

    assert module.audit_local_workflows(policy, REPO_ROOT) == []


def test_local_workflow_audit_rejects_pull_request_target(tmp_path: Path) -> None:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "bad.yml").write_text(
        """name: bad
on: pull_request_target
jobs:
  test:
    runs-on: ubuntu-latest
""",
        encoding="utf-8",
    )

    findings = module.audit_local_workflows(
        {
            "hosted_runner_allowlist": [
                {"path": ".github/workflows/bad.yml", "reason": "test"}
            ]
        },
        tmp_path,
    )

    assert len(findings) == 1
    assert "pull_request_target is prohibited" in findings[0].message


def test_policy_tracks_repos_that_drifted_to_hosted_minutes() -> None:
    import yaml

    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))

    assert policy["trusted_runner_variable"]["name"] == "OMNI_TRUSTED_CI_RUNS_ON_JSON"
    # OMN-16682: the trusted-CI seam was re-flipped to GitHub-hosted runners at
    # org scope and all five repo shadows on 2026-08-27. The policy states the
    # INTENDED value, so it tracks the flip; pinning this assertion to the old
    # self-hosted literal is what kept the hourly audit red on six scopes.
    assert policy["trusted_runner_variable"]["expected_json"] == '["ubuntu-latest"]'
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


def test_github_variable_audit_honors_repository_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-18031: a repo shadow that the policy declares is not drift.

    ``expected_json`` is a single value applied to the org and every repo
    shadow, so before this a deliberately different shadow was indistinguishable
    from silent drift. The interim self-hosted relief flip on omnimarket and
    omnibase_infra is exactly that shape.
    """
    module = _load_script()

    def fake_variables(args: list[str]) -> list[dict[str, str]]:
        if args == ["--org", "OmniNode-ai"]:
            return [
                {
                    "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
                    "value": '["ubuntu-latest"]',
                }
            ]
        return [
            {
                "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
                "value": '["self-hosted","omnibase-ci"]',
            }
        ]

    monkeypatch.setattr(module, "_variables", fake_variables)

    policy = {
        "trusted_runner_variable": {
            "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
            "repository_overrides": {
                "omnimarket": {
                    "expected_json": '["self-hosted","omnibase-ci"]',
                    "revert_when": "interim",
                    "activation_gate": {
                        "sustained_samples": 4,
                        "sustained_min_span_seconds": 3600,
                        "capacity_budget": "measured fan-out leaves headroom",
                        "maintenance_roll_convergence": "roller can drain busy runners",
                        "evidence_companion_fate_isolation": "companion stays hosted",
                        "positive_control_acceptance": "reports file, job, and label",
                    },
                }
            },
        },
        "repositories": ["omnimarket"],
    }

    assert module.audit_github_variables(policy) == []


def test_github_variable_audit_rejects_drift_from_a_repository_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An override is a new assertion to hold, not a hole in the audit."""
    module = _load_script()

    def fake_variables(args: list[str]) -> list[dict[str, str]]:
        if args == ["--org", "OmniNode-ai"]:
            return [
                {
                    "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
                    "value": '["ubuntu-latest"]',
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
            "expected_json": '["ubuntu-latest"]',
            "repository_overrides": {
                "omnimarket": {
                    "expected_json": '["self-hosted","omnibase-ci"]',
                    "revert_when": "interim",
                    "activation_gate": {
                        "sustained_samples": 4,
                        "sustained_min_span_seconds": 3600,
                        "capacity_budget": "measured fan-out leaves headroom",
                        "maintenance_roll_convergence": "roller can drain busy runners",
                        "evidence_companion_fate_isolation": "companion stays hosted",
                        "positive_control_acceptance": "reports file, job, and label",
                    },
                }
            },
        },
        "repositories": ["omnimarket"],
    }

    findings = module.audit_github_variables(policy)

    assert len(findings) == 1
    assert findings[0].scope == "omnimarket"
    assert "self-hosted" in findings[0].message


def test_repository_override_does_not_relax_the_org_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An override is repo-scoped: the org value is still judged on expected_json."""
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
                "value": '["self-hosted","omnibase-ci"]',
            }
        ]

    monkeypatch.setattr(module, "_variables", fake_variables)

    policy = {
        "trusted_runner_variable": {
            "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
            "repository_overrides": {
                "omnimarket": {
                    "expected_json": '["self-hosted","omnibase-ci"]',
                    "revert_when": "interim",
                    "activation_gate": {
                        "sustained_samples": 4,
                        "sustained_min_span_seconds": 3600,
                        "capacity_budget": "measured fan-out leaves headroom",
                        "maintenance_roll_convergence": "roller can drain busy runners",
                        "evidence_companion_fate_isolation": "companion stays hosted",
                        "positive_control_acceptance": "reports file, job, and label",
                    },
                }
            },
        },
        "repositories": ["omnimarket"],
    }

    findings = module.audit_github_variables(policy)

    assert len(findings) == 1
    assert findings[0].scope == "OmniNode-ai"


def test_every_repository_override_carries_a_revert_condition() -> None:
    """A deliberate divergence with no stated end is indistinguishable from drift."""
    module = _load_script()
    import yaml

    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    overrides = policy["trusted_runner_variable"].get("repository_overrides", {})

    # Empty is the correct steady state; the assertion is about SHAPE, so it
    # holds whether or not an override happens to be live right now.
    for repo, entry in overrides.items():
        assert repo in policy["repositories"], f"{repo} is not an audited repository"
        assert entry["expected_json"], f"{repo} override has no expected_json"
        module._canonical_json(entry["expected_json"])
        assert entry.get("revert_when"), f"{repo} override has no revert_when"
        assert entry.get("activation_gate"), f"{repo} override has no activation_gate"


def test_repository_override_without_activation_gate_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future fleet flip needs a measured activation gate, not only a note."""
    module = _load_script()
    monkeypatch.setattr(
        module,
        "_variables",
        lambda args: [
            {
                "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
                "value": '["ubuntu-latest"]',
            }
        ],
    )
    policy = {
        "trusted_runner_variable": {
            "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
            "repository_overrides": {
                "onex_change_control": {
                    "expected_json": '["self-hosted","omnibase-ci"]',
                    "revert_when": "until hosted queue drains",
                }
            },
        },
        "repositories": ["onex_change_control"],
    }

    with pytest.raises(ValueError, match="activation_gate"):
        module.audit_github_variables(policy)


def test_repository_override_activation_gate_requires_all_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sustained samples alone are not enough to authorize a repo shadow flip."""
    module = _load_script()
    monkeypatch.setattr(
        module,
        "_variables",
        lambda args: [
            {
                "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
                "value": '["self-hosted","omnibase-ci"]',
            }
        ],
    )
    policy = {
        "trusted_runner_variable": {
            "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
            "repository_overrides": {
                "onex_change_control": {
                    "expected_json": '["self-hosted","omnibase-ci"]',
                    "revert_when": "until hosted queue drains",
                    "activation_gate": {
                        "sustained_samples": 4,
                    },
                }
            },
        },
        "repositories": ["onex_change_control"],
    }

    with pytest.raises(ValueError, match="sustained_min_span_seconds"):
        module.audit_github_variables(policy)


# ---------------------------------------------------------------------------
# OMN-18031: a bare hosted pin must be caught whatever YAML shape it is written
# in. The check used to be a regex over raw file text anchored on `runs-on:`
# with the label on the SAME line, so every multi-line spelling of the same
# value walked past it. That is a gate failing open, which reads as compliance.
#
# The evasions below are not hypothetical shapes invented for a test: 69 of the
# 72 runs-on declarations in the onex_change_control workflow tree use a folded
# block scalar, and that repo's sibling gate shipped the same line-oriented
# assumption.
# ---------------------------------------------------------------------------

_EVASIONS = {
    "folded_block_scalar": "name: x\njobs:\n  a:\n    runs-on: >-\n      ubuntu-latest\n",
    "literal_block_scalar": "name: x\njobs:\n  a:\n    runs-on: |-\n      ubuntu-latest\n",
    "multiline_sequence": "name: x\njobs:\n  a:\n    runs-on:\n      - ubuntu-latest\n",
    "same_line": "name: x\njobs:\n  a:\n    runs-on: ubuntu-latest\n",
    "inline_sequence": "name: x\njobs:\n  a:\n    runs-on: [ubuntu-latest]\n",
    "group_labels_mapping": (
        "name: x\njobs:\n  a:\n    runs-on:\n      group: g\n      labels:\n        - ubuntu-latest\n"
    ),
}


@pytest.mark.parametrize("shape", sorted(_EVASIONS))
def test_bare_hosted_pin_is_caught_in_every_yaml_shape(
    tmp_path: Path, shape: str
) -> None:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "bad.yml").write_text(_EVASIONS[shape], encoding="utf-8")

    findings = module.audit_local_workflows({"hosted_runner_allowlist": []}, tmp_path)

    assert [f.scope for f in findings] == [".github/workflows/bad.yml:a"], (
        f"a bare hosted pin written as {shape} was not reported"
    )


def test_legacy_line_regex_missed_the_multiline_shapes() -> None:
    """The positive control for the fix: prove the old matcher really failed.

    Without this, "the new check catches it" is unfalsifiable -- a check that
    was never broken and a check that was fixed look identical once green.
    """
    import re

    legacy = re.compile(
        r"^\s*runs-on:\s*(?:\[)?\s*ubuntu-latest(?![\w-])", re.MULTILINE
    )

    # The shapes the old regex did catch.
    assert legacy.search(_EVASIONS["same_line"])
    assert legacy.search(_EVASIONS["inline_sequence"])

    # The shapes it walked straight past, every one a real hosted pin.
    for shape in ("folded_block_scalar", "literal_block_scalar", "multiline_sequence"):
        assert not legacy.search(_EVASIONS[shape]), (
            f"{shape} is expected to defeat the legacy regex; if this now matches, "
            "the control is stale and the fix needs re-justifying"
        )


def test_selector_expression_is_not_a_bare_pin(tmp_path: Path) -> None:
    """A compliant folded selector must not become a false positive."""
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ok.yml").write_text(
        "name: x\njobs:\n  a:\n    runs-on: >-\n"
        "      ${{ fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON"
        ' || \'["self-hosted","omnibase-ci"]\') }}\n',
        encoding="utf-8",
    )

    assert module.audit_local_workflows({"hosted_runner_allowlist": []}, tmp_path) == []


def test_allowlisted_path_still_exempts_a_bare_pin(tmp_path: Path) -> None:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ok.yml").write_text(
        _EVASIONS["folded_block_scalar"], encoding="utf-8"
    )
    policy = {
        "hosted_runner_allowlist": [
            {"path": ".github/workflows/ok.yml", "reason": "test"}
        ]
    }

    assert module.audit_local_workflows(policy, tmp_path) == []


def test_job_delegating_with_uses_is_out_of_scope(tmp_path: Path) -> None:
    """The callee owns placement, so a `uses:` job has no runs-on to audit."""
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ok.yml").write_text(
        "name: x\njobs:\n  a:\n    uses: OmniNode-ai/other/.github/workflows/w.yml@main\n",
        encoding="utf-8",
    )

    assert module.audit_local_workflows({"hosted_runner_allowlist": []}, tmp_path) == []


def test_unreadable_runs_on_is_a_finding_not_an_empty_pass(tmp_path: Path) -> None:
    """Fail loud on a shape we cannot read.

    An unreadable value that returned no labels would be indistinguishable from
    a compliant job -- the same class of silent pass this whole change exists to
    remove.
    """
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "weird.yml").write_text(
        "name: x\njobs:\n  a:\n    runs-on: 17\n", encoding="utf-8"
    )

    findings = module.audit_local_workflows({"hosted_runner_allowlist": []}, tmp_path)

    assert [f.scope for f in findings] == [".github/workflows/weird.yml:a"]
    assert "int" in findings[0].message


def test_job_with_neither_runs_on_nor_uses_is_reported(tmp_path: Path) -> None:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "weird.yml").write_text(
        "name: x\njobs:\n  a:\n    steps:\n      - run: echo hi\n", encoding="utf-8"
    )

    findings = module.audit_local_workflows({"hosted_runner_allowlist": []}, tmp_path)

    assert [f.scope for f in findings] == [".github/workflows/weird.yml:a"]
    assert "neither runs-on nor uses" in findings[0].message


def test_runs_on_labels_reads_each_accepted_shape() -> None:
    module = _load_script()

    assert module.runs_on_labels("ubuntu-latest") == ["ubuntu-latest"]
    assert module.runs_on_labels(["self-hosted", "omnibase-ci"]) == [
        "self-hosted",
        "omnibase-ci",
    ]
    assert module.runs_on_labels({"group": "g", "labels": ["a", "b"]}) == ["a", "b"]
    assert module.runs_on_labels({"group": "g", "labels": "solo"}) == ["solo"]

    for bad in (17, None, {"group": "g", "labels": 3}, ["ok", 5]):
        with pytest.raises(module.UnreadableRunsOnError):
            module.runs_on_labels(bad)
