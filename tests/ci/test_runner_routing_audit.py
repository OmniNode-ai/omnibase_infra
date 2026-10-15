# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


_CALLER = ".github/workflows/call-repo-evidence-gate.yml"
_PINNED = "OmniNode-ai/omnibase_core/.github/workflows/receipt-gate.yml@" + "a" * 40
_EXCEPTION = {
    "path": _CALLER,
    "ruling": "RULING 2026-10-08T22:41:24Z lane=orchestrator-9f8a (OMN-20074)",
    "reason": "test",
}


def _caller_text(
    *,
    trigger: str = "pull_request_target",
    permissions: str = "  contents: read\n  pull-requests: read\n",
    job: str | None = None,
) -> str:
    body = job or f"    uses: {_PINNED}\n    with:\n      evidence-source: caller\n"
    return (
        f"name: Repo Evidence Gate\non:\n  {trigger}:\n    branches: [dev]\n"
        f"permissions:\n{permissions}jobs:\n  repo-evidence:\n{body}"
    )


def _audit_with_caller(tmp_path: Path, text: str, exception: object) -> list:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (tmp_path / _CALLER).write_text(text, encoding="utf-8")
    policy: dict[str, object] = {"hosted_runner_allowlist": []}
    if exception is not None:
        policy["pull_request_target_exception"] = exception
    return module.audit_local_workflows(policy, tmp_path)


def test_pull_request_target_exception_admits_the_named_caller(tmp_path: Path) -> None:
    assert _audit_with_caller(tmp_path, _caller_text(), _EXCEPTION) == []


def test_pull_request_target_exception_admits_no_other_workflow(
    tmp_path: Path,
) -> None:
    module = _load_script()
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (tmp_path / _CALLER).write_text(_caller_text(), encoding="utf-8")
    (workflow_dir / "other.yml").write_text(
        _caller_text().replace("Repo Evidence Gate", "other"), encoding="utf-8"
    )

    findings = module.audit_local_workflows(
        {"pull_request_target_exception": _EXCEPTION}, tmp_path
    )

    assert [f.scope for f in findings] == [".github/workflows/other.yml"]
    assert "pull_request_target is prohibited" in findings[0].message


def test_pull_request_target_is_prohibited_without_a_policy_exception(
    tmp_path: Path,
) -> None:
    findings = _audit_with_caller(tmp_path, _caller_text(), None)

    assert len(findings) == 1
    assert "pull_request_target is prohibited" in findings[0].message


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param({**_EXCEPTION, "path": ".github/workflows/*.yml"}, id="glob"),
        pytest.param({**_EXCEPTION, "path": ".github/workflows/"}, id="directory"),
        pytest.param({**_EXCEPTION, "path": "call-repo-evidence-gate.yml"}, id="bare"),
        pytest.param({**_EXCEPTION, "ruling": ""}, id="no-ruling"),
        pytest.param({**_EXCEPTION, "reason": ""}, id="no-reason"),
        pytest.param([_EXCEPTION], id="list-of-exceptions"),
    ],
)
def test_pull_request_target_exception_must_be_one_named_file_citing_a_ruling(
    tmp_path: Path, exception: object
) -> None:
    findings = _audit_with_caller(tmp_path, _caller_text(), exception)

    messages = [f.message for f in findings]
    assert any("pull_request_target_exception" in m for m in messages), messages
    assert any("pull_request_target is prohibited" in m for m in messages), messages


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        pytest.param(
            _caller_text(trigger="pull_request"),
            "only the pull_request_target trigger",
            id="wrong-trigger",
        ),
        pytest.param(
            _caller_text(permissions="  contents: write\n"),
            "read-only",
            id="write-permission",
        ),
        pytest.param(
            _caller_text(permissions="  contents: read\n"),
            "",
            id="read-only-is-fine",
        ),
        pytest.param(
            _caller_text(
                job="    runs-on: ubuntu-latest\n    steps:\n      - run: x\n"
            ),
            "reusable workflow",
            id="inline-steps",
        ),
        pytest.param(
            _caller_text(job=f"    uses: {_PINNED}\n    secrets: inherit\n"),
            "secrets",
            id="inherits-secrets",
        ),
        pytest.param(
            _caller_text(
                job="    uses: OmniNode-ai/omnibase_core/.github/workflows/receipt-gate.yml@dev\n"
            ),
            "full-SHA",
            id="unpinned",
        ),
    ],
)
def test_pull_request_target_caller_stays_a_pinned_read_only_reusable_call(
    tmp_path: Path, text: str, needle: str
) -> None:
    findings = _audit_with_caller(tmp_path, text, _EXCEPTION)

    if not needle:
        assert findings == []
        return
    assert any(needle in f.message for f in findings), [f.message for f in findings]


def test_policy_names_exactly_the_repo_evidence_caller_as_the_exception() -> None:
    import yaml

    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    exception = policy["pull_request_target_exception"]

    assert exception["path"] == _CALLER
    assert "RULING 2026-10-08T22:41:24Z" in exception["ruling"]
    assert "OMN-20074" in exception["ruling"]


def test_only_the_named_caller_mentions_the_base_branch_trigger() -> None:
    carriers = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / ".github" / "workflows").iterdir()
        if path.is_file() and "pull_request_target" in path.read_text("utf-8")
    )

    assert carriers == [_CALLER]


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


# ---------------------------------------------------------------------------
# OMN-18205: the scoped routing variables, which sit AHEAD of the trusted seam
# ---------------------------------------------------------------------------
#
# These two variables were live for weeks and read by nothing: no rule, no plan
# enumeration, no audit surface. Ten job definitions in this repository resolve
# their runner through the docker one BEFORE the seam is consulted, so a silent
# edit to it moves ten jobs while the seam's own audit reports green.


def _scoped_policy() -> dict[str, object]:
    return {
        "docker_ci_runner_variable": {
            "name": "OMNI_DOCKER_CI_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
            "repositories": ["omnibase_infra"],
            "org_scope": "absent",
        },
        "security_scan_runner_variable": {
            "name": "OMNI_SECURITY_SCAN_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
            "repositories": ["omnibase_infra", "omniclaude"],
            "org_scope": "absent",
        },
        "cloud_canary_runner_variable": {
            "name": "OMNI_CLOUD_CANARY_RUNS_ON_JSON",
            "expected_json": '["self-hosted","omni-cloud-ci"]',
            "repositories": ["omnibase_infra"],
            "org_scope": "absent",
        },
        "repositories": ["omnibase_infra", "omniclaude", "omnimarket"],
    }


def _fake_vars(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    table: dict[str, list[tuple[str, str]]],
) -> None:
    def fake_variables(args: list[str]) -> list[dict[str, str]]:
        key = args[1]
        return [{"name": n, "value": v} for n, v in table.get(key, [])]

    monkeypatch.setattr(module, "_variables", fake_variables)


def test_scoped_variable_audit_passes_on_the_declared_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clean case: declared repos carry the value, others carry nothing."""
    module = _load_script()
    _fake_vars(
        monkeypatch,
        module,
        {
            "OmniNode-ai": [],
            "OmniNode-ai/omnibase_infra": [
                ("OMNI_DOCKER_CI_RUNS_ON_JSON", '["ubuntu-latest"]'),
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]'),
                (
                    "OMNI_CLOUD_CANARY_RUNS_ON_JSON",
                    '["self-hosted","omni-cloud-ci"]',
                ),
            ],
            "OmniNode-ai/omniclaude": [
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]')
            ],
            "OmniNode-ai/omnimarket": [],
        },
    )
    assert module.audit_scoped_variables(_scoped_policy()) == []


def test_scoped_variable_audit_catches_an_undeclared_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finding that matters, and the reason this pass exists.

    A shadow APPEARING on a repository that declares none is the invisible edit:
    nothing else in the estate would ever mention it, and it governs placement
    ahead of the seam. This is the positive control for the clean case above --
    an audit that can only ever pass is indistinguishable from one that is not
    running.
    """
    module = _load_script()
    _fake_vars(
        monkeypatch,
        module,
        {
            "OmniNode-ai": [],
            "OmniNode-ai/omnibase_infra": [
                ("OMNI_DOCKER_CI_RUNS_ON_JSON", '["ubuntu-latest"]'),
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]'),
                (
                    "OMNI_CLOUD_CANARY_RUNS_ON_JSON",
                    '["self-hosted","omni-cloud-ci"]',
                ),
            ],
            "OmniNode-ai/omniclaude": [
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]')
            ],
            "OmniNode-ai/omnimarket": [
                ("OMNI_DOCKER_CI_RUNS_ON_JSON", '["self-hosted","omnibase-ci"]')
            ],
        },
    )
    findings = module.audit_scoped_variables(_scoped_policy())
    assert len(findings) == 1, findings
    assert findings[0].scope == "omnimarket"
    assert "declares no scope" in findings[0].message


def test_scoped_variable_audit_catches_drift_in_a_declared_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    _fake_vars(
        monkeypatch,
        module,
        {
            "OmniNode-ai": [],
            "OmniNode-ai/omnibase_infra": [
                ("OMNI_DOCKER_CI_RUNS_ON_JSON", '["self-hosted","omnibase-ci"]'),
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]'),
                (
                    "OMNI_CLOUD_CANARY_RUNS_ON_JSON",
                    '["self-hosted","omni-cloud-ci"]',
                ),
            ],
            "OmniNode-ai/omniclaude": [
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]')
            ],
            "OmniNode-ai/omnimarket": [],
        },
    )
    findings = module.audit_scoped_variables(_scoped_policy())
    assert len(findings) == 1, findings
    assert findings[0].scope == "omnibase_infra"
    assert "drifted" in findings[0].message


def test_scoped_variable_audit_catches_a_missing_declared_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared scope with no shadow falls through to the seam silently."""
    module = _load_script()
    _fake_vars(
        monkeypatch,
        module,
        {
            "OmniNode-ai": [],
            "OmniNode-ai/omnibase_infra": [
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]'),
                (
                    "OMNI_CLOUD_CANARY_RUNS_ON_JSON",
                    '["self-hosted","omni-cloud-ci"]',
                ),
            ],
            "OmniNode-ai/omniclaude": [
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]')
            ],
            "OmniNode-ai/omnimarket": [],
        },
    )
    findings = module.audit_scoped_variables(_scoped_policy())
    assert len(findings) == 1, findings
    assert "no shadow exists" in findings[0].message


def test_scoped_variable_audit_catches_an_org_level_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An org value for a scoped variable would govern every repository."""
    module = _load_script()
    _fake_vars(
        monkeypatch,
        module,
        {
            "OmniNode-ai": [("OMNI_DOCKER_CI_RUNS_ON_JSON", '["ubuntu-latest"]')],
            "OmniNode-ai/omnibase_infra": [
                ("OMNI_DOCKER_CI_RUNS_ON_JSON", '["ubuntu-latest"]'),
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]'),
                (
                    "OMNI_CLOUD_CANARY_RUNS_ON_JSON",
                    '["self-hosted","omni-cloud-ci"]',
                ),
            ],
            "OmniNode-ai/omniclaude": [
                ("OMNI_SECURITY_SCAN_RUNS_ON_JSON", '["ubuntu-latest"]')
            ],
            "OmniNode-ai/omnimarket": [],
        },
    )
    findings = module.audit_scoped_variables(_scoped_policy())
    assert len(findings) == 1, findings
    assert findings[0].scope == "OmniNode-ai"
    assert "org_scope: absent" in findings[0].message


def test_removing_a_scoped_declaration_is_itself_a_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting the declaration must not silently stop auditing a live variable.

    Without this, the cheapest way to make a drift finding go away would be to
    delete the declaration that produced it.
    """
    module = _load_script()
    _fake_vars(monkeypatch, module, {"OmniNode-ai": []})
    policy = _scoped_policy()
    del policy["docker_ci_runner_variable"]
    findings = module.audit_scoped_variables(policy)
    assert any("is missing from the policy file" in f.message for f in findings)


def test_the_live_policy_declares_every_scoped_variable() -> None:
    """The shipped policy file must carry every declaration, machine-readable."""
    import yaml

    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    for key in (
        "docker_ci_runner_variable",
        "security_scan_runner_variable",
        "cloud_canary_runner_variable",
    ):
        declaration = policy[key]
        assert declaration["org_scope"] == "absent"
        assert isinstance(declaration["repositories"], list)
        assert declaration["repositories"]
        for repo in declaration["repositories"]:
            assert repo in policy["repositories"], (
                f"{key} names {repo}, which the audit never reads"
            )


# ---------------------------------------------------------------------------
# OMN-18205: withdrawn claims must not come back
# ---------------------------------------------------------------------------


def test_the_policy_does_not_assert_the_withdrawn_exclusion_claims() -> None:
    """Two claims in this file were measurably false and are withdrawn.

    Both said, in different words, that the customer-path boundary gate cannot
    run on the self-hosted fleet: that it "needs a hosted class of its own", and
    that flipping either repository shadow "re-runs the 2026-09-07 harm
    verbatim". The runner-topology defect behind both was fixed in #3286, merged
    2026-09-07T15:03:14Z -- two hours and six minutes AFTER the shadow reverted
    to hosted at 12:57:48Z -- so the five-of-five failure A/B those sentences
    rest on is pre-fix by construction, and the gate has since run green on a
    named fleet runner in both repositories.

    This test exists because the file is the asserted-intent surface every
    future flip lane reads, and a stale exclusion reason there gets cited as
    current. That is precisely how the pre-fix A/B blocked two already-fixed
    repositories for four days. The measured A/B itself is deliberately NOT
    pinned here -- it is accurate history and should survive; what must not come
    back is the forward-looking claim attached to it.
    """
    text = POLICY.read_text(encoding="utf-8")
    withdrawn = (
        "needs a hosted class of its own",
        "re-runs the 2026-09-07 harm verbatim",
    )
    present = [claim for claim in withdrawn if claim in text]
    assert not present, (
        "config/runner_routing_policy.yaml asserts a withdrawn claim: "
        f"{present}. The boundary gate has been proven green on a fleet runner; "
        "if it regresses, record the new evidence rather than restoring the old "
        "sentence."
    )


# ---------------------------------------------------------------------------
# OMN-18291: the cloud canary variable
# ---------------------------------------------------------------------------


def test_every_scoped_key_in_the_script_is_declared_in_the_shipped_policy() -> None:
    """The two lists must not drift apart, in either direction.

    A key in the script with no declaration makes every audit run emit a
    "missing from the policy file" finding. A declaration with no key in the
    script is worse and silent: the variable looks audited to anyone reading
    the policy file and is read by nothing.
    """
    import yaml

    module = _load_script()
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))

    for key in module.SCOPED_VARIABLE_KEYS:
        assert key in policy, f"{key} is audited by the script and declared nowhere"

    declared = {
        key
        for key, value in policy.items()
        if isinstance(value, dict) and "org_scope" in value
    }
    assert declared == set(module.SCOPED_VARIABLE_KEYS), (
        "a scoped declaration exists that the script never reads, so it is "
        "documentation rather than an audit"
    )


def test_the_cloud_canary_variable_points_at_the_cloud_fleet_label() -> None:
    """It is the only routing value in the estate that names the cloud fleet.

    Asserted rather than assumed because the label is what makes this a routing
    change AT ALL: the same variable holding a hosted or lab value would read as
    a live canary in every surface while routing nothing.
    """
    import yaml

    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    declaration = policy["cloud_canary_runner_variable"]

    assert declaration["name"] == "OMNI_CLOUD_CANARY_RUNS_ON_JSON"
    assert json.loads(declaration["expected_json"]) == [
        "self-hosted",
        "omni-cloud-ci",
    ]
    assert declaration["repositories"] == ["omnibase_infra"]
    assert declaration["org_scope"] == "absent", (
        "an organization-level value would point EVERY repository's canary job "
        "class at a fleet whose maximum is four instances"
    )
    assert declaration["revert_when"].strip()


def test_the_canary_variable_is_read_ahead_of_the_docker_seam() -> None:
    """Precedence is the whole design, and it is only true if it is written.

    Read from the PARSED runs-on of the one job it governs, so a mention of the
    variable in a comment elsewhere in the file cannot satisfy it.
    """
    import yaml

    workflow = yaml.safe_load(
        (
            REPO_ROOT / ".github" / "workflows" / "application-acl-postgres16-proof.yml"
        ).read_text(encoding="utf-8")
    )
    runs_on = workflow["jobs"]["rebuilt-postgres16-proof"]["runs-on"]

    canary = runs_on.index("OMNI_CLOUD_CANARY_RUNS_ON_JSON")
    docker = runs_on.index("OMNI_DOCKER_CI_RUNS_ON_JSON")
    trusted = runs_on.index("OMNI_TRUSTED_CI_RUNS_ON_JSON")
    assert canary < docker < trusted

    # NEGATIVE CONTROL for the ordering above: the fork branch must still come
    # first of all, because this repository is public and a fork pull request
    # must never reach a self-hosted runner of any kind.
    assert runs_on.index("OMNI_PUBLIC_PR_RUNS_ON_JSON") < canary


def test_the_capacity_requester_cannot_resolve_onto_the_fleet_it_requests() -> None:
    """The requester is hosted by literal, and that is load-bearing.

    A requester that resolved through any routing variable could be relocated
    onto the scale-to-zero fleet it exists to start, which deadlocks: the job
    that would ask for capacity is itself waiting for capacity.
    """
    import yaml

    workflow = yaml.safe_load(
        (
            REPO_ROOT / ".github" / "workflows" / "application-acl-postgres16-proof.yml"
        ).read_text(encoding="utf-8")
    )
    requester = workflow["jobs"]["request-cloud-capacity"]

    assert requester["runs-on"] == "ubuntu-latest"
    assert "${{" not in str(requester["runs-on"])
    assert workflow["jobs"]["rebuilt-postgres16-proof"]["needs"] == (
        "request-cloud-capacity"
    )


def test_the_requester_file_is_allowlisted_for_its_bare_hosted_pin() -> None:
    """Otherwise the local-workflow audit reports it as a routing violation."""
    import yaml

    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in policy["hosted_runner_allowlist"]}
    assert ".github/workflows/application-acl-postgres16-proof.yml" in paths


def test_public_pr_runner_variable_drift_is_reported_at_org_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-16684: audit_github_variables must also audit the public-PR variable.

    Before the fix, ``audit_github_variables`` reads only
    ``policy["trusted_runner_variable"]`` and never looks at
    ``public_pr_runner_variable`` at all, so a drifted org-level value -- the
    exact OMN-16683 shape, where fork PRs routed onto the self-hosted fleet for
    seven weeks -- produces zero findings and the audit reports green.
    """
    module = _load_script()

    def fake_variables(args: list[str]) -> list[dict[str, str]]:
        if args == ["--org", "OmniNode-ai"]:
            return [
                {
                    "name": "OMNI_PUBLIC_PR_RUNS_ON_JSON",
                    "value": '["self-hosted","Linux","X64","omnibase-ci"]',
                },
                {
                    "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
                    "value": '["ubuntu-latest"]',
                },
            ]
        return [
            {"name": "OMNI_TRUSTED_CI_RUNS_ON_JSON", "value": '["ubuntu-latest"]'},
        ]

    monkeypatch.setattr(module, "_variables", fake_variables)

    policy = {
        "trusted_runner_variable": {
            "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
        },
        "public_pr_runner_variable": {
            "name": "OMNI_PUBLIC_PR_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
        },
        "repositories": ["omnibase_infra"],
    }

    findings = module.audit_github_variables(policy)

    assert any(
        f.scope == "OmniNode-ai" and "OMNI_PUBLIC_PR_RUNS_ON_JSON" in f.message
        for f in findings
    ), findings


def test_public_pr_runner_variable_drift_is_reported_at_undeclared_repo_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-18365 AC4 falsifier.

    Sets the public-PR variable on a repository the policy declares should not
    carry it (no ``repository_overrides`` entry for the public-PR variable
    exists anywhere in the live policy) and asserts the audit returns a finding
    naming that scope.
    """
    module = _load_script()

    def fake_variables(args: list[str]) -> list[dict[str, str]]:
        if args == ["--org", "OmniNode-ai"]:
            return [
                {"name": "OMNI_PUBLIC_PR_RUNS_ON_JSON", "value": '["ubuntu-latest"]'},
            ]
        if args == ["--repo", "OmniNode-ai/omnibase_core"]:
            return [
                {
                    "name": "OMNI_PUBLIC_PR_RUNS_ON_JSON",
                    "value": '["self-hosted","omnibase-ci"]',
                },
            ]
        return []

    monkeypatch.setattr(module, "_variables", fake_variables)

    policy = {
        "trusted_runner_variable": {
            "name": "OMNI_TRUSTED_CI_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
        },
        "public_pr_runner_variable": {
            "name": "OMNI_PUBLIC_PR_RUNS_ON_JSON",
            "expected_json": '["ubuntu-latest"]',
        },
        "repositories": ["omnibase_core"],
    }

    findings = module.audit_github_variables(policy)

    assert any(
        f.scope == "omnibase_core" and "self-hosted" in f.message for f in findings
    ), findings


RUNNER_ROUTING_AUDIT_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "runner-routing-audit.yml"
)


def test_second_audit_step_is_not_masked_by_the_first_steps_failure() -> None:
    """OMN-16727: a finding in the local-workflows step must not hide the
    github-vars step.

    ``runner-routing-audit.yml`` runs the two passes as separate ``run:``
    steps. GitHub Actions skips a step whose ``if:`` does not call
    ``always()``/``failure()``/``cancelled()`` once any earlier step fails --
    the default condition is implicitly ANDed with ``success()``. During the
    2026-08-26/27 re-flip, stale policy drift failed the first step and the
    second (the ``OMN-16683`` fork-isolation surface) never ran at all.

    The live pass must run after a local failure. Both passes retain their
    failure status; always() supplies the ordering without error tolerance.
    """
    import yaml

    text = RUNNER_ROUTING_AUDIT_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    steps = workflow["jobs"]["audit"]["steps"]

    local_step = next(
        s for s in steps if s.get("name") == "Audit local workflow routing"
    )
    github_vars_step = next(
        s for s in steps if s.get("name") == "Audit GitHub runner variables"
    )

    assert not local_step.get("continue-on-error", False)
    assert not github_vars_step.get("continue-on-error", False)
    assert "always()" in str(github_vars_step.get("if", "")), (
        "the github-vars step's condition must include always() or a first-"
        "step failure skips it (the implicit success() default)"
    )

    combine_steps = [
        s
        for s in steps
        if "always()" in str(s.get("if", ""))
        and ("outcome" in str(s.get("run", "")) or "outcome" in json.dumps(s))
    ]
    assert combine_steps, (
        "no step evaluates both step outcomes and fails the job once; without "
        "it a per-step green checkmark can still hide a finding"
    )


def test_pr_and_scheduled_audit_execute_the_same_live_assertions() -> None:
    """A PR may not report clean by omitting the pass that finds live drift."""
    import yaml

    workflow = yaml.safe_load(RUNNER_ROUTING_AUDIT_WORKFLOW.read_text())
    steps = workflow["jobs"]["audit"]["steps"]
    live = next(s for s in steps if s.get("id") == "github_vars")
    assert live["if"] == "always()"
    assert (
        live["run"]
        .rstrip()
        .endswith("uv run python scripts/audit-runner-routing.py --github-vars")
    )
    assert 'if [ -z "${GH_TOKEN:-}" ]' in live["run"]
    combined = next(s for s in steps if "combined" in s.get("name", ""))
    # Missing/skipped/error outcomes are not a clean evaluation either.
    assert 'steps.local_workflows.outcome }}" != "success"' in combined["run"]
    assert 'steps.github_vars.outcome }}" != "success"' in combined["run"]
    assert workflow["concurrency"]["group"].startswith("runner-routing-audit-")


def test_runner_audit_is_reused_inside_required_ci_and_precommit() -> None:
    import yaml

    workflow = yaml.safe_load(RUNNER_ROUTING_AUDIT_WORKFLOW.read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert "workflow_call" in triggers
    ci = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    caller = ci["jobs"]["runner-routing-audit"]
    assert caller["uses"] == "./.github/workflows/runner-routing-audit.yml"
    assert "if" not in caller and "needs" not in caller
    gate_path = REPO_ROOT / "scripts/ci/ci_summary_gate.py"
    spec = importlib.util.spec_from_file_location("runner_audit_summary", gate_path)
    assert spec and spec.loader
    summary = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = summary
    spec.loader.exec_module(summary)
    name = "runner-routing-audit / Runner Routing Audit"
    assert name in summary.STRICT_GATE_JOBS
    other_jobs = [
        {"name": n, "status": "completed", "conclusion": "success"}
        for n in summary.GATE_JOBS
        if n != name
    ]
    for conclusion in ("failure", "skipped", "cancelled"):
        result, report = summary.evaluate(
            [
                *other_jobs,
                {"name": name, "status": "completed", "conclusion": conclusion},
            ]
        )
        assert result != 0 and name in report
    result, report = summary.evaluate(other_jobs)
    assert result != 0 and name in report
    result, report = summary.evaluate(
        [*other_jobs, {"name": name, "status": "completed", "conclusion": "success"}]
    )
    assert result == 0, report
    hooks = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        h
        for r in hooks["repos"]
        for h in r["hooks"]
        if h["id"] == "runner-routing-audit"
    )
    assert "tests/ci/test_runner_routing_audit.py" in hook["entry"]
    assert hook["pass_filenames"] is False


@pytest.mark.parametrize(
    ("workflow", "conclusion", "expected_alerts", "listed"),
    [
        (".github/workflows/runner-routing-audit.yml", "failure", 1, True),
        (".github/workflows/runner-routing-audit.yml", "success", 0, True),
        (".github/workflows/dlq-depth-monitor.yml", "failure", 0, True),
        (".github/workflows/runner-routing-audit.yml", "failure", 1, False),
    ],
)
def test_scheduled_routing_audit_reaches_existing_host_reporter(
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    conclusion: str,
    expected_alerts: int,
    listed: bool,
) -> None:
    """Exercise the deployed reader-to-host-reporter path without sending chat."""
    import argparse

    path = REPO_ROOT / "scripts/ci/nonrequired_check_failure_rate.py"
    spec = importlib.util.spec_from_file_location("runner_audit_alerter", path)
    assert spec and spec.loader
    alerter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = alerter
    spec.loader.exec_module(alerter)
    policy = alerter.load_policy(POLICY)
    assert "omnibase_infra" in policy["fleet_repos"]
    assert "omnibase_infra" not in policy["scheduled_actions_repos"]
    monkeypatch.setattr(
        alerter,
        "active_workflows",
        lambda *_: (
            [{"id": 1, "path": workflow}]
            + (
                [{"id": 2, "path": ".github/workflows/runner-routing-audit.yml"}]
                if workflow != ".github/workflows/runner-routing-audit.yml"
                else []
            )
        )
        if listed
        else [],
    )
    reads = []

    def workflow_metadata(path, _token):
        reads.append(path)
        return {"id": 1, "path": workflow, "state": "active"}

    monkeypatch.setattr(alerter, "_gh_object", workflow_metadata)
    run_url = "https://github.com/OmniNode-ai/omnibase_infra/actions/runs/1"
    monkeypatch.setattr(
        alerter,
        "scheduled_runs_for_workflow",
        lambda _slug, workflow_id, *_: [
            {
                "conclusion": conclusion if workflow_id == 1 else "success",
                "created_at": "2026-10-08T02:00:00Z",
                "html_url": run_url,
            }
        ]
        + [{"conclusion": "success", "created_at": "2026-10-07T02:00:00Z"}] * 19,
    )
    alerts: list = []
    report = {"repos": {}}
    alerter._evaluate_one_repo(
        args=argparse.Namespace(),
        repo="omnibase_infra",
        check_runs=False,
        scheduled=True,
        slug="OmniNode-ai/omnibase_infra",
        token=None,
        threshold=3,
        umbrella={},
        scheduled_threshold_pct=policy["scheduled_failure_threshold_pct"],
        scheduled_window_days=policy["scheduled_window_days"],
        scheduled_since="2026-10-01",
        all_alerts=[],
        all_scheduled_alerts=alerts,
        report=report,
        scheduled_workflow_thresholds=policy.get("scheduled_workflow_thresholds", {}),
    )
    assert len(alerts) == expected_alerts
    assert reads == (
        []
        if listed
        else [
            "repos/OmniNode-ai/omnibase_infra/actions/workflows/runner-routing-audit.yml"
        ]
    )
    if not alerts:
        assert (
            report["repos"]["OmniNode-ai/omnibase_infra"]["scheduled"]["alerts"] == []
        )
        return
    assert alerts[0].last_failure_url == run_url
    probe_path = REPO_ROOT / "scripts/omninode-fleet-failure-probe.py"
    probe_spec = importlib.util.spec_from_file_location(
        "runner_audit_fleet_probe", probe_path
    )
    assert probe_spec and probe_spec.loader
    probe = importlib.util.module_from_spec(probe_spec)
    sys.modules[probe_spec.name] = probe
    probe_spec.loader.exec_module(probe)
    rows, heartbeat = probe._rows_from_report(report, policy["failure_threshold"])
    assert len(rows) == 1
    assert rows[0]["status"] == "WARNING"
    assert rows[0]["key"] == "sched/omnibase_infra/runner-routing-audit.yml"
    assert run_url in rows[0]["detail"]
    assert "1 above threshold" in heartbeat


@pytest.mark.parametrize(
    "thresholds",
    [
        [],
        {"outside-fleet": {".github/workflows/runner-routing-audit.yml": 0}},
        {"omnibase_infra": {}},
        {"omnibase_infra": {"not-a-workflow": 0}},
        {"omnibase_infra": {".github/workflows/runner-routing-audit.yml": 11}},
        {"omnibase_infra": {".github/workflows/runner-routing-audit.yml": -1}},
        {"omnibase_infra": {".github/workflows/runner-routing-audit.yml": True}},
        {
            "omnibase_infra": {
                ".github/workflows/runner-routing-audit.yml": float("nan")
            }
        },
    ],
)
def test_scheduled_workflow_thresholds_cannot_relax_or_escape_fleet_policy(
    tmp_path: Path,
    thresholds: object,
) -> None:
    import yaml

    path = REPO_ROOT / "scripts/ci/nonrequired_check_failure_rate.py"
    spec = importlib.util.spec_from_file_location("runner_audit_threshold_policy", path)
    assert spec and spec.loader
    alerter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = alerter
    spec.loader.exec_module(alerter)
    policy = yaml.safe_load(POLICY.read_text())
    policy["route"]["nonrequired_check_alert"]["scheduled_workflow_thresholds"] = (
        thresholds
    )
    fixture = tmp_path / "policy.yaml"
    fixture.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError, match="scheduled_workflow_thresholds"):
        alerter.load_policy(fixture)


def test_repository_universe_derived_from_live_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-18365 AC3 falsifier.

    A fixture private repository carrying one hosted-labelled job on its
    default branch must produce a finding naming that job; a fixture where
    every repository routes onto the fleet must exit clean. The universe of
    repositories to check comes from a live-visibility callable, not from the
    hand-curated ``repositories:`` list in the policy file.
    """
    module = _load_script()

    def fake_visibility(org: str) -> list[tuple[str, bool]]:
        return [("private-repo-with-hosted-job", True), ("public-repo", False)]

    def fake_workflows(repo_name: str) -> dict[str, str]:
        if repo_name == "private-repo-with-hosted-job":
            return {
                ".github/workflows/ci.yml": (
                    "name: ci\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
                )
            }
        if repo_name == "public-repo":
            return {
                ".github/workflows/ci.yml": (
                    "name: ci\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
                )
            }
        return {}

    findings = module.audit_private_repo_hosted_placement(
        visibility_fn=fake_visibility,
        workflow_fetch_fn=fake_workflows,
    )

    assert any(
        "private-repo-with-hosted-job" in f.scope and "build" in f.scope
        for f in findings
    ), findings
    assert not any("public-repo" in f.scope for f in findings), findings

    def fake_workflows_all_fleet(repo_name: str) -> dict[str, str]:
        return {
            ".github/workflows/ci.yml": (
                "name: ci\njobs:\n  build:\n"
                '    runs-on: ["self-hosted", "omnibase-ci"]\n'
            )
        }

    clean_findings = module.audit_private_repo_hosted_placement(
        visibility_fn=fake_visibility,
        workflow_fetch_fn=fake_workflows_all_fleet,
    )
    assert clean_findings == []


def test_live_repository_visibility_uses_a_gh_cli_flag_that_actually_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``gh repo list --visibility all`` is not a real invocation.

    The live post-merge run of this audit (omnibase_infra workflow run
    35003988912, 2026-09-15) crashed with ``invalid argument "all" for
    "--visibility" flag: valid values are {public|private|internal}`` --
    the fixture-based test above mocks ``visibility_fn`` entirely and never
    exercises the real ``gh`` invocation, so it could not catch this. Omitting
    ``--visibility`` returns repositories of every visibility the token can
    see, which is what this check needs since it filters on the returned
    ``isPrivate`` field itself.
    """
    module = _load_script()
    captured_args: list[list[str]] = []

    def fake_run_gh(args: list[str], timeout: int = 20) -> object:
        captured_args.append(args)

        class _Result:
            returncode = 0
            stdout = json.dumps(
                [
                    {"name": "a-private-repo", "isPrivate": True},
                    {"name": "a-public-repo", "isPrivate": False},
                ]
            )
            stderr = ""

        return _Result()

    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    result = module._live_repository_visibility("OmniNode-ai")

    assert captured_args == [
        [
            "repo",
            "list",
            "OmniNode-ai",
            "--no-archived",
            "--limit",
            "500",
            "--json",
            "name,isPrivate",
        ]
    ]
    assert "--visibility" not in captured_args[0]
    # OMN-18780: archived repositories run no workflows, so their pins are
    # out of scope for a merge-gating placement check.
    assert "--no-archived" in captured_args[0]
    assert result == [("a-private-repo", True), ("a-public-repo", False)]


class _GhResult:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _workflow_tree(*entries: dict[str, object]) -> str:
    return json.dumps({"data": {"r0": {"object": {"entries": list(entries)}}}})


def _blob(name: str, text: str, **flags: object) -> dict[str, object]:
    return {
        "name": name,
        "path": f".github/workflows/{name}",
        "type": "blob",
        "object": {"text": text, "isBinary": False, "isTruncated": False, **flags},
    }


def test_workflow_texts_cost_one_graphql_read_per_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-20864: a repository's workflows are ONE read, however many files.

    The contents API read cost one REST call per workflow file plus a listing.
    This pass runs in every omnibase_infra CI run on CROSS_REPO_PAT, the
    operator's token, and across the org's private repositories (the largest
    carries about a hundred workflows) it spent the operator's hourly
    core quota on 2026-10-11 (job 114569998935: rate limit exceeded for the
    operator's user id), which stopped the landing controller with QUOTA.
    """
    module = _load_script()
    calls: list[list[str]] = []
    many = [_blob(f"w{i}.yml", f"name: w{i}\n") for i in range(100)]

    def fake_run_gh(args: list[str], timeout: int = 20) -> _GhResult:
        calls.append(args)
        return _GhResult(_workflow_tree(*many))

    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    texts = module._default_branch_workflow_texts("a-private-repo")

    assert len(calls) == 1
    assert calls[0][:2] == ["api", "graphql"]
    assert "owner=OmniNode-ai" in calls[0] and "n0=a-private-repo" in calls[0]
    assert not any("contents/" in part for part in calls[0])
    assert len(texts) == 100
    assert texts[".github/workflows/w7.yml"] == "name: w7\n"


def test_workflow_texts_keep_only_yaml_blobs(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    tree = _workflow_tree(
        _blob("ci.yml", "a: 1\n"),
        _blob("deploy.yaml", "b: 2\n"),
        _blob("README.md", "docs\n"),
        {"name": "nested.yml", "path": ".github/workflows/nested.yml", "type": "tree"},
    )
    monkeypatch.setattr(module, "_run_gh", lambda args, timeout=20: _GhResult(tree))

    assert module._default_branch_workflow_texts("a-private-repo") == {
        ".github/workflows/ci.yml": "a: 1\n",
        ".github/workflows/deploy.yaml": "b: 2\n",
    }


def test_a_repository_without_a_workflows_directory_reads_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    body = json.dumps({"data": {"r0": {"object": None}}})
    monkeypatch.setattr(module, "_run_gh", lambda args, timeout=20: _GhResult(body))

    assert module._default_branch_workflow_texts("a-private-repo") == {}


@pytest.mark.parametrize(
    "result",
    [
        _GhResult("", returncode=1, stderr="gh: API rate limit exceeded"),
        _GhResult(json.dumps({"data": {"r0": None}})),
        _GhResult(_workflow_tree(_blob("big.yml", "x", isTruncated=True))),
        _GhResult(
            _workflow_tree(
                {
                    "name": "bin.yml",
                    "path": ".github/workflows/bin.yml",
                    "type": "blob",
                    "object": {"text": None, "isBinary": True, "isTruncated": False},
                }
            )
        ),
    ],
    ids=["read-failed", "no-repository", "truncated", "no-text"],
)
def test_an_unread_workflow_raises_instead_of_reading_clean(
    monkeypatch: pytest.MonkeyPatch, result: _GhResult
) -> None:
    """A failed or partial read is never an empty, passing repository (rule 16)."""
    module = _load_script()
    monkeypatch.setattr(module, "_run_gh", lambda args, timeout=20: result)

    with pytest.raises(RuntimeError, match="graphql workflow read"):
        module._default_branch_workflow_texts("a-private-repo")


def test_workflow_texts_batch_reads_45_repositories_in_three_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-20413: repository aliases share one GraphQL read per 20 repos."""
    module = _load_script()
    calls: list[list[str]] = []
    repo_names = [f"private-repo-{i}" for i in range(45)]

    def fake_run_gh(args: list[str], timeout: int = 20) -> _GhResult:
        calls.append(args)
        assert args[:2] == ["api", "graphql"]
        assert "owner=OmniNode-ai" in args
        query = next(
            arg.removeprefix("query=") for arg in args if arg.startswith("query=")
        )
        names = [arg.split("=", 1)[1] for arg in args if arg.startswith("n")]
        assert len(names) <= 20
        data = {}
        for index, name in enumerate(names):
            assert f"$n{index}: String!" in query
            assert f"r{index}: repository(owner: $owner, name: $n{index})" in query
            data[f"r{index}"] = {
                "object": {"entries": [_blob("ci.yml", f"name: {name}\n")]}
            }
        return _GhResult(json.dumps({"data": data}))

    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    texts = module._default_branch_workflow_texts_batch(repo_names)

    assert module.WORKFLOW_READ_BATCH_SIZE == 20
    assert len(calls) == 3
    assert texts == {
        name: {".github/workflows/ci.yml": f"name: {name}\n"} for name in repo_names
    }


def _batch_reader(
    gateway_fails: Callable[[list[str]], bool],
) -> tuple[list[list[str]], Callable[..., _GhResult]]:
    """A fake ``gh api graphql`` that answers HTTP 504 when ``gateway_fails(names)``."""
    calls: list[list[str]] = []

    def fake_run_gh(args: list[str], timeout: int = 20) -> _GhResult:
        names = [arg.split("=", 1)[1] for arg in args if arg.startswith("n")]
        calls.append(names)
        if gateway_fails(names):
            return _GhResult("", returncode=1, stderr="gh: HTTP 504")
        data = {
            f"r{index}": {"object": {"entries": [_blob("ci.yml", f"name: {name}\n")]}}
            for index, name in enumerate(names)
        }
        return _GhResult(json.dumps({"data": data}))

    return calls, fake_run_gh


def test_workflow_texts_batch_gateway_timeout_splits_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-19998: a batch GitHub answers HTTP 504 is read again in halves.

    CI run 38758500873 on dev 421694574 (job 116503220440) failed because one
    seventeen-repository workflow read came back ``gh: HTTP 504`` and the pass
    crashed without a verdict; the scheduled audit fifteen minutes later read
    the same repositories clean. A gateway timeout on a large response is not
    an unreadable repository: smaller reads of the same repositories succeed.
    """
    module = _load_script()
    repo_names = [f"private-repo-{i}" for i in range(17)]
    calls, fake_run_gh = _batch_reader(lambda names: len(names) > 4)
    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    texts = module._default_branch_workflow_texts_batch(repo_names)

    assert texts == {
        name: {".github/workflows/ci.yml": f"name: {name}\n"} for name in repo_names
    }
    assert calls[0] == repo_names
    read = [names for names in calls if len(names) <= 4]
    assert sorted(name for names in read for name in names) == sorted(repo_names)


def test_workflow_texts_batch_gateway_error_on_one_repository_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-19998: splitting never turns an unread repository into a clean one."""
    module = _load_script()
    calls, fake_run_gh = _batch_reader(lambda names: "private-bad" in names)
    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    with pytest.raises(RuntimeError, match=r"private-bad failed: gh: HTTP 504"):
        module._default_branch_workflow_texts_batch(
            ["private-one", "private-bad", "private-two"]
        )
    assert calls.count(["private-bad"]) == 1 + module.WORKFLOW_READ_GATEWAY_RETRIES


@pytest.mark.parametrize(
    "stderr", ["gh: HTTP 401: Bad credentials", "gh: HTTP 404: Not Found"]
)
def test_workflow_texts_batch_non_gateway_failure_is_not_split(
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
) -> None:
    """OMN-19998: only a gateway error is retried; other failures raise at once."""
    module = _load_script()
    calls: list[list[str]] = []

    def fake_run_gh(args: list[str], timeout: int = 20) -> _GhResult:
        calls.append(args)
        return _GhResult("", returncode=1, stderr=stderr)

    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    with pytest.raises(RuntimeError, match="graphql workflow read"):
        module._default_branch_workflow_texts_batch(["private-one", "private-two"])
    assert len(calls) == 1


def test_workflow_texts_batch_rate_limit_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_rate_limit_reset_note", lambda: "")
    monkeypatch.setattr(
        module,
        "_run_gh",
        lambda args, timeout=20: _GhResult(
            "",
            returncode=1,
            stderr="GraphQL: API rate limit already exceeded for user ID 1",
        ),
    )

    with pytest.raises(module.GitHubRateLimitError, match="GitHub API rate limit"):
        module._default_branch_workflow_texts_batch(["private-one", "private-two"])


@pytest.mark.parametrize("repository", [None, "unreadable"])
def test_workflow_texts_batch_null_alias_names_repository(
    monkeypatch: pytest.MonkeyPatch,
    repository: object,
) -> None:
    module = _load_script()
    payload = {"data": {"r0": {"object": None}, "r1": repository}}
    monkeypatch.setattr(
        module, "_run_gh", lambda args, timeout=20: _GhResult(json.dumps(payload))
    )

    with pytest.raises(
        RuntimeError,
        match="graphql workflow read for private-two returned no repository",
    ):
        module._default_branch_workflow_texts_batch(["private-one", "private-two"])


@pytest.mark.parametrize(
    "data", [None, {}, {"r0": None}, {"r0": {"object": None}}, {"r0": {}}]
)
def test_workflow_texts_batch_errors_without_data_raise(
    monkeypatch: pytest.MonkeyPatch,
    data: object,
) -> None:
    module = _load_script()
    payload = {"data": data, "errors": [{"message": "workflow read failed"}]}
    monkeypatch.setattr(
        module, "_run_gh", lambda args, timeout=20: _GhResult(json.dumps(payload))
    )

    with pytest.raises(RuntimeError, match="graphql workflow read for private-one"):
        module._default_branch_workflow_texts_batch(["private-one"])


def test_workflow_texts_batch_names_are_passed_as_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    repo_name = 'repo"with\\special-characters'

    def fake_run_gh(args: list[str], timeout: int = 20) -> _GhResult:
        assert f"n0={repo_name}" in args
        assert "owner=another-org" in args
        query = next(arg for arg in args if arg.startswith("query="))
        assert repo_name not in query
        return _GhResult(json.dumps({"data": {"r0": {"object": None}}}))

    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    assert module._default_branch_workflow_texts_batch([repo_name], "another-org") == {
        repo_name: {}
    }


def test_workflow_texts_batch_empty_input_needs_no_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_run_gh", lambda args: pytest.fail("unexpected read"))

    assert module._default_branch_workflow_texts_batch([]) == {}


def test_private_hosted_placement_batch_default_ignores_public_repos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    calls: list[list[str]] = []

    def fake_run_gh(args: list[str], timeout: int = 20) -> _GhResult:
        calls.append(args)
        if args[:2] == ["repo", "list"]:
            return _GhResult(
                json.dumps(
                    [
                        {"name": "private-repo", "isPrivate": True},
                        {"name": "public-repo", "isPrivate": False},
                        {"name": "private-fleet-repo", "isPrivate": True},
                    ]
                )
            )
        assert args[:2] == ["api", "graphql"]
        assert "n0=private-repo" in args
        assert "n1=private-fleet-repo" in args
        assert not any("public-repo" in arg for arg in args)
        return _GhResult(
            json.dumps(
                {
                    "data": {
                        "r0": {
                            "object": {
                                "entries": [
                                    _blob(
                                        "ci.yml",
                                        "jobs:\n  build:\n    runs-on: ubuntu-latest\n",
                                    )
                                ]
                            }
                        },
                        "r1": {
                            "object": {
                                "entries": [
                                    _blob(
                                        "ci.yml",
                                        "jobs:\n  build:\n    runs-on: [self-hosted, omnibase-ci]\n",
                                    )
                                ]
                            }
                        },
                    }
                }
            )
        )

    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    findings = module.audit_private_repo_hosted_placement()

    assert len(calls) == 2
    assert len(findings) == 1
    assert findings[0].scope == "private-repo:.github/workflows/ci.yml:build"
    assert "pinned to ubuntu-latest" in findings[0].message


def test_private_hosted_placement_batch_fetcher_injection() -> None:
    module = _load_script()
    calls: list[tuple[list[str], str]] = []

    def fake_batch(repo_names: list[str], org: str) -> dict[str, dict[str, str]]:
        calls.append((repo_names, org))
        return {
            name: {
                ".github/workflows/ci.yml": "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
            }
            for name in reversed(repo_names)
        }

    findings = module.audit_private_repo_hosted_placement(
        visibility_fn=lambda org: [
            ("private-two", True),
            ("public", False),
            ("private-one", True),
        ],
        org="another-org",
        workflow_batch_fetch_fn=fake_batch,
    )

    assert calls == [(["private-two", "private-one"], "another-org")]
    assert [finding.scope for finding in findings] == [
        "private-two:.github/workflows/ci.yml:build",
        "private-one:.github/workflows/ci.yml:build",
    ]


def test_each_variable_list_is_read_once_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN-20864: the seam audit and the scoped audit share one read per list."""
    module = _load_script()
    calls: list[list[str]] = []

    def fake_run_gh(args: list[str], timeout: int = 20) -> _GhResult:
        calls.append(args)
        return _GhResult(json.dumps([{"name": "X", "value": "1"}]))

    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    for _ in range(3):
        assert module._variables(["--org", "OmniNode-ai"]) == [
            {"name": "X", "value": "1"}
        ]
        module._variables(["--repo", "OmniNode-ai/omnibase_infra"])

    assert calls == [
        ["variable", "list", "--org", "OmniNode-ai", "--json", "name,value"],
        [
            "variable",
            "list",
            "--repo",
            "OmniNode-ai/omnibase_infra",
            "--json",
            "name,value",
        ],
    ]


# OMN-20864: job 114609134005 (omnibase_infra#4863, 2026-10-11T23:59:14Z) failed
# the live pass on `gh repo list failed: GraphQL: API rate limit already exceeded
# for user ID 1002253.` and the combined step reported it as routing drift.
RATE_LIMITED = "GraphQL: API rate limit already exceeded for user ID 1002253."


def _run_main(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    output = tmp_path / "github_output"
    output.write_text("", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit-runner-routing.py", "--github-vars", "--policy", str(POLICY)],
    )
    code = module.main()
    return code, capsys.readouterr().out, output.read_text(encoding="utf-8")


def test_a_rate_limited_live_read_is_unreadable_not_a_finding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A rate limit is an unread routing state, never routing drift."""
    module = _load_script()
    monkeypatch.setattr(
        module,
        "_run_gh",
        lambda args, timeout=20: _GhResult("", returncode=1, stderr=RATE_LIMITED),
    )

    code, out, github_output = _run_main(module, monkeypatch, tmp_path, capsys)

    assert code == module.EXIT_UNREADABLE
    assert code != 1
    assert "Runner routing drift" not in out
    assert "::error title=Runner audit unreadable::" in out
    assert "rate limit" in out
    assert "findings=0" in github_output.splitlines()
    assert "unreadable=3" in github_output.splitlines()


def test_findings_of_one_live_pass_survive_a_rate_limit_in_another(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    monkeypatch.setattr(
        module,
        "audit_github_variables",
        lambda policy: [module.Finding("omnibase_core", "drifted to hosted")],
    )
    monkeypatch.setattr(
        module,
        "_run_gh",
        lambda args, timeout=20: _GhResult("", returncode=1, stderr=RATE_LIMITED),
    )

    code, out, github_output = _run_main(module, monkeypatch, tmp_path, capsys)

    assert code == 1
    assert "::error title=Runner routing drift::omnibase_core: drifted to hosted" in out
    assert "::error title=Runner audit unreadable::" in out
    assert "findings=1" in github_output.splitlines()
    assert "unreadable=2" in github_output.splitlines()


def test_a_rate_limit_names_when_the_limit_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    rate_limit = json.dumps(
        {
            "resources": {
                "core": {"limit": 5000, "remaining": 4000, "reset": 1791763200},
                "graphql": {"limit": 5000, "remaining": 0, "reset": 1791764160},
            }
        }
    )

    def fake_run_gh(args: list[str], timeout: int = 20) -> _GhResult:
        if args == ["api", "rate_limit"]:
            return _GhResult(rate_limit)
        return _GhResult("", returncode=1, stderr=RATE_LIMITED)

    monkeypatch.setattr(module, "_run_gh", fake_run_gh)

    with pytest.raises(
        module.GitHubRateLimitError, match="resets 2026-10-12T00:16:00Z"
    ):
        module._live_repository_visibility()


def test_a_failed_read_that_is_not_a_rate_limit_is_not_relabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setattr(
        module,
        "_run_gh",
        lambda args, timeout=20: _GhResult("", returncode=1, stderr="HTTP 404"),
    )

    with pytest.raises(RuntimeError, match="gh repo list failed: HTTP 404") as raised:
        module._live_repository_visibility()
    assert not isinstance(raised.value, module.GitHubRateLimitError)


def _render_combined_step(local: str, live: str, findings: str, unreadable: str) -> str:
    import re

    import yaml

    workflow = yaml.safe_load(RUNNER_ROUTING_AUDIT_WORKFLOW.read_text())
    steps = workflow["jobs"]["audit"]["steps"]
    combined = next(s for s in steps if "combined" in s.get("name", ""))
    values = {
        "steps.local_workflows.outcome": local,
        "steps.github_vars.outcome": live,
        "steps.github_vars.outputs.findings": findings,
        "steps.github_vars.outputs.unreadable": unreadable,
    }
    rendered = re.sub(
        r"\$\{\{\s*([A-Za-z0-9_.]+)\s*\}\}",
        lambda m: values[m.group(1)],
        combined["run"],
    )
    return str(rendered)


@pytest.mark.parametrize(
    ("local", "live", "findings", "unreadable", "rc", "present", "absent"),
    [
        (
            "success",
            "failure",
            "0",
            "1",
            1,
            "Runner audit unreadable",
            "Runner routing drift",
        ),
        ("success", "failure", "1", "0", 1, "Runner routing drift", "unreadable"),
        ("failure", "success", "0", "0", 1, "Runner routing drift", "unreadable"),
        ("success", "failure", "", "", 1, "Runner audit incomplete", "drift"),
        ("success", "success", "0", "0", 0, "reported clean", "::error"),
    ],
    ids=["rate-limited", "live-finding", "local-finding", "no-verdict", "clean"],
)
def test_combined_step_names_an_unread_live_pass_as_unreadable_not_drift(
    local: str,
    live: str,
    findings: str,
    unreadable: str,
    rc: int,
    present: str,
    absent: str,
) -> None:
    """OMN-20864: the combined step said "reported findings" for a rate limit."""
    import subprocess

    script = _render_combined_step(local, live, findings, unreadable)
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == rc, result.stdout + result.stderr
    assert present in result.stdout
    assert absent not in result.stdout


@pytest.mark.live_contact(
    "tests/ci/fixtures/runner_routing_audit_rate_limit_omn20864.json"
)
def test_recorded_rate_limit_job_is_unreadable_not_drift(
    recorded_response: dict[str, Any],
) -> None:
    """OMN-20864: replay job 114609134005, whose rate limit was named drift."""
    import subprocess

    lines = recorded_response["response"]["log_lines"]
    module = _load_script()
    stderr = lines["live_pass_error"].split("gh repo list failed: ", 1)[1]
    assert module.RATE_LIMIT_RE.search(stderr)
    assert "reported findings" in lines["combined_annotation"]
    local = lines["local_outcome"].rsplit(": ", 1)[1]
    live = lines["live_outcome"].rsplit(": ", 1)[1]
    assert (local, live) == ("success", "failure")

    # The fixed live pass writes these counts for that read (asserted by
    # test_a_rate_limited_live_read_is_unreadable_not_a_finding).
    script = _render_combined_step(local, live, "0", "1")
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1
    assert "::error title=Runner audit unreadable::" in result.stdout
    # This PR's own job 114620151828 hit the limit with no reset time reported,
    # so the annotation may not promise one.
    assert "reset time when GitHub reports one" in result.stdout
    assert "reported findings" not in result.stdout
