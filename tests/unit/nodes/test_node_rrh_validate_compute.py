# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for the RRH validate compute node.

Covers:
- All 13 rules (RRH-1001 through RRH-1701) pass and fail paths.
  Fail paths for conditional rules (RRH-1201, RRH-1301, RRH-1403)
  are tested via contract tightening governance.  RRH-1701 uses
  seam-ticket profile.
- Profile loading and selection
- Contract tightening enforcement
- Seam-ticket profile override
- Verdict derivation
- Contract.yaml validation
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from omnibase_infra.diagnostics.enum_verdict import EnumVerdict
from omnibase_infra.models.rrh import (
    ModelRRHEnvironmentData,
    ModelRRHRepoState,
    ModelRRHRuntimeTarget,
    ModelRRHToolchainVersions,
)
from omnibase_infra.models.rrh.model_rrh_result import ModelRRHResult
from omnibase_infra.nodes.node_rrh_validate_compute.handlers.handler_rrh_validate import (
    HandlerRRHValidate,
)
from omnibase_infra.nodes.node_rrh_validate_compute.models.model_rrh_contract_governance import (
    ModelRRHContractGovernance,
)
from omnibase_infra.nodes.node_rrh_validate_compute.models.model_rrh_validate_request import (
    ModelRRHValidateRequest,
)
from omnibase_infra.nodes.node_rrh_validate_compute.node import NodeRRHValidateCompute

pytestmark = [pytest.mark.unit]

CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_rrh_validate_compute"
    / "contract.yaml"
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def handler() -> HandlerRRHValidate:
    return HandlerRRHValidate()


@pytest.fixture
def clean_env() -> ModelRRHEnvironmentData:
    """Environment data that passes all rules."""
    return ModelRRHEnvironmentData(
        repo_state=ModelRRHRepoState(
            branch="jonah/omn-2136-some-feature",
            head_sha="abc123def456",
            is_dirty=False,
            repo_root="/code/omnibase_infra2",
            remote_url="https://github.com/org/omnibase_infra2.git",
        ),
        runtime_target=ModelRRHRuntimeTarget(
            environment="dev",
            kafka_broker="192.168.86.200:29092",
            kubernetes_context="minikube",
        ),
        toolchain=ModelRRHToolchainVersions(
            pre_commit="3.6.0",
            ruff="0.4.1",
            pytest="8.1.0",
            mypy="1.9.0",
        ),
        collected_at=datetime.now(UTC),
    )


@pytest.fixture
def default_governance() -> ModelRRHContractGovernance:
    return ModelRRHContractGovernance(ticket_id="OMN-2136")


@pytest.fixture
def full_governance() -> ModelRRHContractGovernance:
    """Governance that activates all conditional rules."""
    return ModelRRHContractGovernance(
        ticket_id="OMN-2136",
        evidence_requirements=("tests",),
        interfaces_touched=("topics",),
        deployment_targets=("k8s",),
        expected_branch_pattern=r"jonah/omn-2136.*",
    )


def _make_request(
    env: ModelRRHEnvironmentData,
    profile: str = "default",
    governance: ModelRRHContractGovernance | None = None,
) -> ModelRRHValidateRequest:
    return ModelRRHValidateRequest(
        environment_data=env,
        profile_name=profile,
        governance=governance or ModelRRHContractGovernance(),
        repo_name="omnibase_infra2",
    )


# ---------------------------------------------------------------
# Contract.yaml
# ---------------------------------------------------------------


class TestContractValidation:
    @pytest.fixture(scope="class")
    def contract_data(self) -> dict:
        with CONTRACT_PATH.open() as f:
            data: dict = yaml.safe_load(f)
        return data

    def test_node_type_is_compute(self, contract_data: dict) -> None:
        assert contract_data.get("node_type") == "COMPUTE_GENERIC"

    def test_has_input_model(self, contract_data: dict) -> None:
        assert "input_model" in contract_data

    def test_has_output_model(self, contract_data: dict) -> None:
        assert "output_model" in contract_data

    def test_has_handler_routing(self, contract_data: dict) -> None:
        assert "handler_routing" in contract_data


# ---------------------------------------------------------------
# Handler properties
# ---------------------------------------------------------------


class TestHandlerProperties:
    def test_handler_type(self, handler: HandlerRRHValidate) -> None:
        from omnibase_infra.enums import EnumHandlerType

        assert handler.handler_type == EnumHandlerType.COMPUTE_HANDLER

    def test_handler_category(self, handler: HandlerRRHValidate) -> None:
        from omnibase_infra.enums import EnumHandlerTypeCategory

        assert handler.handler_category == EnumHandlerTypeCategory.COMPUTE


# ---------------------------------------------------------------
# Node declarative check
# ---------------------------------------------------------------


class TestNodeDeclarative:
    def test_node_has_no_custom_methods(self) -> None:
        custom = [
            m
            for m in dir(NodeRRHValidateCompute)
            if not m.startswith("_")
            and m not in dir(NodeRRHValidateCompute.__bases__[0])
        ]
        assert custom == [], f"Node has custom methods: {custom}"


# ---------------------------------------------------------------
# Full pass path
# ---------------------------------------------------------------


class TestFullPassPath:
    def test_all_pass_default_profile(
        self,
        handler: HandlerRRHValidate,
        clean_env: ModelRRHEnvironmentData,
        default_governance: ModelRRHContractGovernance,
    ) -> None:
        request = _make_request(clean_env, "default", default_governance)
        result = handler.handle(request)
        assert isinstance(result, ModelRRHResult)
        assert result.verdict == EnumVerdict.PASS
        assert bool(result) is True

    def test_all_pass_ticket_pipeline(
        self,
        handler: HandlerRRHValidate,
        clean_env: ModelRRHEnvironmentData,
        full_governance: ModelRRHContractGovernance,
    ) -> None:
        request = _make_request(clean_env, "ticket-pipeline", full_governance)
        result = handler.handle(request)
        assert result.verdict == EnumVerdict.PASS


# ---------------------------------------------------------------
# Individual rule tests
# ---------------------------------------------------------------


class TestRRH1001CleanTree:
    def test_dirty_tree_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        dirty_env = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(update={"is_dirty": True})
            }
        )
        request = _make_request(dirty_env)
        result = handler.handle(request)
        failed_ids = [c.rule_id for c in result.checks if c.is_violation()]
        assert "RRH-1001" in failed_ids


class TestRRH1002ExpectedBranch:
    def test_branch_mismatch_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        gov = ModelRRHContractGovernance(expected_branch_pattern=r"main")
        request = _make_request(clean_env, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1002")
        assert not check.passed

    def test_no_pattern_skips(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        gov = ModelRRHContractGovernance()
        request = _make_request(clean_env, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1002")
        assert check.skipped


class TestRRH1101EnvironmentTarget:
    def test_invalid_env_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        bad_env = clean_env.model_copy(
            update={
                "runtime_target": clean_env.runtime_target.model_copy(
                    update={"environment": "invalid_env"}
                )
            }
        )
        request = _make_request(bad_env)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1101")
        assert not check.passed


class TestRRH1102KafkaBrokerConfigured:
    def test_empty_broker_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        no_broker = clean_env.model_copy(
            update={
                "runtime_target": clean_env.runtime_target.model_copy(
                    update={"kafka_broker": ""}
                )
            }
        )
        request = _make_request(no_broker, "ticket-pipeline")
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1102")
        assert not check.passed


class TestRRH1201KafkaReachable:
    def test_topics_activates_kafka_check(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        gov = ModelRRHContractGovernance(interfaces_touched=("topics",))
        request = _make_request(clean_env, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1201")
        # Kafka broker is configured in clean_env, should pass.
        assert check.passed
        assert not check.skipped

    def test_no_broker_with_topics_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        no_kafka = clean_env.model_copy(
            update={
                "runtime_target": clean_env.runtime_target.model_copy(
                    update={"kafka_broker": ""}
                )
            }
        )
        gov = ModelRRHContractGovernance(interfaces_touched=("topics",))
        request = _make_request(no_kafka, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1201")
        assert not check.passed


class TestRRH1301K8sContextValid:
    def test_empty_k8s_context_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        no_k8s = clean_env.model_copy(
            update={
                "runtime_target": clean_env.runtime_target.model_copy(
                    update={"kubernetes_context": ""}
                )
            }
        )
        gov = ModelRRHContractGovernance(deployment_targets=("k8s",))
        request = _make_request(no_k8s, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1301")
        assert not check.passed


class TestRRH1401PrecommitPresent:
    def test_no_precommit_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        no_precommit = clean_env.model_copy(
            update={
                "toolchain": clean_env.toolchain.model_copy(update={"pre_commit": ""})
            }
        )
        request = _make_request(no_precommit, "ticket-pipeline")
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1401")
        assert not check.passed


class TestRRH1402RuffPresent:
    def test_no_ruff_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        no_ruff = clean_env.model_copy(
            update={"toolchain": clean_env.toolchain.model_copy(update={"ruff": ""})}
        )
        request = _make_request(no_ruff, "ticket-pipeline")
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1402")
        assert not check.passed


class TestRRH1403PytestConditional:
    def test_tests_evidence_activates_pytest_check(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        gov = ModelRRHContractGovernance(evidence_requirements=("tests",))
        request = _make_request(clean_env, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1403")
        assert check.passed
        assert not check.skipped

    def test_no_pytest_with_tests_requirement_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        no_pytest = clean_env.model_copy(
            update={"toolchain": clean_env.toolchain.model_copy(update={"pytest": ""})}
        )
        gov = ModelRRHContractGovernance(evidence_requirements=("tests",))
        request = _make_request(no_pytest, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1403")
        assert not check.passed


class TestRRH1404MypyPresent:
    def test_no_mypy_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        no_mypy = clean_env.model_copy(
            update={"toolchain": clean_env.toolchain.model_copy(update={"mypy": ""})}
        )
        request = _make_request(no_mypy, "ticket-pipeline")
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1404")
        assert not check.passed


class TestRRH1501BranchMatchesTicket:
    def test_ticket_in_branch_passes(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        gov = ModelRRHContractGovernance(ticket_id="OMN-2136")
        request = _make_request(clean_env, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1501")
        assert check.passed

    def test_ticket_not_in_branch_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        wrong_branch = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(
                    update={"branch": "feature/unrelated"}
                )
            }
        )
        gov = ModelRRHContractGovernance(ticket_id="OMN-9999")
        request = _make_request(wrong_branch, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1501")
        assert not check.passed


class TestRRH1601DisallowedFields:
    def test_disallowed_fields_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        gov = ModelRRHContractGovernance(disallowed_fields=("dangerous_field",))
        request = _make_request(clean_env, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1601")
        assert not check.passed
        assert check.details is not None
        assert "dangerous_field" in check.details["disallowed_fields"]


class TestRRH1701RepoBoundary:
    """RRH-1701 is only enabled in seam-ticket profile; use is_seam_ticket governance."""

    def test_empty_repo_root_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        no_root = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(update={"repo_root": ""})
            }
        )
        gov = ModelRRHContractGovernance(is_seam_ticket=True, ticket_id="OMN-2136")
        request = _make_request(no_root, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1701")
        assert not check.passed

    def test_mismatched_remote_url_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        mismatch = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(
                    update={"remote_url": "https://github.com/org/different_repo.git"}
                )
            }
        )
        gov = ModelRRHContractGovernance(is_seam_ticket=True, ticket_id="OMN-2136")
        request = _make_request(mismatch, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1701")
        assert not check.passed
        assert check.details is not None
        assert check.details["remote_repo"] == "different_repo"

    def test_ssh_url_mismatch_fails(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        ssh_mismatch = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(
                    update={"remote_url": "git@github.com:org/other_repo.git"}
                )
            }
        )
        gov = ModelRRHContractGovernance(is_seam_ticket=True, ticket_id="OMN-2136")
        request = _make_request(ssh_mismatch, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1701")
        assert not check.passed

    def test_ssh_url_match_passes(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        ssh_match = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(
                    update={"remote_url": "git@github.com:org/omnibase_infra2.git"}
                )
            }
        )
        gov = ModelRRHContractGovernance(is_seam_ticket=True, ticket_id="OMN-2136")
        request = _make_request(ssh_match, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1701")
        assert check.passed

    def test_case_insensitive_match(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        upper_url = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(
                    update={"remote_url": "https://github.com/org/Omnibase_Infra2.git"}
                )
            }
        )
        gov = ModelRRHContractGovernance(is_seam_ticket=True, ticket_id="OMN-2136")
        request = _make_request(upper_url, governance=gov)
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1701")
        assert check.passed


# ---------------------------------------------------------------
# Profile behavior
# ---------------------------------------------------------------


class TestProfileBehavior:
    def test_ci_repair_allows_dirty_tree(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        dirty_env = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(update={"is_dirty": True})
            }
        )
        request = _make_request(dirty_env, "ci-repair")
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1001")
        # ci-repair disables RRH-1001, so it should be skipped.
        assert check.skipped

    def test_seam_ticket_overrides_profile(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        gov = ModelRRHContractGovernance(is_seam_ticket=True)
        request = _make_request(clean_env, "default", gov)
        result = handler.handle(request)
        assert result.profile_name == "seam-ticket"
        # All rules should be active (not skipped) in seam-ticket.
        skipped_ids = [c.rule_id for c in result.checks if c.skipped]
        # RRH-1002 and RRH-1501 may still be skipped if governance
        # fields are empty, which is expected.
        non_governance_skipped = [
            r for r in skipped_ids if r not in ("RRH-1002", "RRH-1501")
        ]
        assert non_governance_skipped == []


# ---------------------------------------------------------------
# Contract tightening
# ---------------------------------------------------------------


class TestContractTightening:
    def test_cant_loosen_enabled_rule(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        """Verify that contract governance cannot disable RRH-1001.

        The default profile enables RRH-1001. Even without evidence_requirements
        or interfaces_touched, the rule should stay enabled.
        """
        request = _make_request(clean_env, "default")
        result = handler.handle(request)
        check = next(c for c in result.checks if c.rule_id == "RRH-1001")
        # Should be evaluated, not skipped.
        assert not check.skipped

    def test_tightening_promotes_warn_to_fail(
        self, handler: HandlerRRHValidate
    ) -> None:
        """Tightening promotes an enabled rule from WARN to FAIL severity.

        Uses a custom profile where RRH-1403 is enabled with WARN severity.
        Contract governance with evidence_requirements=["tests"] must promote
        it to FAIL.
        """
        from omnibase_infra.models.rrh.model_rrh_profile import ModelRRHProfile
        from omnibase_infra.models.rrh.model_rrh_rule_severity import (
            ModelRRHRuleSeverity,
        )

        profile = ModelRRHProfile(
            name="warn-profile",
            description="Test profile with WARN severity on RRH-1403.",
            rules=(
                ModelRRHRuleSeverity(
                    rule_id="RRH-1403", enabled=True, severity=EnumVerdict.WARN
                ),
            ),
        )
        gov = ModelRRHContractGovernance(evidence_requirements=("tests",))
        result = handler._apply_tightening(profile, gov)
        rule = result["RRH-1403"]
        assert rule.enabled
        assert rule.severity == EnumVerdict.FAIL

    def test_tightening_keeps_fail_unchanged(self, handler: HandlerRRHValidate) -> None:
        """Tightening does not alter a rule already at FAIL severity."""
        from omnibase_infra.models.rrh.model_rrh_profile import ModelRRHProfile
        from omnibase_infra.models.rrh.model_rrh_rule_severity import (
            ModelRRHRuleSeverity,
        )

        profile = ModelRRHProfile(
            name="fail-profile",
            description="Test profile with FAIL severity on RRH-1403.",
            rules=(
                ModelRRHRuleSeverity(
                    rule_id="RRH-1403", enabled=True, severity=EnumVerdict.FAIL
                ),
            ),
        )
        gov = ModelRRHContractGovernance(evidence_requirements=("tests",))
        result = handler._apply_tightening(profile, gov)
        rule = result["RRH-1403"]
        assert rule.enabled
        assert rule.severity == EnumVerdict.FAIL

    def test_tightening_enables_disabled_rule(
        self, handler: HandlerRRHValidate
    ) -> None:
        """Tightening enables a disabled rule with FAIL severity."""
        from omnibase_infra.models.rrh.model_rrh_profile import ModelRRHProfile
        from omnibase_infra.models.rrh.model_rrh_rule_severity import (
            ModelRRHRuleSeverity,
        )

        profile = ModelRRHProfile(
            name="disabled-profile",
            description="Test profile with disabled RRH-1201.",
            rules=(
                ModelRRHRuleSeverity(
                    rule_id="RRH-1201", enabled=False, severity=EnumVerdict.WARN
                ),
            ),
        )
        gov = ModelRRHContractGovernance(interfaces_touched=("topics",))
        result = handler._apply_tightening(profile, gov)
        rule = result["RRH-1201"]
        assert rule.enabled
        assert rule.severity == EnumVerdict.FAIL

    def test_tightening_promotes_k8s_warn_to_fail(
        self, handler: HandlerRRHValidate
    ) -> None:
        """Tightening promotes RRH-1301 from WARN to FAIL via deployment_targets."""
        from omnibase_infra.models.rrh.model_rrh_profile import ModelRRHProfile
        from omnibase_infra.models.rrh.model_rrh_rule_severity import (
            ModelRRHRuleSeverity,
        )

        profile = ModelRRHProfile(
            name="k8s-warn-profile",
            description="Test profile with WARN severity on RRH-1301.",
            rules=(
                ModelRRHRuleSeverity(
                    rule_id="RRH-1301", enabled=True, severity=EnumVerdict.WARN
                ),
            ),
        )
        gov = ModelRRHContractGovernance(deployment_targets=("k8s",))
        result = handler._apply_tightening(profile, gov)
        rule = result["RRH-1301"]
        assert rule.enabled
        assert rule.severity == EnumVerdict.FAIL


# ---------------------------------------------------------------
# Verdict derivation
# ---------------------------------------------------------------


class TestVerdictDerivation:
    def test_fail_verdict_on_critical_failure(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        dirty_env = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(update={"is_dirty": True})
            }
        )
        request = _make_request(dirty_env, "ticket-pipeline")
        result = handler.handle(request)
        # RRH-1001 fails with FAIL severity in ticket-pipeline.
        assert result.verdict == EnumVerdict.FAIL
        assert bool(result) is False

    def test_warn_verdict_on_soft_failure(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        bad_env = clean_env.model_copy(
            update={
                "runtime_target": clean_env.runtime_target.model_copy(
                    update={"environment": "custom_env"}
                )
            }
        )
        request = _make_request(bad_env, "default")
        result = handler.handle(request)
        # RRH-1101 has WARN severity in default profile.
        assert result.verdict == EnumVerdict.WARN


# ---------------------------------------------------------------
# ModelRRHResult properties
# ---------------------------------------------------------------


class TestModelRRHResult:
    def test_failed_checks_property(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        dirty_env = clean_env.model_copy(
            update={
                "repo_state": clean_env.repo_state.model_copy(update={"is_dirty": True})
            }
        )
        request = _make_request(dirty_env)
        result = handler.handle(request)
        assert len(result.failed_checks) >= 1
        assert all(not c.passed for c in result.failed_checks)

    def test_applicable_checks_property(
        self, handler: HandlerRRHValidate, clean_env: ModelRRHEnvironmentData
    ) -> None:
        request = _make_request(clean_env)
        result = handler.handle(request)
        for check in result.applicable_checks:
            assert not check.skipped
