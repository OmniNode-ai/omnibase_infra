# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20823: the developer principal mint orchestrator.

AC1: in ``converge`` the handler calls list, plan, apply and verify in that order
and stops at the first failure, so a refused plan makes no apply call; in
``audit`` it calls nothing that mutates and reports every ACL outside the plan.
AC9: neither ``main`` nor ``effects`` owns the node, its contract publishes the
mint request through the runtime publisher inventory, and one pre-commit entry
runs these tests.

The steps are fakes that journal every call into one list, so the order, and
the absence of a call, is what the tests read.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.handlers.handler_broker_principal_list import (
    BrokerPrincipalError,
)
from omnibase_infra.handlers.handler_developer_principal_mint import (
    HandlerDeveloperPrincipalMint,
)
from omnibase_infra.models.developer_principal import (
    ModelBrokerAclBinding,
    ModelBrokerPrincipalApplyRequest,
    ModelBrokerPrincipalApplyResult,
    ModelBrokerPrincipalDescribeRequest,
    ModelBrokerPrincipalDescribeResult,
    ModelBrokerPrincipalListRequest,
    ModelBrokerPrincipalListResult,
    ModelBrokerPrincipalVerifyRequest,
    ModelBrokerPrincipalVerifyResult,
    ModelDeveloperPrincipalGrant,
    ModelDeveloperPrincipalMintRequest,
    ModelDeveloperPrincipalMintResult,
    ModelDeveloperPrincipalPlan,
    ModelDeveloperPrincipalPlanRequest,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
NODE_DIR = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_developer_principal_mint_orchestrator"
)
DECLARATION = (
    REPO_ROOT / "deploy" / "local" / "developer-principal-grants.yaml"
).read_text()
ADMIN_URL = "http://redpanda:9644"
PASSWORD_ENV = "ONEX_MINT_PASSWORD"
SENTINEL = "sentinel-pw-7c1d2e3f4a5b"
PRINCIPAL = "dev-person-mac"
TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"

GRANTS = (
    ModelDeveloperPrincipalGrant(
        resource="topic",
        name=TOPIC,
        pattern="literal",
        operations=("describe", "write"),
    ),
)
PLANNED = (
    ModelBrokerAclBinding(
        resource="topic", name=TOPIC, pattern="literal", operation="describe"
    ),
    ModelBrokerAclBinding(
        resource="topic", name=TOPIC, pattern="literal", operation="write"
    ),
)
DENY = ModelBrokerAclBinding(
    resource="topic", name=TOPIC, pattern="literal", operation="read", permission="deny"
)
HOST_SCOPED = ModelBrokerAclBinding(
    resource="topic",
    name=TOPIC,
    pattern="literal",
    operation="describe",
    host="203.0.113.9",
)
MUTATING = {"apply", "verify"}


Decision = Literal["create", "rotate", "refused"]
ApplyAction = Literal["create", "rotate", "refused"]
Mode = Literal["converge", "audit"]


def _plan(decision: Decision = "create") -> ModelDeveloperPrincipalPlan:
    if decision == "refused":
        return ModelDeveloperPrincipalPlan(
            decision="refused",
            refusal_kind="device_cap",
            refusal="dev- already holds 5 principals",
        )
    return ModelDeveloperPrincipalPlan(
        decision=decision,
        principal=PRINCIPAL,
        lane="local",
        grants=GRANTS,
        declaration_digest="d" * 64,
    )


class _Steps:
    """Every step of the mint, journalling each call into ``calls``."""

    def __init__(
        self,
        *,
        listed: tuple[str, ...] = (),
        decision: Decision = "create",
        held: tuple[ModelBrokerAclBinding, ...] = PLANNED,
        apply_action: ApplyAction = "create",
        verified: bool = True,
        fail_at: str = "",
    ) -> None:
        self.calls: list[str] = []
        self.requests: list[object] = []
        self._listed = listed
        self._decision: Decision = decision
        self._held = held
        self._apply_action: ApplyAction = apply_action
        self._verified = verified
        self._fail_at = fail_at

    def _enter(self, step: str, request: object) -> None:
        self.calls.append(step)
        self.requests.append(request)
        if step == self._fail_at:
            raise BrokerPrincipalError(
                f"{step} failed",
                operation=f"broker.principal.{step}",
                url=ADMIN_URL,
                acls_applied=step == "apply",
                password_set=False,
            )

    def handler(self) -> HandlerDeveloperPrincipalMint:
        steps = self

        class _List:
            def handle(
                self, request: ModelBrokerPrincipalListRequest
            ) -> ModelBrokerPrincipalListResult:
                steps._enter("list", request)
                return ModelBrokerPrincipalListResult(principals=steps._listed)

        class _Plan:
            def handle(
                self, request: ModelDeveloperPrincipalPlanRequest
            ) -> ModelDeveloperPrincipalPlan:
                steps._enter("plan", request)
                return _plan(steps._decision)

        class _Describe:
            def handle(
                self, request: ModelBrokerPrincipalDescribeRequest
            ) -> ModelBrokerPrincipalDescribeResult:
                steps._enter("describe", request)
                return ModelBrokerPrincipalDescribeResult(
                    principal=request.principal, acls=steps._held
                )

        class _Apply:
            def handle(
                self, request: ModelBrokerPrincipalApplyRequest
            ) -> ModelBrokerPrincipalApplyResult:
                steps._enter("apply", request)
                if steps._apply_action == "refused":
                    return ModelBrokerPrincipalApplyResult(
                        principal=PRINCIPAL,
                        action="refused",
                        extra_acls=(DENY,),
                        refusal=f"{PRINCIPAL} holds ACLs outside the plan: {DENY.label()}",
                    )
                return ModelBrokerPrincipalApplyResult(
                    principal=PRINCIPAL,
                    action=steps._apply_action,
                    acls_created=len(PLANNED),
                    acls_applied=True,
                    password_set=True,
                )

        class _Verify:
            def handle(
                self, request: ModelBrokerPrincipalVerifyRequest
            ) -> ModelBrokerPrincipalVerifyResult:
                steps._enter("verify", request)
                return ModelBrokerPrincipalVerifyResult(
                    principal=PRINCIPAL,
                    verified=steps._verified,
                    reason="" if steps._verified else "ACLs differ from the plan",
                )

        return HandlerDeveloperPrincipalMint(
            list_step=_List().handle,
            plan_step=_Plan().handle,
            describe_step=_Describe().handle,
            apply_step=_Apply().handle,
            verify_step=_Verify().handle,
        )


def _request(mode: Mode = "converge") -> ModelDeveloperPrincipalMintRequest:
    return ModelDeveloperPrincipalMintRequest(
        mode=mode,
        declaration_yaml=DECLARATION,
        login="person",
        device="mac",
        admin_url=ADMIN_URL,
        password_env_ref=PASSWORD_ENV if mode == "converge" else "",
    )


def _mint(steps: _Steps, mode: Mode = "converge") -> ModelDeveloperPrincipalMintResult:
    return steps.handler().handle(_request(mode))


# --- AC1: converge order ----------------------------------------------------------


def test_order_converge_calls_list_plan_apply_verify() -> None:
    steps = _Steps()
    result = _mint(steps)
    assert steps.calls == ["list", "plan", "apply", "verify"]
    assert result.status == "success"
    assert result.action == "create"
    assert result.verified and result.acls_applied and result.password_set
    assert result.principal == PRINCIPAL
    assert result.grant_count == len(PLANNED)


def test_order_list_feeds_the_plan_and_the_plan_feeds_apply_and_verify() -> None:
    steps = _Steps(
        listed=(PRINCIPAL, "dev-other-air"), decision="rotate", apply_action="rotate"
    )
    result = _mint(steps)
    listed, planned, applied, verified = steps.requests
    assert isinstance(listed, ModelBrokerPrincipalListRequest)
    assert listed.prefix == "dev-"
    assert isinstance(planned, ModelDeveloperPrincipalPlanRequest)
    assert planned.existing_principals == (PRINCIPAL, "dev-other-air")
    assert planned.login == "person" and planned.device == "mac"
    for request in (applied, verified):
        assert isinstance(
            request,
            ModelBrokerPrincipalApplyRequest | ModelBrokerPrincipalVerifyRequest,
        )
        assert request.plan == _plan("rotate")
        assert request.password_env == PASSWORD_ENV
    assert result.action == "rotate"
    assert result.exists is True


def test_order_plan_refusal_makes_no_apply_call() -> None:
    steps = _Steps(decision="refused")
    result = _mint(steps)
    assert steps.calls == ["list", "plan"]
    assert result.status == "failure"
    assert result.failed_step == "plan"
    assert "device_cap" in result.reason


def test_order_apply_refusal_makes_no_verify_call_and_names_extras() -> None:
    steps = _Steps(listed=(PRINCIPAL,), decision="rotate", apply_action="refused")
    result = _mint(steps)
    assert steps.calls == ["list", "plan", "apply"]
    assert result.status == "failure"
    assert result.action == "refused"
    assert result.failed_step == "apply"
    assert result.extra_acls == (DENY.label(),)
    assert not result.password_set


@pytest.mark.parametrize(
    ("fail_at", "expected_calls"),
    [
        ("list", ["list"]),
        ("apply", ["list", "plan", "apply"]),
        ("verify", ["list", "plan", "apply", "verify"]),
    ],
)
def test_order_a_failed_step_ends_the_run_and_is_named(
    fail_at: str, expected_calls: list[str]
) -> None:
    steps = _Steps(fail_at=fail_at)
    result = _mint(steps)
    assert steps.calls == expected_calls
    assert result.status == "failure"
    assert result.failed_step == fail_at
    assert f"{fail_at} failed" in result.reason
    assert result.acls_applied is (fail_at in ("apply", "verify"))


def test_verify_error_keeps_apply_outcome() -> None:
    """A verify that raises after apply keeps apply's outcome (OMN-20823)."""
    steps = _Steps(fail_at="verify")
    result = _mint(steps)
    assert steps.calls == ["list", "plan", "apply", "verify"]
    assert result.status == "failure"
    assert result.failed_step == "verify"
    assert "verify failed" in result.reason
    assert result.acls_applied is True
    assert result.password_set is True
    assert result.verified is False


def test_order_a_verify_that_does_not_pass_is_a_failure_after_the_password() -> None:
    steps = _Steps(verified=False)
    result = _mint(steps)
    assert steps.calls == ["list", "plan", "apply", "verify"]
    assert result.status == "failure"
    assert result.failed_step == "verify"
    assert result.password_set is True
    assert result.verified is False


# --- AC1: audit -------------------------------------------------------------------


def test_audit_makes_no_mutating_call() -> None:
    steps = _Steps(listed=(PRINCIPAL,))
    result = _mint(steps, mode="audit")
    assert steps.calls == ["list", "plan", "describe"]
    assert MUTATING.isdisjoint(steps.calls)
    assert result.status == "success"
    assert result.action == "audited"
    assert result.exists is True
    assert result.extra_acls == () and result.missing_acls == ()


def test_audit_reports_every_acl_outside_the_plan() -> None:
    """A DENY and a host-scoped ALLOW are distinct bindings, never equal to a planned one."""
    steps = _Steps(listed=(PRINCIPAL,), held=(*PLANNED, DENY, HOST_SCOPED))
    result = _mint(steps, mode="audit")
    assert MUTATING.isdisjoint(steps.calls)
    assert result.extra_acls == tuple(sorted((DENY.label(), HOST_SCOPED.label())))
    assert result.missing_acls == ()


def test_audit_reports_a_planned_acl_the_broker_lacks() -> None:
    steps = _Steps(listed=(PRINCIPAL,), held=PLANNED[:1])
    result = _mint(steps, mode="audit")
    assert result.missing_acls == (PLANNED[1].label(),)


def test_audit_unknown_principal_makes_no_describe_call() -> None:
    steps = _Steps(listed=("dev-someone-else",))
    result = _mint(steps, mode="audit")
    assert steps.calls == ["list", "plan"]
    assert result.exists is False
    assert result.missing_acls == tuple(sorted(b.label() for b in PLANNED))


def test_audit_takes_no_password_and_converge_needs_one() -> None:
    with pytest.raises(ValidationError, match="audit reads only"):
        ModelDeveloperPrincipalMintRequest(
            mode="audit",
            declaration_yaml=DECLARATION,
            login="person",
            device="mac",
            admin_url=ADMIN_URL,
            password_env_ref=PASSWORD_ENV,
        )
    with pytest.raises(ValidationError, match="converge needs password_env_ref"):
        ModelDeveloperPrincipalMintRequest(
            mode="converge",
            declaration_yaml=DECLARATION,
            login="person",
            device="mac",
            admin_url=ADMIN_URL,
        )


def test_a_declaration_without_a_prefix_calls_no_step() -> None:
    steps = _Steps()
    result = steps.handler().handle(
        ModelDeveloperPrincipalMintRequest(
            mode="audit",
            declaration_yaml="schema: developer-principal-grants.v1\n",
            login="person",
            device="mac",
            admin_url=ADMIN_URL,
        )
    )
    assert steps.calls == []
    assert result.status == "failure"
    assert result.failed_step == "plan"


# --- no secret in the result -------------------------------------------------------


def test_no_secret_the_request_and_result_carry_only_the_variable_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PASSWORD_ENV, SENTINEL)
    for steps in (_Steps(), _Steps(fail_at="verify"), _Steps(apply_action="refused")):
        result = _mint(steps)
        assert SENTINEL not in result.model_dump_json()
        assert SENTINEL not in repr(result)
        # The one password-named field is the boolean that says it was set.
        assert {f for f in type(result).model_fields if "password" in f} == {
            "password_set"
        }
    assert SENTINEL not in _request().model_dump_json()


# --- AC9: wiring --------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["main", "effects"])
def test_profile_ownership_main_and_effects_own_none_of_the_node(profile: str) -> None:
    manifest = discover_contracts_from_paths([NODE_DIR / "contract.yaml"])
    assert [c.name for c in manifest.contracts] == [
        "node_developer_principal_mint_orchestrator"
    ]
    result = filter_manifest_for_runtime_profile(manifest, profile, environ={})
    assert [c.name for c in result.manifest.contracts] == []


def test_profile_ownership_the_local_dev_profile_owns_the_node() -> None:
    """Positive control: the filter is not refusing every contract."""
    manifest = discover_contracts_from_paths([NODE_DIR / "contract.yaml"])
    result = filter_manifest_for_runtime_profile(manifest, "local-dev", environ={})
    assert [c.name for c in result.manifest.contracts] == [
        "node_developer_principal_mint_orchestrator"
    ]


def test_the_contract_routes_the_mint_and_publishes_one_terminal() -> None:
    contract = yaml.safe_load((NODE_DIR / "contract.yaml").read_text())
    assert contract["node_type"] == "ORCHESTRATOR_GENERIC"
    assert contract["runtime_profiles"] == ["local-dev"]
    (route,) = contract["handler_routing"]["handlers"]
    assert route["handler"]["name"] == "HandlerDeveloperPrincipalMint"
    (command,) = contract["event_bus"]["subscribe_topics"]
    assert command == "onex.cmd.omnibase-infra.developer-principal-mint-requested.v1"
    assert contract["event_bus"]["publish_topics"] == [contract["terminal_event"]]
    inventory = yaml.safe_load(
        (REPO_ROOT / "src" / "omnibase_infra" / "runtime" / "topics.yaml").read_text()
    )
    assert command in inventory["topics"], "the CLI publisher of the mint request"


def test_pre_commit_entry_runs_the_mint_tests() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("id") == "developer-principal-mint"
    ]
    assert len(hooks) == 1, "no single developer-principal-mint pre-commit entry"
    (hook,) = hooks
    for path in (
        "tests/unit/nodes/node_developer_principal_mint_orchestrator",
        "tests/unit/infra/test_make_mint_local.py",
        "tests/unit/cli/test_delegate_lane_local_omn20823.py",
    ):
        assert path in hook["entry"], path
    assert hook["pass_filenames"] is False
    files = re.compile(hook["files"])
    for path in (
        "src/omnibase_infra/nodes/node_developer_principal_mint_orchestrator/contract.yaml",
        "src/omnibase_infra/handlers/handler_developer_principal_mint.py",
        "Makefile",
        "deploy/local/developer-principal-grants.yaml",
    ):
        assert files.search(path), path
