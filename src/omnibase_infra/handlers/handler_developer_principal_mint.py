# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerDeveloperPrincipalMint: one machine's broker principal, minted or audited.

The handler of ``node_developer_principal_mint_orchestrator`` (plan T4,
OMN-20823). It sequences the broker effect (``node_broker_principal_effect``)
and the plan compute node (``node_developer_principal_plan_compute``) in
process, because ``make mint-local`` runs it with ``onex node`` inside the
stack's ``runtime-effects`` container: the first principal has no login on the
bus it would ask over, and a request over the bus would carry the password
(plan D5, accepted as plan section 13 question 1).

``converge`` calls list, plan, apply and verify in that order and stops at the
first failure: a refused plan makes no apply call, a refused apply (drift) makes
no verify call. ``audit`` calls list, plan and, for a principal the broker
already holds, describe; it compares the principal's ACLs with the plan and
calls nothing that mutates, so it needs no password.

A step's ``BrokerPrincipalError`` becomes a ``failure`` result naming the step
and the mutations it had completed; any other exception propagates. The result
carries no password: the request names the password variable, and every step
error is already scrubbed by the effect.

Canonical def-B dispatch entrypoint: ``handle(request) -> ModelDeveloperPrincipalMintResult``.
Ticket: OMN-20823
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.handler_broker_principal_apply import (
    HandlerBrokerPrincipalApply,
    planned_bindings,
)
from omnibase_infra.handlers.handler_broker_principal_describe import (
    HandlerBrokerPrincipalDescribe,
)
from omnibase_infra.handlers.handler_broker_principal_list import (
    BrokerPrincipalError,
    HandlerBrokerPrincipalList,
)
from omnibase_infra.handlers.handler_broker_principal_verify import (
    HandlerBrokerPrincipalVerify,
)
from omnibase_infra.handlers.handler_developer_principal_plan import (
    HandlerDeveloperPrincipalPlan,
)
from omnibase_infra.models.developer_principal.model_broker_acl_binding import (
    ModelBrokerAclBinding,
)
from omnibase_infra.models.developer_principal.model_broker_principal_apply_request import (
    ModelBrokerPrincipalApplyRequest,
)
from omnibase_infra.models.developer_principal.model_broker_principal_apply_result import (
    ModelBrokerPrincipalApplyResult,
)
from omnibase_infra.models.developer_principal.model_broker_principal_describe_request import (
    ModelBrokerPrincipalDescribeRequest,
)
from omnibase_infra.models.developer_principal.model_broker_principal_describe_result import (
    ModelBrokerPrincipalDescribeResult,
)
from omnibase_infra.models.developer_principal.model_broker_principal_list_request import (
    ModelBrokerPrincipalListRequest,
)
from omnibase_infra.models.developer_principal.model_broker_principal_list_result import (
    ModelBrokerPrincipalListResult,
)
from omnibase_infra.models.developer_principal.model_broker_principal_verify_request import (
    ModelBrokerPrincipalVerifyRequest,
)
from omnibase_infra.models.developer_principal.model_broker_principal_verify_result import (
    ModelBrokerPrincipalVerifyResult,
)
from omnibase_infra.models.developer_principal.model_developer_principal_mint_request import (
    ModelDeveloperPrincipalMintRequest,
)
from omnibase_infra.models.developer_principal.model_developer_principal_mint_result import (
    ModelDeveloperPrincipalMintResult,
)
from omnibase_infra.models.developer_principal.model_developer_principal_plan import (
    ModelDeveloperPrincipalPlan,
)
from omnibase_infra.models.developer_principal.model_developer_principal_plan_request import (
    ModelDeveloperPrincipalPlanRequest,
)

Step = Literal["list", "plan", "describe", "apply", "verify"]


class MintStepFailedError(Exception):
    """A step's typed failure, carried to ``handle`` with the step's name."""

    def __init__(self, step: Step, error: BrokerPrincipalError) -> None:
        super().__init__(step)
        self.step: Step = step
        self.error = error


def _run[T](step: Step, call: Callable[[], T]) -> T:
    try:
        return call()
    except BrokerPrincipalError as exc:
        failed = MintStepFailedError(step, exc)
    raise failed


def _labels(bindings: frozenset[ModelBrokerAclBinding]) -> tuple[str, ...]:
    return tuple(sorted(b.label() for b in bindings))


def _result(
    known: dict[str, object], **fields: object
) -> ModelDeveloperPrincipalMintResult:
    """The result, from what the run has learned so far plus this outcome."""
    return ModelDeveloperPrincipalMintResult.model_validate({**known, **fields})


def _principal_prefix(declaration_yaml: str) -> str:
    try:
        document = yaml.safe_load(declaration_yaml)
    except yaml.YAMLError:
        return ""
    prefix = document.get("principal_prefix") if isinstance(document, dict) else None
    return prefix if isinstance(prefix, str) else ""


class HandlerDeveloperPrincipalMint:
    """Orchestrator handler: converge or audit one developer principal."""

    def __init__(
        self,
        *,
        list_step: Callable[
            [ModelBrokerPrincipalListRequest], ModelBrokerPrincipalListResult
        ]
        | None = None,
        plan_step: Callable[
            [ModelDeveloperPrincipalPlanRequest], ModelDeveloperPrincipalPlan
        ]
        | None = None,
        describe_step: Callable[
            [ModelBrokerPrincipalDescribeRequest], ModelBrokerPrincipalDescribeResult
        ]
        | None = None,
        apply_step: Callable[
            [ModelBrokerPrincipalApplyRequest], ModelBrokerPrincipalApplyResult
        ]
        | None = None,
        verify_step: Callable[
            [ModelBrokerPrincipalVerifyRequest], ModelBrokerPrincipalVerifyResult
        ]
        | None = None,
    ) -> None:
        # Under ``onex node`` each step is the real handler's ``handle``; a test
        # injects its own.
        self._list = list_step or HandlerBrokerPrincipalList().handle
        self._plan = plan_step or HandlerDeveloperPrincipalPlan().handle
        self._describe = describe_step or HandlerBrokerPrincipalDescribe().handle
        self._apply = apply_step or HandlerBrokerPrincipalApply().handle
        self._verify = verify_step or HandlerBrokerPrincipalVerify().handle

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        # It reaches the broker through the effect handlers it sequences.
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelDeveloperPrincipalMintRequest
    ) -> ModelDeveloperPrincipalMintResult:
        """Converge or audit the principal; a failed step ends the run."""
        known: dict[str, object] = {
            "mode": request.mode,
            "correlation_id": request.correlation_id,
        }
        try:
            return self._sequence(request, known)
        except MintStepFailedError as failed:
            return _result(
                known,
                status="failure",
                failed_step=failed.step,
                reason=str(failed.error),
                acls_applied=failed.error.acls_applied,
                password_set=failed.error.password_set,
            )

    def _sequence(
        self,
        request: ModelDeveloperPrincipalMintRequest,
        known: dict[str, object],
    ) -> ModelDeveloperPrincipalMintResult:
        prefix = _principal_prefix(request.declaration_yaml)
        if not prefix:
            return _result(
                known,
                status="failure",
                failed_step="plan",
                reason="the declaration names no principal_prefix",
            )

        listed = _run(
            "list",
            lambda: self._list(
                ModelBrokerPrincipalListRequest(
                    admin_url=request.admin_url, prefix=prefix
                )
            ),
        )
        plan = _run(
            "plan",
            lambda: self._plan(
                ModelDeveloperPrincipalPlanRequest(
                    declaration_yaml=request.declaration_yaml,
                    derived_documents=request.derived_documents,
                    login=request.login,
                    device=request.device,
                    existing_principals=listed.principals,
                )
            ),
        )
        planned = planned_bindings(plan)
        exists = bool(plan.principal) and plan.principal in listed.principals
        known.update(
            principal=plan.principal,
            exists=exists,
            grant_count=len(planned),
            declaration_digest=plan.declaration_digest,
        )
        if plan.decision == "refused":
            return _result(
                known,
                status="failure",
                action="refused",
                failed_step="plan",
                reason=f"{plan.refusal_kind}: {plan.refusal}",
            )

        if request.mode == "audit":
            return self._audit(plan, planned, exists=exists, known=known)
        return self._converge(request, plan, known)

    def _audit(
        self,
        plan: ModelDeveloperPrincipalPlan,
        planned: frozenset[ModelBrokerAclBinding],
        *,
        exists: bool,
        known: dict[str, object],
    ) -> ModelDeveloperPrincipalMintResult:
        held: frozenset[ModelBrokerAclBinding] = frozenset()
        if exists:
            described = _run(
                "describe",
                lambda: self._describe(
                    ModelBrokerPrincipalDescribeRequest(principal=plan.principal)
                ),
            )
            held = frozenset(described.acls)
        return _result(
            known,
            status="success",
            action="audited",
            extra_acls=_labels(held - planned),
            missing_acls=_labels(planned - held),
        )

    def _converge(
        self,
        request: ModelDeveloperPrincipalMintRequest,
        plan: ModelDeveloperPrincipalPlan,
        known: dict[str, object],
    ) -> ModelDeveloperPrincipalMintResult:
        applied = _run(
            "apply",
            lambda: self._apply(
                ModelBrokerPrincipalApplyRequest(
                    plan=plan,
                    admin_url=request.admin_url,
                    password_env=request.password_env,
                )
            ),
        )
        if applied.action == "refused":
            return _result(
                known,
                status="failure",
                action="refused",
                failed_step="apply",
                extra_acls=_labels(frozenset(applied.extra_acls)),
                reason=applied.refusal,
            )
        try:
            verified = _run(
                "verify",
                lambda: self._verify(
                    ModelBrokerPrincipalVerifyRequest(
                        plan=plan,
                        admin_url=request.admin_url,
                        password_env=request.password_env,
                    )
                ),
            )
        except MintStepFailedError as failed:
            # OMN-20823: apply already mutated the broker,
            # so its outcome is kept when verify raises.
            return _result(
                known,
                status="failure",
                action=applied.action,
                verified=False,
                acls_applied=applied.acls_applied,
                password_set=applied.password_set,
                failed_step="verify",
                reason=str(failed.error),
            )
        return _result(
            known,
            status="success" if verified.verified else "failure",
            action=applied.action,
            verified=verified.verified,
            acls_applied=applied.acls_applied,
            password_set=applied.password_set,
            failed_step="" if verified.verified else "verify",
            reason=verified.reason,
        )


__all__: list[str] = ["HandlerDeveloperPrincipalMint"]
