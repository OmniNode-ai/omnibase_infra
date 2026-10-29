# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerBrokerPrincipalApply: create a planned principal's ACLs, then its password.

The order is the mint plan's D9 and it is the whole point of this handler:

1. list the broker's users through the Admin API, which is the reachability
   check, so an unreachable Admin API stops the run before any mutation;
2. read the principal's existing ACLs; any ACL outside the plan is drift, and the
   apply is refused with each extra named and nothing deleted or changed;
3. create the planned ACLs the broker lacks;
4. read them back and compare as a set with the plan;
5. only then set the password, read from the environment variable the request
   names (never from the request itself).

A failure raises ``BrokerPrincipalError`` stating which mutations completed
(``acls_applied``, ``password_set``). The password is held in a ``SecretStr``,
scrubbed from any client error text, and appears in no result, log record or
exception; errors are raised through ``attempt``, so none keeps the original
exception as its cause or context.

Canonical def-B dispatch entrypoint: ``handle(request) -> ModelBrokerPrincipalApplyResult``.
Ticket: OMN-20822
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import SecretStr

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.handlers.handler_broker_principal_list import (
    BrokerPrincipalError,
    admin_api_from_env,
    attempt,
    environment_value,
    kafka_acls_from_env,
    scrub,
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
from omnibase_infra.models.developer_principal.model_developer_principal_plan import (
    ModelDeveloperPrincipalPlan,
)
from omnibase_infra.protocols.protocol_broker_principal_acls import (
    ProtocolBrokerPrincipalAcls,
)
from omnibase_infra.protocols.protocol_broker_principal_admin import (
    ProtocolBrokerPrincipalAdmin,
)

OPERATION = "broker.principal.apply"


def planned_bindings(
    plan: ModelDeveloperPrincipalPlan,
) -> frozenset[ModelBrokerAclBinding]:
    """One binding per grant and operation: the set the broker must hold exactly."""
    return frozenset(
        ModelBrokerAclBinding(
            resource=grant.resource,
            name=grant.name,
            pattern=grant.pattern,
            operation=operation,
        )
        for grant in plan.grants
        for operation in grant.operations
    )


def _labels(bindings: frozenset[ModelBrokerAclBinding]) -> str:
    return ", ".join(sorted(b.label() for b in bindings))


class HandlerBrokerPrincipalApply:
    """Effect handler: apply one planned principal in the plan's fixed order."""

    def __init__(
        self,
        admin: ProtocolBrokerPrincipalAdmin | None = None,
        acls: ProtocolBrokerPrincipalAcls | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        # None under ``onex node``: the clients are built per request from the
        # request's Admin API URL and the runtime's Kafka environment.
        self._admin = admin
        self._acls = acls
        self._environ = environ

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _fail(
        self,
        request: ModelBrokerPrincipalApplyRequest,
        message: str,
        *,
        acls_applied: bool = False,
        password_set: bool = False,
    ) -> BrokerPrincipalError:
        return BrokerPrincipalError(
            message,
            operation=OPERATION,
            url=request.admin_url,
            acls_applied=acls_applied,
            password_set=password_set,
        )

    def handle(
        self, request: ModelBrokerPrincipalApplyRequest
    ) -> ModelBrokerPrincipalApplyResult:
        """Apply the plan: list, drift check, ACLs, read-back, then password."""
        plan = request.plan
        if plan.decision == "refused" or not plan.principal:
            raise self._fail(
                request, f"the plan was refused ({plan.refusal_kind}): {plan.refusal}"
            )
        raw = environment_value(request.password_env, self._environ)
        if not raw:
            raise self._fail(
                request,
                f"the password variable {request.password_env} is not set; "
                "nothing was changed",
            )
        password = SecretStr(raw)
        principal = plan.principal
        planned = planned_bindings(plan)

        admin, acls = attempt(
            lambda: (
                self._admin or admin_api_from_env(request.admin_url, self._environ),
                self._acls or kafka_acls_from_env(),
            ),
            lambda exc: self._fail(
                request, f"the broker clients could not be built ({type(exc).__name__})"
            ),
            (ValueError, RuntimeHostError),
        )
        users = attempt(
            admin.list_users,
            lambda exc: self._fail(
                request, f"the Admin API is unreachable ({type(exc).__name__})"
            ),
        )
        exists = principal in users

        held = attempt(
            lambda: acls.list_acls(principal),
            lambda exc: self._fail(
                request,
                f"the principal's ACLs could not be read ({type(exc).__name__})",
            ),
        )
        extras = held - planned
        if extras:
            return ModelBrokerPrincipalApplyResult(
                principal=principal,
                action="refused",
                extra_acls=tuple(sorted(extras, key=lambda b: b.label())),
                refusal=f"{principal} holds ACLs outside the plan: {_labels(extras)}",
            )

        missing = planned - held
        if missing:
            attempt(
                lambda: acls.create_acls(principal, missing),
                lambda exc: self._fail(
                    request,
                    "the broker did not create every planned ACL: "
                    + scrub(str(exc), password),
                ),
            )

        readback = attempt(
            lambda: acls.list_acls(principal),
            lambda exc: self._fail(
                request, f"the ACLs could not be read back ({type(exc).__name__})"
            ),
        )
        if readback != planned:
            raise self._fail(
                request,
                "the ACLs read back differ from the plan; missing: "
                f"{_labels(planned - readback) or 'none'}; extra: "
                f"{_labels(readback - planned) or 'none'}",
            )

        attempt(
            lambda: admin.set_password(principal, password, exists=exists),
            lambda exc: self._fail(
                request,
                "the Admin API did not set the password: " + scrub(str(exc), password),
                acls_applied=True,
            ),
        )

        return ModelBrokerPrincipalApplyResult(
            principal=principal,
            action="rotate" if exists else "create",
            acls_created=len(missing),
            acls_applied=True,
            password_set=True,
        )
