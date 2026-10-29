# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerBrokerPrincipalVerify: the principal's ACLs match the plan and it can log in.

Verify reads the principal's ACLs and compares them as a set with the plan; only
when they match does it authenticate as the principal, with the password from the
environment variable the request names, and list the topics it can see. A
refused login is a typed error whose text never carries the password.

Canonical def-B dispatch entrypoint: ``handle(request) -> ModelBrokerPrincipalVerifyResult``.
Ticket: OMN-20822
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import SecretStr

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.handlers.handler_broker_principal_apply import planned_bindings
from omnibase_infra.handlers.handler_broker_principal_list import (
    BrokerPrincipalError,
    attempt,
    environment_value,
    kafka_acls_from_env,
    kafka_session_from_env,
    scrub,
)
from omnibase_infra.models.developer_principal.model_broker_principal_verify_request import (
    ModelBrokerPrincipalVerifyRequest,
)
from omnibase_infra.models.developer_principal.model_broker_principal_verify_result import (
    ModelBrokerPrincipalVerifyResult,
)
from omnibase_infra.protocols.protocol_broker_principal_acls import (
    ProtocolBrokerPrincipalAcls,
)
from omnibase_infra.protocols.protocol_broker_principal_session import (
    ProtocolBrokerPrincipalSession,
)

OPERATION = "broker.principal.verify"


class HandlerBrokerPrincipalVerify:
    """Effect handler: verify one principal's grants and login."""

    def __init__(
        self,
        acls: ProtocolBrokerPrincipalAcls | None = None,
        session: ProtocolBrokerPrincipalSession | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        # None under ``onex node``: built per request from the runtime's Kafka env.
        self._acls = acls
        self._session = session
        self._environ = environ

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _fail(
        self, request: ModelBrokerPrincipalVerifyRequest, message: str
    ) -> BrokerPrincipalError:
        return BrokerPrincipalError(
            message,
            operation=OPERATION,
            url=request.admin_url,
            acls_applied=False,
            password_set=False,
        )

    def handle(
        self, request: ModelBrokerPrincipalVerifyRequest
    ) -> ModelBrokerPrincipalVerifyResult:
        """Compare the ACLs with the plan, then log in as the principal."""
        principal = request.plan.principal
        planned = planned_bindings(request.plan)
        acls, session = attempt(
            lambda: (
                self._acls or kafka_acls_from_env(),
                self._session or kafka_session_from_env(),
            ),
            lambda exc: self._fail(
                request, f"the broker clients could not be built ({type(exc).__name__})"
            ),
            (ValueError, RuntimeHostError),
        )
        held = attempt(
            lambda: acls.list_acls(principal),
            lambda exc: self._fail(
                request,
                f"the principal's ACLs could not be read ({type(exc).__name__})",
            ),
        )
        if held != planned:
            missing = sorted(b.label() for b in planned - held)
            extra = sorted(b.label() for b in held - planned)
            return ModelBrokerPrincipalVerifyResult(
                principal=principal,
                verified=False,
                reason=f"ACLs differ from the plan; missing: {missing}; extra: {extra}",
            )
        raw = environment_value(request.password_env, self._environ)
        if not raw:
            raise self._fail(
                request, f"the password variable {request.password_env} is not set"
            )
        password = SecretStr(raw)
        topics = attempt(
            lambda: session.list_topics_as(principal, password),
            lambda exc: self._fail(
                request, "the principal could not log in: " + scrub(str(exc), password)
            ),
        )
        return ModelBrokerPrincipalVerifyResult(
            principal=principal,
            verified=True,
            visible_topics=tuple(sorted(topics)),
        )
