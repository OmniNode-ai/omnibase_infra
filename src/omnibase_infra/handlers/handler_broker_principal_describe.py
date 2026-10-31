# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerBrokerPrincipalDescribe: read a principal's ACLs, change nothing.

The audit mode of the developer principal mint (plan T4, OMN-20823) compares a
principal's ACLs with its plan without a password and without a mutation. Verify
cannot serve it: verify needs the principal's password, because it also logs in.
This handler reads the ACLs only, as the stack superuser the runtime's Kafka
environment names, and returns every binding the broker reports, a DENY or a
host-scoped ALLOW included. A reply that is not a list of bindings is a typed
error, never an empty set.

Canonical def-B dispatch entrypoint: ``handle(request) -> ModelBrokerPrincipalDescribeResult``.
Ticket: OMN-20823
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.handlers.handler_broker_principal_list import (
    BrokerPrincipalError,
    attempt,
    kafka_acls_from_env,
)
from omnibase_infra.models.developer_principal.model_broker_principal_describe_request import (
    ModelBrokerPrincipalDescribeRequest,
)
from omnibase_infra.models.developer_principal.model_broker_principal_describe_result import (
    ModelBrokerPrincipalDescribeResult,
)
from omnibase_infra.protocols.protocol_broker_principal_acls import (
    ProtocolBrokerPrincipalAcls,
)

OPERATION = "broker.principal.describe"
# Describe talks to the Kafka API only; the error names the broker it read.
BROKER = "the runtime's Kafka broker"


class HandlerBrokerPrincipalDescribe:
    """Effect handler: the principal's ACLs, read only."""

    def __init__(self, acls: ProtocolBrokerPrincipalAcls | None = None) -> None:
        # None under ``onex node``: the ACL client is built per request from the
        # runtime's Kafka environment.
        self._acls = acls

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _fail(self, message: str) -> BrokerPrincipalError:
        return BrokerPrincipalError(
            message,
            operation=OPERATION,
            url=BROKER,
            acls_applied=False,
            password_set=False,
        )

    def handle(
        self, request: ModelBrokerPrincipalDescribeRequest
    ) -> ModelBrokerPrincipalDescribeResult:
        """Read every ACL the broker holds for the principal."""
        acls = attempt(
            lambda: self._acls or kafka_acls_from_env(),
            lambda exc: self._fail(
                f"the broker client could not be built ({type(exc).__name__})"
            ),
            (ValueError, RuntimeHostError),
        )
        held = attempt(
            lambda: acls.list_acls(request.principal),
            lambda exc: self._fail(
                f"the principal's ACLs could not be read ({type(exc).__name__})"
            ),
        )
        return ModelBrokerPrincipalDescribeResult(
            principal=request.principal,
            acls=tuple(sorted(held, key=lambda b: b.label())),
        )


__all__: list[str] = ["HandlerBrokerPrincipalDescribe"]
