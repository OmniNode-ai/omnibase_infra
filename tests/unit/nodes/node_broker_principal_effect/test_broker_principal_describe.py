# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20823: ``broker.principal.describe``, the mint audit's read-only ACL read.

The audit of ``make mint-local`` compares a principal's ACLs with its plan with
no password and no mutation. Describe reads them through the same confluent
client the apply and verify handlers use and keeps every binding the broker
reports: an ALLOW, a DENY, a host-scoped ALLOW and an operation the plan cannot
express each come back as their own binding, so the audit can name each one. A
reply that is not a list of bindings is a typed error, never an empty set, and
the handler never creates an ACL and takes no password.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

import pytest
from confluent_kafka import KafkaException
from confluent_kafka.admin import (
    AclBinding,
    AclBindingFilter,
    AclOperation,
    AclPermissionType,
    ResourcePatternType,
    ResourceType,
)

from omnibase_infra.handlers.handler_broker_principal_describe import (
    HandlerBrokerPrincipalDescribe,
)
from omnibase_infra.handlers.handler_broker_principal_list import (
    BrokerPrincipalError,
    BrokerPrincipalKafkaAcls,
)
from omnibase_infra.handlers.handler_developer_principal_mint import (
    HandlerDeveloperPrincipalMint,
)
from omnibase_infra.models.developer_principal import (
    ModelBrokerPrincipalDescribeRequest,
    ModelBrokerPrincipalDescribeResult,
    ModelBrokerPrincipalListResult,
    ModelDeveloperPrincipalMintRequest,
    ModelDeveloperPrincipalPlan,
)

pytestmark = pytest.mark.unit

PRINCIPAL = "dev-person-mac"
TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"


def _binding(
    operation: AclOperation,
    permission: AclPermissionType = AclPermissionType.ALLOW,
    host: str = "*",
) -> AclBinding:
    return AclBinding(
        ResourceType.TOPIC,
        TOPIC,
        ResourcePatternType.LITERAL,
        f"User:{PRINCIPAL}",
        host,
        operation,
        permission,
    )


class _FakeKafkaAdmin:
    """Answers describe_acls with a canned reply; any create is a test failure."""

    def __init__(self, reply: object = None, error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.filters: list[AclBindingFilter] = []
        self.creates = 0

    def describe_acls(
        self, acl_binding_filter: AclBindingFilter, **_: Any
    ) -> Future[object]:
        self.filters.append(acl_binding_filter)
        future: Future[object] = Future()
        if self._error is not None:
            future.set_exception(self._error)
        else:
            future.set_result(self._reply)
        return future

    def create_acls(
        self, acls: list[AclBinding], **_: Any
    ) -> dict[AclBinding, Future[object]]:
        self.creates += 1
        raise AssertionError("describe never creates an ACL")


def _describe(admin: _FakeKafkaAdmin) -> ModelBrokerPrincipalDescribeResult:
    handler = HandlerBrokerPrincipalDescribe(
        acls=BrokerPrincipalKafkaAcls(admin_client=admin, timeout_seconds=1.0)
    )
    return handler.handle(ModelBrokerPrincipalDescribeRequest(principal=PRINCIPAL))


def test_describe_keeps_allow_deny_host_scoped_and_alter_bindings() -> None:
    admin = _FakeKafkaAdmin(
        reply=[
            _binding(AclOperation.WRITE),
            _binding(AclOperation.READ, AclPermissionType.DENY),
            _binding(AclOperation.DESCRIBE, host="203.0.113.9"),
            _binding(AclOperation.ALTER),
        ]
    )
    result = _describe(admin)
    labels = [b.label() for b in result.acls]
    assert result.principal == PRINCIPAL
    assert labels == sorted(
        [
            f"topic:{TOPIC}(literal):write",
            f"topic:{TOPIC}(literal):read[deny host=*]",
            f"topic:{TOPIC}(literal):describe[allow host=203.0.113.9]",
            f"topic:{TOPIC}(literal):alter",
        ]
    )
    # The filter names the principal and matches every permission and operation.
    (acl_filter,) = admin.filters
    assert acl_filter.principal == f"User:{PRINCIPAL}"
    assert acl_filter.permission_type == AclPermissionType.ANY
    assert acl_filter.operation == AclOperation.ANY
    assert admin.creates == 0


def test_describe_an_empty_reply_is_an_empty_set() -> None:
    result = _describe(_FakeKafkaAdmin(reply=[]))
    assert result.acls == ()


@pytest.mark.parametrize("reply", [None, {"bindings": []}, "not a list"])
def test_describe_a_malformed_reply_is_a_typed_error(reply: object) -> None:
    with pytest.raises(BrokerPrincipalError) as raised:
        _describe(_FakeKafkaAdmin(reply=reply))
    assert raised.value.operation == "broker.principal.describe"
    assert raised.value.acls_applied is False
    assert raised.value.password_set is False


def test_describe_a_broker_refusal_is_a_typed_error() -> None:
    with pytest.raises(BrokerPrincipalError) as raised:
        _describe(_FakeKafkaAdmin(error=KafkaException("CLUSTER_AUTHORIZATION_FAILED")))
    assert "broker.principal.describe" in str(raised.value)


def test_describe_takes_no_password_and_changes_nothing() -> None:
    assert set(ModelBrokerPrincipalDescribeRequest.model_fields) == {"principal"}
    admin = _FakeKafkaAdmin(reply=[_binding(AclOperation.WRITE)])
    _describe(admin)
    assert admin.creates == 0


@pytest.mark.parametrize(
    "error_type", [KafkaException, RuntimeError, ValueError, OSError, TimeoutError]
)
def test_describe_omits_raw_client_error_text_and_chain(
    error_type: type[Exception],
) -> None:
    sentinel = "synthetic-secret-20823"
    address = "broker-sensitive.invalid:9092"
    upstream = error_type(f"{sentinel} at {address}")
    admin = _FakeKafkaAdmin(error=upstream)
    with pytest.raises(BrokerPrincipalError) as raised:
        _describe(admin)
    error = raised.value
    # KafkaException is converted to RuntimeError by the real shared adapter.
    expected_type = (
        "RuntimeError" if error_type is KafkaException else error_type.__name__
    )
    assert "could not be read" in str(error)
    assert expected_type in str(error)
    for visible in (str(error), repr(error), repr(error.args), repr(vars(error))):
        assert sentinel not in visible
        assert address not in visible
    assert error.operation == "broker.principal.describe"
    assert error.acls_applied is False
    assert error.password_set is False
    assert error.__cause__ is None
    assert error.__context__ is None
    assert admin.creates == 0


@pytest.mark.parametrize("error_type", [KafkaException, TimeoutError])
def test_audit_result_omits_real_describe_client_error_text(
    error_type: type[Exception],
) -> None:
    sentinel = "synthetic-secret-audit-20823"
    address = "broker-sensitive.invalid:9092"
    admin = _FakeKafkaAdmin(error=error_type(f"{sentinel} at {address}"))
    describe = HandlerBrokerPrincipalDescribe(
        acls=BrokerPrincipalKafkaAcls(admin_client=admin, timeout_seconds=1.0)
    )
    calls: list[str] = []

    def read_acls(
        request: ModelBrokerPrincipalDescribeRequest,
    ) -> ModelBrokerPrincipalDescribeResult:
        calls.append("describe")
        return describe.handle(request)

    handler = HandlerDeveloperPrincipalMint(
        list_step=lambda _: ModelBrokerPrincipalListResult(principals=(PRINCIPAL,)),
        plan_step=lambda _: ModelDeveloperPrincipalPlan(
            decision="create",
            principal=PRINCIPAL,
            lane="local",
            grants=(),
            declaration_digest="d" * 64,
        ),
        describe_step=read_acls,
    )
    result = handler.handle(
        ModelDeveloperPrincipalMintRequest(
            mode="audit",
            declaration_yaml="principal_prefix: dev-\n",
            login="person",
            device="mac",
            admin_url="http://redpanda:9644",
        )
    )
    assert result.status == "failure"
    assert result.failed_step == "describe"
    assert result.acls_applied is False
    assert result.password_set is False
    assert "could not be read" in result.reason
    assert sentinel not in result.model_dump_json()
    assert address not in result.model_dump_json()
    assert calls == ["describe"]
    assert admin.creates == 0
