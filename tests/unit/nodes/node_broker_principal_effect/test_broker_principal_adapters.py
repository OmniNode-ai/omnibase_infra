# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20822: the broker principal clients wait for every ACL and keep the secret out.

``AdminClient.create_acls`` returns one future per binding, and they complete
independently. The ACL client must wait for every one before it returns, so the
apply handler's read-back and password step never run while a binding is still
pending, and it must raise when any one failed, naming it. The Admin API client
sends the password only in the request body and never in error text.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
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
from pydantic import SecretStr

from omnibase_infra.handlers.handler_broker_principal_list import (
    BrokerPrincipalAdminApi,
    BrokerPrincipalKafkaAcls,
    BrokerPrincipalKafkaSession,
)
from omnibase_infra.models.developer_principal import ModelBrokerAclBinding

pytestmark = pytest.mark.unit

PRINCIPAL = "dev-dev-person-mac"
SENTINEL = "sentinel-pw-3c2b1a0f9e8d"
BINDINGS = (
    ModelBrokerAclBinding(
        resource="topic", name="onex.cmd.a.v1", pattern="literal", operation="write"
    ),
    ModelBrokerAclBinding(
        resource="topic", name="onex.cmd.b.v1", pattern="literal", operation="describe"
    ),
    ModelBrokerAclBinding(
        resource="group",
        name="local.omnibase_core.runtime_local_",
        pattern="prefixed",
        operation="read",
    ),
)


class _FakeAdminClient:
    """``create_acls`` returns futures completed by timers, as the real client's are."""

    def __init__(self, *, delays: list[float], fail_index: int | None = None) -> None:
        self._delays = delays
        self._fail_index = fail_index
        self.completed: list[int] = []

    def create_acls(
        self, bindings: list[AclBinding], **_: Any
    ) -> dict[AclBinding, Future[None]]:
        futures: dict[AclBinding, Future[None]] = {}
        for index, binding in enumerate(bindings):
            future: Future[None] = Future()
            delay = self._delays[index]

            def _complete(i: int = index, f: Future[None] = future) -> None:
                if i == self._fail_index:
                    f.set_exception(RuntimeError(f"broker refused binding {i}"))
                else:
                    self.completed.append(i)
                    f.set_result(None)

            threading.Timer(delay, _complete).start()
            futures[binding] = future
        return futures


def test_grants_complete_before_password_the_acl_client_awaits_a_delayed_future() -> (
    None
):
    admin = _FakeAdminClient(delays=[0.0, 0.3, 0.05])
    client = BrokerPrincipalKafkaAcls(admin_client=admin, timeout_seconds=5.0)
    started = time.monotonic()
    client.create_acls(PRINCIPAL, BINDINGS)
    assert time.monotonic() - started >= 0.25, (
        "returned before the slow binding completed"
    )
    assert sorted(admin.completed) == [0, 1, 2]


def test_grants_complete_before_password_the_acl_client_raises_when_one_fails() -> None:
    admin = _FakeAdminClient(delays=[0.0, 0.2, 0.0], fail_index=1)
    client = BrokerPrincipalKafkaAcls(admin_client=admin, timeout_seconds=5.0)
    with pytest.raises(RuntimeError) as raised:
        client.create_acls(PRINCIPAL, BINDINGS)
    assert "1 of 3" in str(raised.value)
    assert sorted(admin.completed) == [0, 2], "the other bindings still completed"


def test_grants_complete_before_password_a_future_past_its_timeout_is_a_failure() -> (
    None
):
    admin = _FakeAdminClient(delays=[0.0, 2.0, 0.0])
    client = BrokerPrincipalKafkaAcls(admin_client=admin, timeout_seconds=0.2)
    with pytest.raises(RuntimeError):
        client.create_acls(PRINCIPAL, BINDINGS)


def test_no_secret_the_admin_api_error_text_never_carries_the_password() -> None:
    sent: list[tuple[str, str, bytes | None]] = []

    def _transport(
        method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> bytes:
        sent.append((method, url, body))
        raise OSError(f"HTTP 500 for body {body!r}")

    client = BrokerPrincipalAdminApi(
        admin_url="http://redpanda:9644",
        username="superuser",
        password=SecretStr("superuser-secret"),
        transport=_transport,
    )
    with pytest.raises(OSError) as raised:
        client.set_password(PRINCIPAL, SecretStr(SENTINEL), exists=False)
    assert SENTINEL not in str(raised.value)
    assert "superuser-secret" not in str(raised.value)
    method, url, body = sent[0]
    assert (method, url) == ("POST", "http://redpanda:9644/v1/security/users")
    assert body is not None and SENTINEL.encode() in body


def test_the_admin_api_client_updates_an_existing_user_with_put() -> None:
    sent: list[tuple[str, str]] = []

    def _transport(
        method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> bytes:
        sent.append((method, url))
        return b""

    client = BrokerPrincipalAdminApi(
        admin_url="http://redpanda:9644",
        username="superuser",
        password=SecretStr("superuser-secret"),
        transport=_transport,
    )
    client.set_password(PRINCIPAL, SecretStr(SENTINEL), exists=True)
    assert sent == [("PUT", f"http://redpanda:9644/v1/security/users/{PRINCIPAL}")]


def _chain(exc: BaseException | None) -> Iterator[BaseException]:
    """The exception and every exception reachable through its cause or context."""
    seen: set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack += [current.__cause__, current.__context__]


def _assert_chain_clean(exc: BaseException, *secrets: str) -> None:
    for link in _chain(exc):
        for text in (str(link), repr(link), repr(link.args), repr(vars(link))):
            for secret in secrets:
                assert secret not in text, f"{type(link).__name__} in the chain"


def test_no_secret_the_admin_api_error_chain_never_carries_the_password() -> None:
    def _transport(
        method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> bytes:
        raise OSError(f"HTTP 500 for body {body!r} with {headers!r}")

    client = BrokerPrincipalAdminApi(
        admin_url="http://redpanda:9644",
        username="superuser",
        password=SecretStr("superuser-secret"),
        transport=_transport,
    )
    with pytest.raises(OSError) as raised:
        client.set_password(PRINCIPAL, SecretStr(SENTINEL), exists=False)
    _assert_chain_clean(raised.value, SENTINEL, "superuser-secret")


def test_no_secret_the_login_client_error_chain_never_carries_the_password() -> None:
    class _Refusing:
        def list_topics(self, *args: object, **kwargs: object) -> object:
            raise KafkaException(f"SASL authentication failed for {SENTINEL}")

    session = BrokerPrincipalKafkaSession(
        base_config={"bootstrap.servers": "redpanda:9092"},
        client_factory=lambda _config: _Refusing(),
    )
    with pytest.raises(PermissionError) as raised:
        session.list_topics_as(PRINCIPAL, SecretStr(SENTINEL))
    _assert_chain_clean(raised.value, SENTINEL)


# --- D9 drift: every ACL the principal holds is read, whatever its shape ---------


def _acl(
    operation: AclOperation = AclOperation.WRITE,
    *,
    permission: AclPermissionType = AclPermissionType.ALLOW,
    host: str = "*",
    restype: ResourceType = ResourceType.TOPIC,
    name: str = "onex.cmd.a.v1",
) -> AclBinding:
    return AclBinding(
        restype,
        name,
        ResourcePatternType.LITERAL,
        f"User:{PRINCIPAL}",
        host,
        operation,
        permission,
    )


class _DescribingAdminClient:
    """``describe_acls`` answers the filter as the broker does, permission included."""

    def __init__(self, held: list[AclBinding]) -> None:
        self._held = held
        self.filters: list[AclBindingFilter] = []

    def describe_acls(self, acl_filter: AclBindingFilter, **_: Any) -> Future[object]:
        self.filters.append(acl_filter)
        wanted = acl_filter.permission_type
        future: Future[object] = Future()
        future.set_result(
            [
                acl
                for acl in self._held
                if wanted == AclPermissionType.ANY or acl.permission_type == wanted
            ]
        )
        return future


PLAIN = ModelBrokerAclBinding(
    resource="topic", name="onex.cmd.a.v1", pattern="literal", operation="write"
)


@pytest.mark.parametrize(
    ("held", "differs_in"),
    [
        (_acl(permission=AclPermissionType.DENY), "permission"),
        (_acl(host="192.0.2.10"), "host"),
        (_acl(AclOperation.ALTER), "operation"),
        (_acl(restype=ResourceType.TRANSACTIONAL_ID), "resource"),
    ],
    ids=["deny", "host_scoped_allow", "unsupported_operation", "transactional_id"],
)
def test_drift_list_acls_reads_every_acl_and_keeps_what_sets_it_apart(
    held: AclBinding, differs_in: str
) -> None:
    admin = _DescribingAdminClient([_acl(), held])
    found = BrokerPrincipalKafkaAcls(admin_client=admin).list_acls(PRINCIPAL)
    assert PLAIN in found, "the plain planned binding is still read"
    assert len(found) == 2, f"the {differs_in} ACL was dropped or merged: {found}"
    (other,) = found - {PLAIN}
    assert other.label() != PLAIN.label(), "the refusal could not name it"
    assert differs_in in {
        field
        for field in ("resource", "operation", "permission", "host")
        if getattr(other, field) != getattr(PLAIN, field)
    }


def test_drift_list_acls_asks_the_broker_for_every_permission_type() -> None:
    admin = _DescribingAdminClient([_acl()])
    BrokerPrincipalKafkaAcls(admin_client=admin).list_acls(PRINCIPAL)
    (acl_filter,) = admin.filters
    assert acl_filter.permission_type == AclPermissionType.ANY
