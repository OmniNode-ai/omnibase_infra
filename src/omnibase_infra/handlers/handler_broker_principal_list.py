# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerBrokerPrincipalList and the broker clients shared by the principal effect.

``HandlerBrokerPrincipalList`` lists the broker's developer principals under a
prefix. It is the Admin API reachability check of the mint (plan D9): an
unreachable Admin API stops the run here, before any mutation.

This module also holds what the three handlers of ``node_broker_principal_effect``
share: the typed failure ``BrokerPrincipalError`` and the broker clients.

* ``BrokerPrincipalAdminApi`` lists SCRAM users and sets one user's password through
  the Redpanda Admin API (``GET``/``POST /v1/security/users``,
  ``PUT /v1/security/users/<name>``) as the stack superuser with HTTP Basic auth,
  the calls ``scripts/lab_principal_issuer.py`` makes on the lab.
* ``BrokerPrincipalKafkaAcls`` reads and creates a principal's ALLOW ACLs through a
  confluent admin client. ``create_acls`` returns one future per binding and they
  complete independently, so it waits for every one and raises, naming how many
  failed, unless all succeeded.
* ``BrokerPrincipalKafkaSession`` logs in as the principal and lists its topics.

No client and no error carries a password: a failure is re-raised with the
password values scrubbed from its text, through ``attempt``, which leaves the new
error with no cause and no context, so the original is not reachable from it.

Canonical def-B dispatch entrypoint: ``handle(request) -> ModelBrokerPrincipalListResult``.
Ticket: OMN-20822
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import CancelledError
from enum import Enum
from typing import Final, cast

from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient

# The package re-exports these without ``__all__``, which ``mypy --strict`` does
# not count as an export; import them from the modules that define them.
from confluent_kafka.admin._acl import (
    AclBinding,
    AclBindingFilter,
    AclOperation,
    AclPermissionType,
)
from confluent_kafka.admin._resource import ResourcePatternType, ResourceType
from pydantic import SecretStr

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.event_bus.kafka_auth import build_confluent_auth_config_from_env
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.developer_principal.model_broker_acl_binding import (
    ModelBrokerAclBinding,
)
from omnibase_infra.models.developer_principal.model_broker_principal_list_request import (
    ModelBrokerPrincipalListRequest,
)
from omnibase_infra.models.developer_principal.model_broker_principal_list_result import (
    ModelBrokerPrincipalListResult,
)
from omnibase_infra.protocols.protocol_broker_principal_admin import (
    ProtocolBrokerPrincipalAdmin,
)
from omnibase_infra.protocols.protocol_broker_principal_kafka_admin import (
    ProtocolBrokerPrincipalKafkaAdmin,
)
from omnibase_infra.protocols.protocol_broker_principal_topic_lister import (
    ProtocolBrokerPrincipalTopicLister,
)

OPERATION = "broker.principal.list"
SCRAM_ALGORITHM: Final[str] = "SCRAM-SHA-256"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 15.0

# What an Admin API, Kafka or login client raises: network and OS errors (URLError,
# ConnectionError and PermissionError are OSError), and the RuntimeError or
# ValueError the clients raise for a broker refusal or an unreadable reply.
CLIENT_ERRORS: tuple[type[Exception], ...] = (OSError, RuntimeError, ValueError)

# What one create_acls future raises: the broker's refusal, a timeout, a
# cancellation, or the RuntimeError a client raises for a failed binding.
_FUTURE_ERRORS: Final[tuple[type[BaseException], ...]] = (
    KafkaException,
    RuntimeError,
    TimeoutError,
    CancelledError,
)

_RESOURCE = {"topic": ResourceType.TOPIC, "group": ResourceType.GROUP}
_PATTERN = {
    "literal": ResourcePatternType.LITERAL,
    "prefixed": ResourcePatternType.PREFIXED,
}
_OPERATION = {
    "describe": AclOperation.DESCRIBE,
    "read": AclOperation.READ,
    "write": AclOperation.WRITE,
}

# confluent_kafka treats a None name or host in an AclBindingFilter as "match
# any"; its type stubs declare str, so the wildcard is cast rather than ignored.
_ANY: Final = cast("str", None)

# The Admin API URL is operator configuration; the opener and request class are
# held as values so a test can replace the transport whole.
_REQUEST: Final = urllib.request.Request
_URLOPEN: Final = urllib.request.urlopen

Transport = Callable[[str, str, bytes | None, dict[str, str]], bytes]


def attempt[T](
    call: Callable[[], T],
    failure: Callable[[Exception], Exception],
    errors: tuple[type[Exception], ...] = CLIENT_ERRORS,
) -> T:
    """Run ``call``; on one of ``errors`` raise ``failure(exc)`` with no chain.

    ``raise ... from None`` only hides a chain from the traceback: the new error's
    ``__context__`` still holds the original, whose text or arguments can carry a
    password. The failure is therefore built inside the handler and raised after
    it, where no exception is being handled, so its cause and context are None.
    """
    try:
        return call()
    except errors as exc:
        error = failure(exc)
    raise error


class BrokerPrincipalError(RuntimeHostError):
    """Listing, applying or verifying a developer principal failed.

    The mint spans two APIs and promises no cross-API atomicity, so the error
    states which mutations had completed: ``acls_applied`` (every planned ACL
    created and read back) and ``password_set``. It names the operation and the
    Admin API URL. Its text never carries the password: callers pass scrubbed
    text and raise it through ``attempt``.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        url: str,
        acls_applied: bool,
        password_set: bool,
    ) -> None:
        self.operation = operation
        self.url = url
        self.acls_applied = acls_applied
        self.password_set = password_set
        super().__init__(
            message=f"{operation} against {url}: {message} "
            f"(acls_applied={acls_applied}, password_set={password_set})",
            error_code=EnumCoreErrorCode.OPERATION_FAILED,
            operation_name=operation,
            admin_url=url,
            acls_applied=acls_applied,
            password_set=password_set,
        )


def scrub(text: str, *secrets: SecretStr | None) -> str:
    """Remove every secret's value from text a client produced."""
    for secret in secrets:
        if secret is None:
            continue
        value = secret.get_secret_value()
        if value:
            text = text.replace(value, "***")
    return text


def _urllib_transport(
    method: str, url: str, body: bytes | None, headers: dict[str, str]
) -> bytes:
    request = _REQUEST(url, data=body, method=method, headers=headers)
    with _URLOPEN(request, timeout=DEFAULT_TIMEOUT_SECONDS) as reply:
        payload: bytes = reply.read()
    return payload


class BrokerPrincipalAdminApi:
    """The Redpanda Admin API's SCRAM user calls, as the stack superuser."""

    def __init__(
        self,
        *,
        admin_url: str,
        username: str,
        password: SecretStr,
        transport: Transport | None = None,
    ) -> None:
        self._base = admin_url.rstrip("/")
        self._password = password
        token = base64.b64encode(
            f"{username}:{password.get_secret_value()}".encode()
        ).decode()
        self._authorization = f"Basic {token}"
        self._transport = transport or _urllib_transport

    def _call(
        self,
        method: str,
        path: str,
        body: Mapping[str, str] | None,
        hidden: SecretStr | None = None,
    ) -> object:
        data = json.dumps(dict(body)).encode() if body is not None else None
        headers = {"Authorization": self._authorization}
        if data is not None:
            headers["Content-Type"] = "application/json"
        payload = attempt(
            lambda: self._transport(method, f"{self._base}{path}", data, headers),
            lambda exc: OSError(
                f"Admin API {method} {path} failed: "
                + scrub(str(exc), self._password, hidden)
            ),
            (OSError,),
        )
        return json.loads(payload) if payload else None

    def list_users(self) -> tuple[str, ...]:
        users = self._call("GET", "/v1/security/users", None)
        if not isinstance(users, list) or not all(isinstance(u, str) for u in users):
            raise ValueError(
                "Admin API /v1/security/users did not return a list of names"
            )
        return tuple(users)

    def set_password(self, name: str, password: SecretStr, *, exists: bool) -> None:
        body = {
            "username": name,
            "password": password.get_secret_value(),
            "algorithm": SCRAM_ALGORITHM,
        }
        if exists:
            quoted = urllib.parse.quote(name, safe="")
            self._call("PUT", f"/v1/security/users/{quoted}", body, password)
        else:
            self._call("POST", "/v1/security/users", body, password)


def _enum_name(value: object) -> str:
    """A confluent enum member as the binding names it: ``AclOperation.ALTER`` -> ``alter``."""
    return value.name.lower() if isinstance(value, Enum) else str(value).lower()


def _to_binding(principal: str, binding: ModelBrokerAclBinding) -> AclBinding:
    return AclBinding(
        _RESOURCE[binding.resource],
        binding.name,
        _PATTERN[binding.pattern],
        f"User:{principal}",
        "*",
        _OPERATION[binding.operation],
        AclPermissionType.ALLOW,
    )


class BrokerPrincipalKafkaAcls:
    """A principal's ALLOW ACLs through a confluent admin client."""

    def __init__(
        self,
        *,
        admin_client: ProtocolBrokerPrincipalKafkaAdmin,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._admin = admin_client
        self._timeout = timeout_seconds

    def list_acls(self, principal: str) -> frozenset[ModelBrokerAclBinding]:
        """Every ACL naming the principal, of any resource, operation, permission or host."""
        acl_filter = AclBindingFilter(
            ResourceType.ANY,
            _ANY,
            ResourcePatternType.ANY,
            f"User:{principal}",
            _ANY,
            AclOperation.ANY,
            AclPermissionType.ANY,
        )
        found = attempt(
            lambda: self._admin.describe_acls(acl_filter).result(timeout=self._timeout),
            lambda exc: RuntimeError(f"describe_acls failed: {exc}"),
            (KafkaException,),
        )
        if not isinstance(found, list):
            raise RuntimeError("describe_acls did not return a list of bindings")
        return frozenset(
            ModelBrokerAclBinding(
                resource=_enum_name(acl.restype),
                name=acl.name,
                pattern=_enum_name(acl.resource_pattern_type),
                operation=_enum_name(acl.operation),
                permission=_enum_name(acl.permission_type),
                host=acl.host or "*",
            )
            for acl in found
        )

    def create_acls(
        self, principal: str, bindings: Iterable[ModelBrokerAclBinding]
    ) -> None:
        wanted = [_to_binding(principal, b) for b in bindings]
        if not wanted:
            return
        futures = self._admin.create_acls(wanted)
        failed: list[str] = []
        for binding, future in futures.items():
            try:
                future.result(timeout=self._timeout)
            except _FUTURE_ERRORS as exc:
                failed.append(f"{binding.name}: {type(exc).__name__}: {exc}")
        if failed:
            raise RuntimeError(
                f"{len(failed)} of {len(wanted)} ACL bindings were not created: "
                + "; ".join(sorted(failed))
            )


def _confluent_admin(config: dict[str, str]) -> ProtocolBrokerPrincipalTopicLister:
    return AdminClient({**config})


class BrokerPrincipalKafkaSession:
    """Log in to the broker as a principal and list the topics it can see."""

    def __init__(
        self,
        *,
        base_config: Mapping[str, str],
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client_factory: Callable[
            [dict[str, str]], ProtocolBrokerPrincipalTopicLister
        ] = _confluent_admin,
    ) -> None:
        self._base = dict(base_config)
        self._timeout = timeout_seconds
        self._factory = client_factory

    def list_topics_as(self, principal: str, password: SecretStr) -> tuple[str, ...]:
        config = {
            **self._base,
            "sasl.username": principal,
            "sasl.password": password.get_secret_value(),
        }
        metadata = attempt(
            lambda: self._factory(config).list_topics(timeout=self._timeout),
            lambda exc: PermissionError(
                "login as the principal failed: " + scrub(str(exc), password)
            ),
            (KafkaException,),
        )
        topics = getattr(metadata, "topics", None)
        if not isinstance(topics, Mapping):
            raise ValueError("the broker's metadata reply carried no topic map")
        return tuple(sorted(str(name) for name in topics))


def admin_api_from_env(
    admin_url: str, environ: Mapping[str, str] | None = None
) -> ProtocolBrokerPrincipalAdmin:
    """The Admin API client as the stack superuser the runtime's Kafka env names."""
    env = os.environ if environ is None else environ
    username = env.get("KAFKA_SASL_USERNAME", "")
    password = env.get("KAFKA_SASL_PASSWORD", "")
    if not username or not password:
        raise ValueError(
            "KAFKA_SASL_USERNAME and KAFKA_SASL_PASSWORD must name the stack superuser"
        )
    return BrokerPrincipalAdminApi(
        admin_url=admin_url, username=username, password=SecretStr(password)
    )


def kafka_acls_from_env() -> BrokerPrincipalKafkaAcls:
    """An ACL client on the runtime's broker, as the runtime's Kafka login."""
    admin = AdminClient(
        {
            "bootstrap.servers": ModelKafkaEventBusConfig.default().bootstrap_servers,
            **build_confluent_auth_config_from_env(),
        }
    )
    return BrokerPrincipalKafkaAcls(admin_client=admin)


def kafka_session_from_env() -> BrokerPrincipalKafkaSession:
    """A login client on the runtime's broker with its transport settings."""
    auth = build_confluent_auth_config_from_env()
    base = {
        "bootstrap.servers": ModelKafkaEventBusConfig.default().bootstrap_servers,
        **{
            k: v for k, v in auth.items() if k not in ("sasl.username", "sasl.password")
        },
    }
    return BrokerPrincipalKafkaSession(base_config=base)


class HandlerBrokerPrincipalList:
    """Effect handler: list the principals that carry the declaration's prefix."""

    def __init__(self, admin: ProtocolBrokerPrincipalAdmin | None = None) -> None:
        # None under ``onex node``: the Admin API client is built per request from
        # the request's URL and the stack superuser in the container environment.
        self._admin = admin

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelBrokerPrincipalListRequest
    ) -> ModelBrokerPrincipalListResult:
        """List the broker's users and keep those whose name starts with the prefix."""
        users = attempt(
            lambda: (self._admin or admin_api_from_env(request.admin_url)).list_users(),
            lambda exc: BrokerPrincipalError(
                f"the Admin API is unreachable ({type(exc).__name__})",
                operation=OPERATION,
                url=request.admin_url,
                acls_applied=False,
                password_set=False,
            ),
        )
        return ModelBrokerPrincipalListResult(
            principals=tuple(sorted(u for u in users if u.startswith(request.prefix)))
        )
