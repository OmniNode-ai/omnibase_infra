# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20822: the broker principal effect against the auth-required Redpanda.

The apply and verify handlers run with their real clients: the Redpanda Admin API
for SCRAM users and a confluent admin client, as the harness superuser, for ACLs.
A rerun rotates and leaves the ACL set identical (AC2), a principal holding an ACL
outside the plan is refused with the password untouched (AC2), and a principal
granted two of three provisioned topics sees and writes exactly those two (AC4).

The harness publishes only the Kafka listener, not the Admin API port. The Admin
API calls therefore run inside the broker container through ``docker exec ...
curl`` (the container admin ``rpk`` is exec'd into, for a started and for a
declared broker alike), and the request body, which carries the principal's
password, reaches curl on stdin rather than in its argv.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from collections.abc import Callable

import pytest
from confluent_kafka import KafkaError, Producer
from confluent_kafka.admin import AdminClient
from pydantic import SecretStr

from omnibase_infra.handlers.handler_broker_principal_apply import (
    HandlerBrokerPrincipalApply,
    planned_bindings,
)
from omnibase_infra.handlers.handler_broker_principal_list import (
    BrokerPrincipalAdminApi,
    BrokerPrincipalKafkaAcls,
    BrokerPrincipalKafkaSession,
)
from omnibase_infra.handlers.handler_broker_principal_verify import (
    HandlerBrokerPrincipalVerify,
)
from omnibase_infra.models.developer_principal import (
    ModelBrokerPrincipalApplyRequest,
    ModelBrokerPrincipalApplyResult,
    ModelBrokerPrincipalVerifyRequest,
    ModelDeveloperPrincipalGrant,
    ModelDeveloperPrincipalPlan,
)

from .redpanda_sasl_harness import SECURITY_PROTOCOL, RedpandaSasl, RuntimePrincipal

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
    pytest.mark.slow,
    pytest.mark.timeout(600),
    pytest.mark.xdist_group("omn18012_redpanda_sasl"),
]

# The Admin API as the broker container itself reaches it.
ADMIN_URL = "http://localhost:9644"
PASSWORD_ENV = "ONEX_MINT_PASSWORD"


def _docker_transport(
    broker: RedpandaSasl,
) -> Callable[[str, str, bytes | None, dict[str, str]], bytes]:
    def _send(
        method: str, url: str, body: bytes | None, headers: dict[str, str]
    ) -> bytes:
        args = ["docker", "exec", "-i", broker.container, "curl", "-sS", "-f"]
        args += ["-X", method]
        for name, value in headers.items():
            args += ["-H", f"{name}: {value}"]
        if body is not None:
            args += ["--data-binary", "@-"]
        args.append(url)
        proc = subprocess.run(
            args, input=body or b"", capture_output=True, timeout=60, check=False
        )
        if proc.returncode != 0:
            raise OSError(
                f"curl rc={proc.returncode}: {proc.stderr.decode(errors='replace')}"
            )
        return proc.stdout

    return _send


def _client_config(
    broker: RedpandaSasl, username: str, password: str
) -> dict[str, str | int | float | bool]:
    return {
        "bootstrap.servers": broker.bootstrap,
        "security.protocol": SECURITY_PROTOCOL,
        "sasl.mechanism": broker.mechanism,
        "sasl.username": username,
        "sasl.password": password,
    }


def _acls(broker: RedpandaSasl) -> BrokerPrincipalKafkaAcls:
    admin = AdminClient(_client_config(broker, broker.username, broker.password))
    return BrokerPrincipalKafkaAcls(admin_client=admin, timeout_seconds=30.0)


def _session(broker: RedpandaSasl) -> BrokerPrincipalKafkaSession:
    base = {
        "bootstrap.servers": broker.bootstrap,
        "security.protocol": SECURITY_PROTOCOL,
        "sasl.mechanism": broker.mechanism,
    }
    return BrokerPrincipalKafkaSession(base_config=base, timeout_seconds=15.0)


def _apply(
    broker: RedpandaSasl, plan: ModelDeveloperPrincipalPlan, password: str
) -> ModelBrokerPrincipalApplyResult:
    admin = BrokerPrincipalAdminApi(
        admin_url=ADMIN_URL,
        username=broker.username,
        password=SecretStr(broker.password),
        transport=_docker_transport(broker),
    )
    handler = HandlerBrokerPrincipalApply(
        admin=admin, acls=_acls(broker), environ={PASSWORD_ENV: password}
    )
    return handler.handle(
        ModelBrokerPrincipalApplyRequest(
            plan=plan, admin_url=ADMIN_URL, password_env=PASSWORD_ENV
        )
    )


def _plan(principal: str, granted: tuple[str, ...]) -> ModelDeveloperPrincipalPlan:
    return ModelDeveloperPrincipalPlan(
        decision="create",
        principal=principal,
        lane="local",
        grants=tuple(
            ModelDeveloperPrincipalGrant(
                resource="topic",
                name=topic,
                pattern="literal",
                operations=("describe", "write"),
            )
            for topic in sorted(granted)
        ),
        declaration_digest="c" * 64,
    )


def _topics(broker: RedpandaSasl, count: int) -> tuple[str, ...]:
    run = uuid.uuid4().hex[:8]
    names = tuple(f"onex.evt.omn20822-{run}.t{i}.v1" for i in range(count))
    for name in names:
        broker.create_topic(name)
    return names


def _principal() -> str:
    return f"dev-omn20822-{uuid.uuid4().hex[:8]}"


def _can_login(broker: RedpandaSasl, principal: str, password: str) -> bool:
    try:
        _session(broker).list_topics_as(principal, SecretStr(password))
    except PermissionError:
        return False
    return True


def _logs_in_within(
    broker: RedpandaSasl, principal: str, password: str, seconds: float = 20.0
) -> bool:
    """A new SCRAM credential can take a moment to reach the broker's listener."""
    deadline = time.monotonic() + seconds
    while True:
        if _can_login(broker, principal, password):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(1.0)


def _produce(
    broker: RedpandaSasl, username: str, password: str, topic: str
) -> KafkaError | None:
    """Produce one record and return the delivery error, or None when delivered."""
    producer = Producer(
        {
            **_client_config(broker, username, password),
            "message.timeout.ms": "20000",
        }
    )
    errors: list[KafkaError | None] = []
    producer.produce(
        topic, b"omn20822", on_delivery=lambda err, _msg: errors.append(err)
    )
    producer.flush(30)
    assert errors, f"no delivery report for {topic} within 30 s"
    return errors[0]


# --- AC2: a rerun rotates, drift is refused --------------------------------------


def test_rerun_rotates_and_leaves_the_acl_set_identical(
    redpanda_sasl: RedpandaSasl,
) -> None:
    broker = redpanda_sasl
    principal = _principal()
    plan = _plan(principal, _topics(broker, 2))
    acls = _acls(broker)

    first = _apply(broker, plan, "first-password-omn20822")
    assert (first.action, first.acls_created) == ("create", 4)
    held_after_first = acls.list_acls(principal)
    assert held_after_first == planned_bindings(plan)

    second = _apply(broker, plan, "second-password-omn20822")
    assert (second.action, second.acls_created) == ("rotate", 0)
    assert (second.acls_applied, second.password_set) == (True, True)
    assert acls.list_acls(principal) == held_after_first
    assert _logs_in_within(broker, principal, "second-password-omn20822")


def test_drift_an_acl_outside_the_plan_is_refused_and_the_password_untouched(
    redpanda_sasl: RedpandaSasl,
) -> None:
    broker = redpanda_sasl
    principal = _principal()
    granted_a, granted_b, outside = _topics(broker, 3)
    plan = _plan(principal, (granted_a, granted_b))

    assert _apply(broker, plan, "old-password-omn20822").action == "create"
    assert _logs_in_within(broker, principal, "old-password-omn20822")
    broker.grant(
        RuntimePrincipal(principal, "old-password-omn20822"),
        operations=("read",),
        topics=(outside,),
    )
    held_before = _acls(broker).list_acls(principal)

    result = _apply(broker, plan, "new-password-omn20822")
    assert result.action == "refused"
    assert [b.name for b in result.extra_acls] == [outside]
    assert outside in result.refusal
    assert (result.acls_applied, result.password_set) == (False, False)
    assert _acls(broker).list_acls(principal) == held_before
    assert _can_login(broker, principal, "old-password-omn20822")
    assert not _can_login(broker, principal, "new-password-omn20822")


# --- AC4: the broker enforces exactly the planned grants -------------------------


def test_enforcement_the_principal_sees_and_writes_only_its_two_granted_topics(
    redpanda_sasl: RedpandaSasl,
) -> None:
    broker = redpanda_sasl
    principal = _principal()
    password = "enforcement-password-omn20822"
    granted_a, granted_b, ungranted = _topics(broker, 3)
    plan = _plan(principal, (granted_a, granted_b))

    assert _apply(broker, plan, password).action == "create"
    assert _logs_in_within(broker, principal, password)

    verified = HandlerBrokerPrincipalVerify(
        acls=_acls(broker), session=_session(broker), environ={PASSWORD_ENV: password}
    ).handle(
        ModelBrokerPrincipalVerifyRequest(
            plan=plan, admin_url=ADMIN_URL, password_env=PASSWORD_ENV
        )
    )
    assert verified.verified is True, verified.reason
    assert verified.visible_topics == tuple(sorted((granted_a, granted_b)))

    assert _produce(broker, principal, password, granted_a) is None
    # Positive control: the third topic exists and accepts a write from a
    # principal allowed to make it, so the refusal below is authorization.
    assert _produce(broker, broker.username, broker.password, ungranted) is None
    refused = _produce(broker, principal, password, ungranted)
    assert refused is not None, "a produce to the ungranted topic was delivered"
    assert refused.name() == "TOPIC_AUTHORIZATION_FAILED", refused
