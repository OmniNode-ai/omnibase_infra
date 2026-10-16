# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One answer to "is this Kafka failure permanent" (OMN-19092).

Before this module the answer was given in four places that could not see each
other: a publish-side ``except`` branch in ``EventBusKafka``, a connect-side
tuple in ``kafka_connect_retry``, the gateway forwarder's egress tuple, and a
cause-chain predicate in a layer ABOVE all three. These tests pin the shared
declaration, and then pin that each consumer reads it rather than keeping a
list of its own.

AC1 is the declaration (both a connect-time authentication rejection and a
publish-time authorization refusal, no widening by a caller). AC2 is
type-and-chain, never message text. AC3 is that each caller's handling is
unchanged. AC4 is the positive control: the transient classes still classify as
"not permanent", which is what keeps a shared predicate from swallowing the
retry feature.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from aiokafka.errors import (
    AuthenticationFailedError,
    AuthenticationMethodNotSupported,
    ClusterAuthorizationFailedError,
    IllegalSaslStateError,
    KafkaConnectionError,
    KafkaError,
    KafkaTimeoutError,
    MessageSizeTooLargeError,
    NodeNotReadyError,
    TopicAuthorizationFailedError,
    UnknownTopicOrPartitionError,
    UnsupportedSaslMechanismError,
    UnsupportedVersionError,
)

from omnibase_infra.enums.enum_kafka_permanent_failure import (
    EnumKafkaPermanentFailure,
)
from omnibase_infra.handlers import kafka_permanent_failure
from omnibase_infra.handlers.kafka_permanent_failure import (
    classify_permanent_kafka_failure,
    kafka_failure_types,
)

pytestmark = pytest.mark.unit

_SRC = Path(inspect.getfile(kafka_permanent_failure)).resolve().parents[1]


class TestOneDeclarationExpressesBothQuestions:
    """AC1: connect-time authentication AND publish-time authorization."""

    @pytest.mark.parametrize(
        "exc_type",
        [
            AuthenticationFailedError,
            AuthenticationMethodNotSupported,
            IllegalSaslStateError,
            UnsupportedSaslMechanismError,
            UnsupportedVersionError,
        ],
    )
    def test_a_connect_authentication_rejection_is_permanent(
        self, exc_type: type[KafkaError]
    ) -> None:
        assert (
            classify_permanent_kafka_failure(exc_type())
            is EnumKafkaPermanentFailure.CONNECT_AUTHENTICATION_REJECTED
        )

    def test_a_topic_authorization_refusal_is_permanent(self) -> None:
        assert (
            classify_permanent_kafka_failure(TopicAuthorizationFailedError())
            is EnumKafkaPermanentFailure.TOPIC_AUTHORIZATION_REFUSED
        )

    def test_a_cluster_authorization_refusal_is_permanent(self) -> None:
        assert (
            classify_permanent_kafka_failure(ClusterAuthorizationFailedError())
            is EnumKafkaPermanentFailure.CLUSTER_AUTHORIZATION_REFUSED
        )

    def test_the_types_for_an_except_clause_come_from_the_same_declaration(
        self,
    ) -> None:
        """``except`` cannot call a predicate, so it reads the same table."""
        assert kafka_failure_types(
            EnumKafkaPermanentFailure.TOPIC_AUTHORIZATION_REFUSED
        ) == (TopicAuthorizationFailedError,)
        assert set(
            kafka_failure_types(
                EnumKafkaPermanentFailure.CONNECT_AUTHENTICATION_REJECTED
            )
        ) == {
            AuthenticationFailedError,
            AuthenticationMethodNotSupported,
            IllegalSaslStateError,
            UnsupportedSaslMechanismError,
            UnsupportedVersionError,
        }

    def test_every_kind_has_at_least_one_declared_type(self) -> None:
        for kind in EnumKafkaPermanentFailure:
            assert kafka_failure_types(kind), f"{kind} declares no exception type"


class TestKeyedOnTypeAndChainNeverOnText:
    """AC2."""

    def test_a_wrapped_refusal_is_found_through_cause(self) -> None:
        try:
            try:
                raise TopicAuthorizationFailedError
            except TopicAuthorizationFailedError as inner:
                raise RuntimeError("transport wrapper") from inner
        except RuntimeError as wrapper:
            assert (
                classify_permanent_kafka_failure(wrapper)
                is EnumKafkaPermanentFailure.TOPIC_AUTHORIZATION_REFUSED
            )

    def test_a_wrapped_refusal_is_found_through_context(self) -> None:
        try:
            try:
                raise AuthenticationFailedError
            except AuthenticationFailedError:
                raise RuntimeError("raised while handling") from None
        except RuntimeError as wrapper:
            # ``from None`` suppresses display but ``__context__`` is still set.
            assert (
                classify_permanent_kafka_failure(wrapper)
                is EnumKafkaPermanentFailure.CONNECT_AUTHENTICATION_REJECTED
            )

    def test_a_cyclic_chain_terminates(self) -> None:
        first = RuntimeError("first")
        second = RuntimeError("second")
        first.__cause__ = second
        second.__cause__ = first
        assert classify_permanent_kafka_failure(first) is None

    def test_a_message_that_names_a_permanent_class_is_not_permanent(self) -> None:
        """The falsifier for a string match: text says it, the type does not."""
        liar = RuntimeError("TopicAuthorizationFailedError: AuthenticationFailedError")
        assert classify_permanent_kafka_failure(liar) is None

    def test_the_module_never_reads_an_exception_message(self) -> None:
        """No ``str(exc)``, ``exc.args`` or ``.message`` in the classifier."""
        tree = ast.parse(Path(inspect.getfile(kafka_permanent_failure)).read_text())
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"args", "message"}:
                offenders.append(f"line {node.lineno}: .{node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"str", "repr"}
            ):
                offenders.append(f"line {node.lineno}: {node.func.id}(...)")
        assert not offenders, offenders


class TestTransientFailuresStayTransient:
    """AC4: the positive control that keeps the predicate narrow."""

    @pytest.mark.parametrize(
        "exc",
        [
            KafkaConnectionError("connection refused"),
            KafkaTimeoutError("timed out"),
            NodeNotReadyError("node not ready"),
            TimeoutError("deadline"),
            ConnectionResetError("reset"),
            RuntimeError("anything else"),
            # The sibling fail-fast branches are configuration/payload verdicts,
            # not authentication or authorization ones, and keep their own
            # branches in EventBusKafka.
            UnknownTopicOrPartitionError(),
            MessageSizeTooLargeError(),
        ],
    )
    def test_a_transient_failure_is_not_permanent(self, exc: BaseException) -> None:
        assert classify_permanent_kafka_failure(exc) is None


class TestConsumersReadTheSharedDeclaration:
    """AC1's falsifier: a caller still keeps a private list beside the shared one."""

    _PERMANENT_CLASS_NAMES = frozenset(
        {
            "AuthenticationFailedError",
            "AuthenticationMethodNotSupported",
            "IllegalSaslStateError",
            "UnsupportedSaslMechanismError",
            "UnsupportedVersionError",
            "TopicAuthorizationFailedError",
            "ClusterAuthorizationFailedError",
        }
    )
    _CONSUMERS = (
        "event_bus/kafka_connect_retry.py",
        "event_bus/event_bus_kafka.py",
        "runtime/gateway_forwarder.py",
    )

    @pytest.mark.parametrize("relative", _CONSUMERS)
    def test_the_consumer_names_no_permanent_class_itself(self, relative: str) -> None:
        tree = ast.parse((_SRC / relative).read_text())
        named: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in self._PERMANENT_CLASS_NAMES:
                named.append(f"line {node.lineno}: {node.id}")
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "aiokafka.errors"
                and any(
                    alias.name in self._PERMANENT_CLASS_NAMES for alias in node.names
                )
            ):
                named.append(f"line {node.lineno}: imports from aiokafka.errors")
        assert not named, (
            f"{relative} keeps its own list of permanent classes instead of "
            f"reading kafka_permanent_failure: {named}"
        )


class TestEachCallerKeepsItsOwnHandling:
    """AC3 and AC4 at the connect caller: same decisions, now chain-aware."""

    @pytest.fixture(autouse=True)
    def _collapse_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the retry SHAPE and drop the wall clock (the declared dial)."""
        monkeypatch.setenv("KAFKA_RETRY_BACKOFF_BASE", "0.001")

    @pytest.mark.asyncio
    async def test_a_wrapped_authentication_rejection_raises_on_the_first_attempt(
        self,
    ) -> None:
        from unittest.mock import AsyncMock

        from omnibase_infra.event_bus.kafka_connect_retry import (
            connect_with_bounded_retry,
        )
        from omnibase_infra.event_bus.models.config.model_kafka_connect_retry_policy import (
            ModelKafkaConnectRetryPolicy,
        )

        calls = 0

        async def _connect() -> None:
            nonlocal calls
            calls += 1
            try:
                raise AuthenticationFailedError
            except AuthenticationFailedError as inner:
                raise RuntimeError("wrapped by a transport") from inner

        with pytest.raises(RuntimeError, match="wrapped by a transport"):
            await connect_with_bounded_retry(
                policy=ModelKafkaConnectRetryPolicy(attempt_timeout_seconds=1.0),
                connect=_connect,
                cleanup=AsyncMock(),
                target="broker:19092",
            )

        assert calls == 1

    @pytest.mark.asyncio
    async def test_connect_does_not_act_on_a_publish_time_verdict(self) -> None:
        """Connect raises at once on authentication only; a topic refusal that
        somehow surfaces during connect keeps its retry, exactly as before."""
        from unittest.mock import AsyncMock

        from omnibase_infra.event_bus.kafka_connect_retry import (
            connect_with_bounded_retry,
        )
        from omnibase_infra.event_bus.models.config.model_kafka_connect_retry_policy import (
            ModelKafkaConnectRetryPolicy,
        )

        calls = 0

        async def _connect() -> None:
            nonlocal calls
            calls += 1
            raise TopicAuthorizationFailedError

        policy = ModelKafkaConnectRetryPolicy(attempt_timeout_seconds=1.0)
        with pytest.raises(TopicAuthorizationFailedError):
            await connect_with_bounded_retry(
                policy=policy,
                connect=_connect,
                cleanup=AsyncMock(),
                target="broker:19092",
            )

        assert calls == policy.total_attempts
