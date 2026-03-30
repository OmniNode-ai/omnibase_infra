# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for handler execution verification probe [OMN-7040]."""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
)
from omnibase_infra.verification.probes.probe_handler_execution import (
    VERIFY_CORRELATION_PREFIX,
    check_handler_execution,
)


def _make_contract(
    *,
    name: str = "node_registration_orchestrator",
    subscribe_topics: tuple[str, ...] = ("onex.evt.registration.requested.v1",),
    publish_topics: tuple[str, ...] = ("onex.evt.registration.completed.v1",),
) -> ModelParsedContractForVerification:
    """Build a minimal parsed contract for testing."""
    return ModelParsedContractForVerification(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        subscribe_topics=subscribe_topics,
        publish_topics=publish_topics,
        handler_names=("handle_registration",),
    )


@pytest.mark.unit
class TestCheckHandlerExecution:
    """Tests for check_handler_execution probe."""

    def test_pass_when_correlated_response_received(self) -> None:
        """Handler produces a matching correlation_id response."""
        published: list[tuple[str, dict[str, str]]] = []

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            published.append((topic, payload))

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            # Return a message with the same correlation_id that was published
            assert len(published) == 1
            return [{"correlation_id": published[0][1]["correlation_id"]}]

        contract = _make_contract()
        result = check_handler_execution(contract, mock_publish, mock_consume)

        assert result.verdict == EnumValidationVerdict.PASS
        assert result.check_type == EnumContractCheckType.HANDLER_EXECUTION
        assert "response received" in result.message.lower()

    def test_fail_when_no_response_within_timeout(self) -> None:
        """No matching message on publish topic."""

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            pass

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            return []

        contract = _make_contract()
        result = check_handler_execution(
            contract, mock_publish, mock_consume, timeout_seconds=0.1
        )

        assert result.verdict == EnumValidationVerdict.FAIL
        assert "no response" in result.message.lower()

    def test_fail_when_wrong_correlation_id(self) -> None:
        """Messages exist but none match the verify correlation_id."""

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            pass

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            return [
                {"correlation_id": "unrelated-1234"},
                {"correlation_id": "other-5678"},
            ]

        contract = _make_contract()
        result = check_handler_execution(contract, mock_publish, mock_consume)

        assert result.verdict == EnumValidationVerdict.FAIL

    def test_fail_when_no_subscribe_topics(self) -> None:
        """Contract declares no subscribe topics."""

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            pass

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            return []

        contract = _make_contract(subscribe_topics=())
        result = check_handler_execution(contract, mock_publish, mock_consume)

        assert result.verdict == EnumValidationVerdict.FAIL
        assert "no subscribe" in result.message.lower()

    def test_fail_when_no_publish_topics(self) -> None:
        """Contract declares no publish topics."""

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            pass

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            return []

        contract = _make_contract(publish_topics=())
        result = check_handler_execution(contract, mock_publish, mock_consume)

        assert result.verdict == EnumValidationVerdict.FAIL
        assert "no publish" in result.message.lower()

    def test_fail_when_publish_raises(self) -> None:
        """Publish function raises an exception."""

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            raise ConnectionError("Kafka unavailable")

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            return []

        contract = _make_contract()
        result = check_handler_execution(contract, mock_publish, mock_consume)

        assert result.verdict == EnumValidationVerdict.FAIL
        assert "publish error" in result.message.lower()

    def test_fail_when_consume_raises(self) -> None:
        """Consume function raises an exception."""

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            pass

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            raise TimeoutError("Consumer timed out")

        contract = _make_contract()
        result = check_handler_execution(contract, mock_publish, mock_consume)

        assert result.verdict == EnumValidationVerdict.FAIL
        assert "consume error" in result.message.lower()

    def test_synthetic_event_has_verify_prefix(self) -> None:
        """Published event uses verify- correlation_id prefix."""
        published_payloads: list[dict[str, str]] = []

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            published_payloads.append(payload)

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            return []

        contract = _make_contract()
        check_handler_execution(contract, mock_publish, mock_consume)

        assert len(published_payloads) == 1
        cid = published_payloads[0]["correlation_id"]
        assert cid.startswith(VERIFY_CORRELATION_PREFIX)

    def test_publishes_to_first_subscribe_topic(self) -> None:
        """Synthetic event is published to the first subscribe topic."""
        published_topics: list[str] = []

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            published_topics.append(topic)

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            return []

        contract = _make_contract(
            subscribe_topics=("topic.a.v1", "topic.b.v1"),
        )
        check_handler_execution(contract, mock_publish, mock_consume)

        assert published_topics == ["topic.a.v1"]

    def test_consumes_from_first_publish_topic(self) -> None:
        """Probe consumes from the first publish topic."""
        consumed_topics: list[str] = []

        def mock_publish(topic: str, payload: dict[str, str]) -> None:
            pass

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            consumed_topics.append(topic)
            return []

        contract = _make_contract(
            publish_topics=("out.a.v1", "out.b.v1"),
        )
        check_handler_execution(contract, mock_publish, mock_consume)

        assert consumed_topics == ["out.a.v1"]

    def test_contract_name_preserved(self) -> None:
        """Result carries the contract name through."""
        contract = _make_contract(name="my_custom_contract")
        result = check_handler_execution(
            contract,
            lambda t, p: None,
            lambda t, to: [],
        )
        assert result.contract_name == "my_custom_contract"

    def test_custom_timeout_passed_to_consume(self) -> None:
        """Custom timeout is forwarded to the consume function."""
        received_timeouts: list[float] = []

        def mock_consume(topic: str, timeout: float) -> list[dict[str, str]]:
            received_timeouts.append(timeout)
            return []

        contract = _make_contract()
        check_handler_execution(
            contract,
            lambda t, p: None,
            mock_consume,
            timeout_seconds=42.5,
        )

        assert received_timeouts == [42.5]
