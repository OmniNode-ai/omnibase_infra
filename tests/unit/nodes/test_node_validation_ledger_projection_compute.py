# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for NodeValidationLedgerProjectionCompute and handler.

Tests cover:
    - HandlerValidationLedgerProjection: domain field extraction, base64/hash
    - Contract configuration: 3 validation topics, consumer settings
    - Node declarative pattern compliance
    - Edge cases: missing fields, invalid JSON, fallback defaults

Ticket: OMN-1908
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.errors import OnexError
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.validation_ledger import ModelValidationLedgerEntry
from omnibase_infra.nodes.node_validation_ledger_projection_compute import (
    HandlerValidationLedgerProjection,
    NodeValidationLedgerProjectionCompute,
)

# =============================================================================
# Path Constants
# =============================================================================


def _get_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent.parent.parent


_PROJECT_ROOT = _get_project_root()
NODE_DIR = (
    _PROJECT_ROOT / "src/omnibase_infra/nodes/node_validation_ledger_projection_compute"
)
CONTRACT_PATH = NODE_DIR / "contract.yaml"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a simple mock container."""
    container = MagicMock()
    container.config = MagicMock()
    return container


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerValidationLedgerProjection:
    """Create handler instance for testing."""
    return HandlerValidationLedgerProjection(mock_container)


@pytest.fixture
def sample_headers() -> ModelEventHeaders:
    """Create sample event headers."""
    return ModelEventHeaders(
        source="cross-repo-validator",
        event_type="onex.validation.cross_repo.run.started.v1",
        timestamp=datetime.now(UTC),
        correlation_id=uuid4(),
        message_id=uuid4(),
    )


@pytest.fixture
def sample_run_id() -> UUID:
    """Create a sample run_id."""
    return uuid4()


@pytest.fixture
def sample_validation_payload(sample_run_id: UUID) -> bytes:
    """Create a sample validation run started event payload."""
    return json.dumps(
        {
            "event_type": "onex.validation.cross_repo.run.started.v1",
            "run_id": str(sample_run_id),
            "repo_id": "omnibase_core",
            "timestamp": datetime.now(UTC).isoformat(),
            "schema_version": "1.0.0",
            "policy_name": "default",
            "rules_enabled": ["naming", "contract"],
            "baseline_applied": False,
        }
    ).encode()


@pytest.fixture
def sample_message(
    sample_headers: ModelEventHeaders,
    sample_validation_payload: bytes,
) -> ModelEventMessage:
    """Create a complete sample validation message."""
    return ModelEventMessage(
        topic="onex.validation.cross_repo.run.started.v1",
        key=None,
        value=sample_validation_payload,
        headers=sample_headers,
        partition=0,
        offset="100",
    )


# =============================================================================
# TestHandlerProject - Main projection logic
# =============================================================================


class TestHandlerProject:
    """Test the handler project method."""

    def test_project_returns_validation_ledger_entry(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_message: ModelEventMessage,
    ) -> None:
        """project() returns ModelValidationLedgerEntry."""
        result = handler.project(sample_message)
        assert isinstance(result, ModelValidationLedgerEntry)

    def test_project_extracts_run_id(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_message: ModelEventMessage,
        sample_run_id: UUID,
    ) -> None:
        """run_id is extracted from JSON payload."""
        result = handler.project(sample_message)
        assert result.run_id == sample_run_id

    def test_project_extracts_repo_id(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_message: ModelEventMessage,
    ) -> None:
        """repo_id is extracted from JSON payload."""
        result = handler.project(sample_message)
        assert result.repo_id == "omnibase_core"

    def test_project_extracts_event_type(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_message: ModelEventMessage,
    ) -> None:
        """event_type is extracted from JSON payload."""
        result = handler.project(sample_message)
        assert result.event_type == "onex.validation.cross_repo.run.started.v1"

    def test_project_base64_encodes_envelope(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_message: ModelEventMessage,
    ) -> None:
        """envelope_bytes is base64-encoded raw message value."""
        result = handler.project(sample_message)

        decoded = base64.b64decode(result.envelope_bytes)
        assert decoded == sample_message.value

    def test_project_computes_sha256_hash(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_message: ModelEventMessage,
    ) -> None:
        """envelope_hash is SHA-256 hex digest of raw value."""
        result = handler.project(sample_message)

        expected_hash = hashlib.sha256(sample_message.value).hexdigest()
        assert result.envelope_hash == expected_hash

    def test_project_captures_kafka_position(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_message: ModelEventMessage,
    ) -> None:
        """Kafka topic, partition, offset captured correctly."""
        result = handler.project(sample_message)

        assert result.kafka_topic == sample_message.topic
        assert result.kafka_partition == 0
        assert result.kafka_offset == 100

    def test_project_generates_uuid_id(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_message: ModelEventMessage,
    ) -> None:
        """Each projection generates a unique UUID id."""
        result1 = handler.project(sample_message)
        result2 = handler.project(sample_message)
        assert result1.id != result2.id


# =============================================================================
# TestDomainFieldExtraction - Required and best-effort fields
# =============================================================================


class TestDomainFieldExtraction:
    """Test domain field extraction from JSON payload."""

    def test_raises_for_none_value(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """None message.value raises OnexError."""
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=None,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        with pytest.raises(OnexError, match=r"message\.value is None"):
            handler.project(message)

    def test_raises_for_invalid_json(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """Invalid JSON raises OnexError."""
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=b"not valid json {{{",
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        with pytest.raises(OnexError, match="Cannot parse validation event JSON"):
            handler.project(message)

    def test_raises_for_missing_run_id(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """Missing run_id raises OnexError."""
        payload = json.dumps({"repo_id": "test", "event_type": "test.v1"}).encode()
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        with pytest.raises(OnexError, match="missing required field 'run_id'"):
            handler.project(message)

    def test_raises_for_missing_repo_id(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """Missing repo_id raises OnexError."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "event_type": "test.v1",
            }
        ).encode()
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        with pytest.raises(OnexError, match="missing or invalid 'repo_id'"):
            handler.project(message)

    def test_event_type_falls_back_to_topic(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """Missing event_type in payload falls back to Kafka topic."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "repo_id": "test",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ).encode()
        message = ModelEventMessage(
            topic="onex.validation.cross_repo.run.started.v1",
            key=None,
            value=payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler.project(message)
        assert result.event_type == "onex.validation.cross_repo.run.started.v1"

    def test_event_version_defaults_to_v1(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """Missing schema_version defaults to 'v1'."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "repo_id": "test",
                "event_type": "test.v1",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ).encode()
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler.project(message)
        assert result.event_version == "v1"

    def test_event_version_extracted_from_schema_version(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """schema_version field used as event_version."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "repo_id": "test",
                "event_type": "test.v1",
                "schema_version": "2.0.0",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ).encode()
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler.project(message)
        assert result.event_version == "2.0.0"

    def test_invalid_timestamp_uses_current_time(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """Invalid timestamp falls back to current time."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "repo_id": "test",
                "event_type": "test.v1",
                "timestamp": "not-a-timestamp",
            }
        ).encode()
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        before = datetime.now(UTC)
        result = handler.project(message)
        after = datetime.now(UTC)

        # occurred_at should be approximately now (normalize tz awareness)
        occurred = result.occurred_at
        if occurred.tzinfo is None:
            from datetime import timezone

            occurred = occurred.replace(tzinfo=UTC)
        assert occurred >= before
        assert occurred <= after


# =============================================================================
# TestKafkaPositionHandling - Partition and offset edge cases
# =============================================================================


class TestKafkaPositionHandling:
    """Test Kafka position field handling."""

    def test_none_partition_defaults_to_zero(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """None partition defaults to 0."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "repo_id": "test",
                "event_type": "test.v1",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ).encode()
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=payload,
            headers=sample_headers,
            partition=None,
            offset="42",
        )

        result = handler.project(message)
        assert result.kafka_partition == 0

    def test_none_offset_defaults_to_zero(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """None offset defaults to 0."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "repo_id": "test",
                "event_type": "test.v1",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ).encode()
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=payload,
            headers=sample_headers,
            partition=0,
            offset=None,
        )

        result = handler.project(message)
        assert result.kafka_offset == 0

    def test_invalid_offset_defaults_to_zero(
        self,
        handler: HandlerValidationLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """Non-integer offset defaults to 0."""
        payload = json.dumps(
            {
                "run_id": str(uuid4()),
                "repo_id": "test",
                "event_type": "test.v1",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ).encode()
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=payload,
            headers=sample_headers,
            partition=0,
            offset="not-a-number",
        )

        result = handler.project(message)
        assert result.kafka_offset == 0


# =============================================================================
# TestContractValidation - Contract configuration
# =============================================================================


class TestContractValidation:
    """Test contract.yaml configuration for validation ledger node."""

    @pytest.fixture(scope="class")
    def contract_data(self) -> dict:
        """Load contract.yaml data."""
        if not CONTRACT_PATH.exists():
            pytest.skip(f"Contract file not found: {CONTRACT_PATH}")
        with open(CONTRACT_PATH) as f:
            return yaml.safe_load(f)

    def test_subscribes_to_three_validation_topics(self, contract_data: dict) -> None:
        """Contract subscribes to exactly 3 validation topics."""
        topics = contract_data.get("event_bus", {}).get("subscribe_topics", [])
        assert len(topics) == 3

        expected_suffixes = [
            "run.started",
            "violations.batch",
            "run.completed",
        ]
        for suffix in expected_suffixes:
            matching = [t for t in topics if suffix in t]
            assert matching, f"No topic found containing '{suffix}'"

    def test_consumer_purpose_is_audit(self, contract_data: dict) -> None:
        """consumer_purpose must be 'audit'."""
        purpose = contract_data.get("event_bus", {}).get("consumer_purpose")
        assert purpose == "audit"

    def test_auto_offset_reset_is_earliest(self, contract_data: dict) -> None:
        """auto_offset_reset must be 'earliest'."""
        reset = contract_data.get("event_bus", {}).get("auto_offset_reset")
        assert reset == "earliest"

    def test_consumer_group_configured(self, contract_data: dict) -> None:
        """Consumer group is specified."""
        group = contract_data.get("event_bus", {}).get("consumer_group")
        assert group is not None
        assert "validation-ledger" in group

    def test_node_type_is_compute_generic(self, contract_data: dict) -> None:
        """Node type is COMPUTE_GENERIC."""
        assert contract_data.get("node_type") == "COMPUTE_GENERIC"

    def test_handler_routing_configured(self, contract_data: dict) -> None:
        """Handler routing has validation ledger projection handler."""
        routing = contract_data.get("handler_routing", {})
        handlers = routing.get("handlers", [])
        assert len(handlers) > 0

        handler = handlers[0]
        assert handler.get("handler_type") == "validation_ledger_projection"
        assert "validation_ledger.project" in handler.get("supported_operations", [])


# =============================================================================
# TestNodeDeclarativePattern
# =============================================================================


class TestNodeDeclarativePattern:
    """Test node follows declarative pattern."""

    def test_extends_node_compute(self) -> None:
        """Node extends NodeCompute."""
        from omnibase_core.nodes.node_compute import NodeCompute

        assert issubclass(NodeValidationLedgerProjectionCompute, NodeCompute)

    def test_node_has_no_custom_methods(self, mock_container: MagicMock) -> None:
        """Node should not have custom compute methods."""
        node = NodeValidationLedgerProjectionCompute(mock_container)
        assert not hasattr(node, "project")
        assert not hasattr(node, "_extract_domain_fields")

    def test_handler_type_is_compute(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Handler type is COMPUTE_HANDLER."""
        assert handler.handler_type == EnumHandlerType.COMPUTE_HANDLER

    def test_handler_category_is_compute(
        self, handler: HandlerValidationLedgerProjection
    ) -> None:
        """Handler category is COMPUTE."""
        assert handler.handler_category == EnumHandlerTypeCategory.COMPUTE


# Import for handler type checks
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
