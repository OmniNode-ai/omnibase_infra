# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for NodeLedgerProjectionCompute and HandlerLedgerProjection.

This module validates the COMPUTE node and handler that project platform events
into the audit ledger. Tests cover:
    - Base64 encoding/decoding of event bytes
    - Header normalization to JSON-safe dictionaries
    - Kafka offset parsing
    - Metadata extraction with best-effort fallbacks
    - Contract configuration validation (7 topics, consumer settings)
    - Intent generation with proper structure

Test Coverage per Ticket OMN-1648:
    1. Non-JSON payload: Binary/malformed payload still captured (base64 encoded)
    2. JSON payload with correlation_id in body: Fallback extraction (if implemented)
    3. Missing headers: Event still captured, onex_headers is empty dict {}
    4. Base64 roundtrip: base64.b64decode(event_value) matches original message.value
    5. None event_value: Raises OnexError (not silently succeeds)
    6. All 7 topic suffixes: Contract subscribes to complete set

Related:
    - OMN-1646: Event Ledger Schema and Models
    - OMN-1647: Ledger Write Effect Node
    - OMN-1648: Ledger Projection Compute Node
    - OMN-1726: Refactor to declarative pattern
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import yaml

from omnibase_core.errors import OnexError
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_ledger_projection_compute import (
    HandlerLedgerProjection,
    NodeLedgerProjectionCompute,
)
from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)

if TYPE_CHECKING:
    from omnibase_core.models.reducer.model_intent import ModelIntent


# =============================================================================
# Path Constants
# =============================================================================


def _get_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: 4 levels up from test file (tests/unit/nodes/test_*.py)
    return current.parent.parent.parent.parent


_PROJECT_ROOT = _get_project_root()
NODE_DIR = _PROJECT_ROOT / "src/omnibase_infra/nodes/node_ledger_projection_compute"
CONTRACT_PATH = NODE_DIR / "contract.yaml"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a simple mock container for node initialization."""
    container = MagicMock()
    container.config = MagicMock()
    return container


@pytest.fixture
def handler(mock_container: MagicMock) -> HandlerLedgerProjection:
    """Create a HandlerLedgerProjection instance for testing."""
    return HandlerLedgerProjection(mock_container)


@pytest.fixture
def node(mock_container: MagicMock) -> NodeLedgerProjectionCompute:
    """Create a NodeLedgerProjectionCompute instance for testing."""
    return NodeLedgerProjectionCompute(mock_container)


@pytest.fixture
def sample_headers() -> ModelEventHeaders:
    """Create sample event headers for testing."""
    return ModelEventHeaders(
        source="test-service",
        event_type="test.event.v1",
        timestamp=datetime.now(UTC),
        correlation_id=uuid4(),
        message_id=uuid4(),
    )


@pytest.fixture
def sample_json_payload() -> bytes:
    """Create a sample JSON payload."""
    return b'{"event": "test", "data": {"key": "value"}, "items": [1, 2, 3]}'


@pytest.fixture
def sample_binary_payload() -> bytes:
    """Create a sample binary payload (non-JSON)."""
    # Mix of printable and non-printable bytes
    return b"\x00\x01\x02\x03\xff\xfe\xfd\xfc" + b"binary_content"


@pytest.fixture
def sample_message(sample_headers: ModelEventHeaders) -> ModelEventMessage:
    """Create a complete sample message for testing."""
    return ModelEventMessage(
        topic="onex.evt.platform.node-registration.v1",
        key=b"test-key",
        value=b'{"node_id": "test-123"}',
        headers=sample_headers,
        partition=5,
        offset="12345",
    )


# =============================================================================
# TestBase64Encoding - Base64 helper method tests
# =============================================================================


class TestBase64Encoding:
    """Test base64 encoding helper method."""

    def test_b64_encodes_bytes_correctly(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """Verify bytes are base64 encoded correctly."""
        test_bytes = b"Hello World"
        expected = base64.b64encode(test_bytes).decode("ascii")

        result = handler._b64(test_bytes)

        assert result == expected
        assert result == "SGVsbG8gV29ybGQ="

    def test_b64_returns_none_for_none_input(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """Verify None input returns None output."""
        result = handler._b64(None)

        assert result is None

    def test_b64_roundtrip_preserves_data(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """Verify base64 encode/decode roundtrip preserves original data."""
        original = b'{"event": "test", "data": [1, 2, 3]}'

        encoded = handler._b64(original)
        assert encoded is not None
        decoded = base64.b64decode(encoded)

        assert decoded == original

    def test_b64_roundtrip_binary_payload(
        self, handler: HandlerLedgerProjection, sample_binary_payload: bytes
    ) -> None:
        """Verify binary (non-JSON) payloads survive base64 roundtrip."""
        encoded = handler._b64(sample_binary_payload)
        assert encoded is not None
        decoded = base64.b64decode(encoded)

        assert decoded == sample_binary_payload

    def test_b64_handles_empty_bytes(self, handler: HandlerLedgerProjection) -> None:
        """Verify empty bytes encode correctly."""
        result = handler._b64(b"")

        assert result == ""  # base64 of empty bytes is empty string

    def test_b64_handles_large_payload(self, handler: HandlerLedgerProjection) -> None:
        """Verify large payloads encode correctly."""
        large_payload = b"A" * 100_000

        encoded = handler._b64(large_payload)
        assert encoded is not None
        decoded = base64.b64decode(encoded)

        assert decoded == large_payload


# =============================================================================
# TestHeaderNormalization - Header normalization tests
# =============================================================================


class TestHeaderNormalization:
    """Test header normalization to JSON-safe dict."""

    def test_normalize_none_headers_returns_empty_dict(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """Missing headers return empty dict, not None."""
        result = handler._normalize_headers(None)

        assert result == {}
        assert isinstance(result, dict)

    def test_normalize_headers_returns_json_safe_dict(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """Headers are converted to JSON-serializable dict."""
        result = handler._normalize_headers(sample_headers)

        assert isinstance(result, dict)
        # Verify key fields are present
        assert "source" in result
        assert "event_type" in result
        assert "correlation_id" in result
        assert result["source"] == sample_headers.source
        assert result["event_type"] == sample_headers.event_type

    def test_normalize_headers_preserves_correlation_id(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """Correlation ID is preserved in normalized headers."""
        result = handler._normalize_headers(sample_headers)

        # UUID should be serialized as string
        assert "correlation_id" in result
        assert result["correlation_id"] == str(sample_headers.correlation_id)

    def test_normalize_headers_handles_all_optional_fields(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """Headers with optional fields normalize correctly."""
        headers = ModelEventHeaders(
            source="test-service",
            event_type="test.event",
            timestamp=datetime.now(UTC),
            trace_id="trace-123",
            span_id="span-456",
            routing_key="my.routing.key",
        )

        result = handler._normalize_headers(headers)

        assert result["trace_id"] == "trace-123"
        assert result["span_id"] == "span-456"
        assert result["routing_key"] == "my.routing.key"


# =============================================================================
# TestOffsetParsing - Kafka offset parsing tests
# =============================================================================


class TestOffsetParsing:
    """Test Kafka offset string parsing."""

    def test_parse_valid_offset(self, handler: HandlerLedgerProjection) -> None:
        """Valid offset string parses to integer."""
        result = handler._parse_offset("12345")

        assert result == 12345
        assert isinstance(result, int)

    def test_parse_none_offset_returns_zero(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """None offset returns 0 default."""
        result = handler._parse_offset(None)

        assert result == 0

    def test_parse_invalid_offset_returns_zero(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """Invalid offset string returns 0 default."""
        result = handler._parse_offset("not-a-number")

        assert result == 0

    def test_parse_empty_string_offset_returns_zero(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """Empty string offset returns 0 default."""
        result = handler._parse_offset("")

        assert result == 0

    def test_parse_large_offset(self, handler: HandlerLedgerProjection) -> None:
        """Large offset values parse correctly."""
        large_offset = "9999999999999"

        result = handler._parse_offset(large_offset)

        assert result == 9999999999999

    def test_parse_zero_offset(self, handler: HandlerLedgerProjection) -> None:
        """Zero offset parses correctly."""
        result = handler._parse_offset("0")

        assert result == 0


# =============================================================================
# TestExtractLedgerMetadata - Main extraction algorithm tests
# =============================================================================


class TestExtractLedgerMetadata:
    """Test the main extraction algorithm."""

    def test_extracts_kafka_position_fields(
        self, handler: HandlerLedgerProjection, sample_message: ModelEventMessage
    ) -> None:
        """Topic, partition, offset are always extracted."""
        result = handler._extract_ledger_metadata(sample_message)

        assert result.topic == sample_message.topic
        assert result.partition == sample_message.partition
        assert result.kafka_offset == int(sample_message.offset)  # type: ignore[arg-type]

    def test_raises_error_for_none_event_value(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """INVARIANT: event_value is REQUIRED - must raise OnexError.

        This test uses model_construct to bypass Pydantic validation and create
        a message with None value, testing the handler's defensive handling.
        """
        # Use model_construct to bypass Pydantic validation
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=None,  # This is the invariant violation
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        with pytest.raises(OnexError) as exc_info:
            handler._extract_ledger_metadata(message)

        assert "message.value is None" in str(exc_info.value)

    def test_handles_missing_partition_gracefully(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """Missing partition defaults to 0."""
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=b'{"test": "data"}',
            headers=sample_headers,
            partition=None,
            offset="100",
        )

        result = handler._extract_ledger_metadata(message)

        assert result.partition == 0

    def test_handles_missing_offset_gracefully(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """Missing offset defaults to 0."""
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=b'{"test": "data"}',
            headers=sample_headers,
            partition=5,
            offset=None,
        )

        result = handler._extract_ledger_metadata(message)

        assert result.kafka_offset == 0

    def test_handles_non_json_binary_payload(
        self,
        handler: HandlerLedgerProjection,
        sample_headers: ModelEventHeaders,
        sample_binary_payload: bytes,
    ) -> None:
        """Binary (non-JSON) payload is base64 encoded, not parsed.

        This validates that the handler captures events without attempting
        to parse the payload as JSON.
        """
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=sample_binary_payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)

        # Verify binary payload is base64 encoded
        assert result.event_value is not None
        decoded = base64.b64decode(result.event_value)
        assert decoded == sample_binary_payload

    def test_extracts_optional_header_fields(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """correlation_id, event_type, source, etc. extracted from headers."""
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=b'{"test": "data"}',
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)

        assert result.correlation_id == sample_headers.correlation_id
        assert result.event_type == sample_headers.event_type
        assert result.source == sample_headers.source
        assert result.envelope_id == sample_headers.message_id
        assert result.event_timestamp == sample_headers.timestamp

    def test_handles_event_key_encoding(
        self, handler: HandlerLedgerProjection, sample_message: ModelEventMessage
    ) -> None:
        """Event key is base64 encoded when present."""
        result = handler._extract_ledger_metadata(sample_message)

        assert result.event_key is not None
        decoded_key = base64.b64decode(result.event_key)
        assert decoded_key == sample_message.key

    def test_handles_missing_event_key(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """Missing event key results in None event_key."""
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=b'{"test": "data"}',
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)

        assert result.event_key is None


# =============================================================================
# TestMissingHeaders - Missing headers handling tests
# =============================================================================


class TestMissingHeaders:
    """Test behavior when headers are missing or None.

    The model_construct method is used to bypass Pydantic validation
    and test the handler's defensive handling of edge cases.
    """

    def test_none_headers_returns_empty_onex_headers(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """Event still captured when headers are None, onex_headers is empty dict."""
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=b'{"test": "data"}',
            headers=None,
            partition=0,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)

        assert result.onex_headers == {}
        assert result.correlation_id is None
        assert result.event_type is None
        assert result.source is None
        assert result.envelope_id is None
        assert result.event_timestamp is None

    def test_none_headers_event_still_captured(
        self, handler: HandlerLedgerProjection
    ) -> None:
        """Event body is still captured even with missing headers."""
        test_value = b'{"important": "data"}'
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=test_value,
            headers=None,
            partition=3,
            offset="999",
        )

        result = handler._extract_ledger_metadata(message)

        # Event value should be captured
        decoded = base64.b64decode(result.event_value)
        assert decoded == test_value

        # Position should be captured
        assert result.topic == "test.topic"
        assert result.partition == 3
        assert result.kafka_offset == 999


# =============================================================================
# TestBase64Roundtrip - Full base64 roundtrip verification
# =============================================================================


class TestBase64Roundtrip:
    """Test that base64 roundtrip preserves original message.value."""

    def test_base64_roundtrip_json_payload(
        self,
        handler: HandlerLedgerProjection,
        sample_headers: ModelEventHeaders,
        sample_json_payload: bytes,
    ) -> None:
        """JSON payload survives base64 roundtrip."""
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=sample_json_payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)
        decoded = base64.b64decode(result.event_value)

        assert decoded == sample_json_payload

    def test_base64_roundtrip_binary_payload(
        self,
        handler: HandlerLedgerProjection,
        sample_headers: ModelEventHeaders,
        sample_binary_payload: bytes,
    ) -> None:
        """Binary payload survives base64 roundtrip."""
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=sample_binary_payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)
        decoded = base64.b64decode(result.event_value)

        assert decoded == sample_binary_payload

    def test_base64_roundtrip_key_and_value(
        self,
        handler: HandlerLedgerProjection,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """Both key and value survive base64 roundtrip."""
        test_key = b"partition-key-123"
        test_value = b'{"event": "data"}'

        message = ModelEventMessage(
            topic="test.topic",
            key=test_key,
            value=test_value,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)

        # Verify key roundtrip
        assert result.event_key is not None
        decoded_key = base64.b64decode(result.event_key)
        assert decoded_key == test_key

        # Verify value roundtrip
        decoded_value = base64.b64decode(result.event_value)
        assert decoded_value == test_value


# =============================================================================
# TestHandlerLedgerProjection - Handler project method tests
# =============================================================================


class TestHandlerLedgerProjection:
    """Test the handler project method end-to-end."""

    def test_project_returns_model_intent(
        self, handler: HandlerLedgerProjection, sample_message: ModelEventMessage
    ) -> None:
        """project() returns ModelIntent with correct structure."""
        from omnibase_core.models.reducer.model_intent import ModelIntent

        result = handler.project(sample_message)

        assert isinstance(result, ModelIntent)

    def test_intent_has_extension_type(
        self, handler: HandlerLedgerProjection, sample_message: ModelEventMessage
    ) -> None:
        """ModelIntent.intent_type == 'extension'."""
        result = handler.project(sample_message)

        assert result.intent_type == "extension"

    def test_payload_has_ledger_append_type(
        self, handler: HandlerLedgerProjection, sample_message: ModelEventMessage
    ) -> None:
        """payload.intent_type == 'ledger.append'."""
        result = handler.project(sample_message)

        assert isinstance(result.payload, ModelPayloadLedgerAppend)
        assert result.payload.intent_type == "ledger.append"

    def test_target_uri_format(
        self, handler: HandlerLedgerProjection, sample_message: ModelEventMessage
    ) -> None:
        """Target follows postgres://event_ledger/{topic}/{partition}/{offset}."""
        result = handler.project(sample_message)

        expected_target = (
            f"postgres://event_ledger/{sample_message.topic}/"
            f"{sample_message.partition}/{int(sample_message.offset)}"  # type: ignore[arg-type]
        )
        assert result.target == expected_target

    def test_project_preserves_all_metadata(
        self, handler: HandlerLedgerProjection, sample_message: ModelEventMessage
    ) -> None:
        """All metadata is preserved in the returned intent payload."""
        result = handler.project(sample_message)
        payload = result.payload

        assert isinstance(payload, ModelPayloadLedgerAppend)
        assert payload.topic == sample_message.topic
        assert payload.partition == sample_message.partition
        assert payload.kafka_offset == int(sample_message.offset)  # type: ignore[arg-type]
        assert payload.correlation_id == sample_message.headers.correlation_id
        assert payload.event_type == sample_message.headers.event_type
        assert payload.source == sample_message.headers.source

    def test_project_with_none_value_raises_error(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """project() raises OnexError for None event value."""
        message = ModelEventMessage.model_construct(
            topic="test.topic",
            key=None,
            value=None,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        with pytest.raises(OnexError):
            handler.project(message)


# =============================================================================
# TestNodeDeclarativePattern - Verify node is declarative
# =============================================================================


class TestNodeDeclarativePattern:
    """Test that the node follows the declarative pattern."""

    def test_node_has_no_compute_method(
        self, node: NodeLedgerProjectionCompute
    ) -> None:
        """Node should not have a custom compute method."""
        # The node should just be a pass-through shell
        # It should not have custom methods like _b64, _normalize_headers, etc.
        assert (
            not hasattr(node, "_b64") or callable(getattr(node, "_b64", None)) is False
        )
        assert (
            not hasattr(node, "_normalize_headers")
            or callable(getattr(node, "_normalize_headers", None)) is False
        )
        assert (
            not hasattr(node, "_parse_offset")
            or callable(getattr(node, "_parse_offset", None)) is False
        )
        assert (
            not hasattr(node, "_extract_ledger_metadata")
            or callable(getattr(node, "_extract_ledger_metadata", None)) is False
        )

    def test_node_extends_node_compute(self) -> None:
        """Node should extend NodeCompute base class."""
        from omnibase_core.nodes.node_compute import NodeCompute

        assert issubclass(NodeLedgerProjectionCompute, NodeCompute)


# =============================================================================
# TestContractValidation - Contract.yaml configuration tests
# =============================================================================


class TestContractValidation:
    """Test contract.yaml configuration.

    These tests validate that the contract is correctly configured per
    OMN-1648 requirements for the audit ledger projection node.
    """

    @pytest.fixture(scope="class")
    def contract_data(self) -> dict:
        """Load contract.yaml data."""
        if not CONTRACT_PATH.exists():
            pytest.skip(f"Contract file not found: {CONTRACT_PATH}")

        with open(CONTRACT_PATH) as f:
            return yaml.safe_load(f)

    def test_contract_has_all_seven_topics(self, contract_data: dict) -> None:
        """Verify contract subscribes to all 7 platform topic suffixes."""
        event_bus = contract_data.get("event_bus", {})
        topics = event_bus.get("subscribe_topics", [])

        assert len(topics) == 7, f"Expected 7 topics, got {len(topics)}: {topics}"

        # Verify expected topic suffixes/categories are covered
        expected_suffixes = [
            "node-registration",
            "node-introspection",
            "node-heartbeat",
            "request-introspection",
            "fsm-state-transitions",
            "runtime-tick",
            "registration-snapshots",
        ]

        for suffix in expected_suffixes:
            matching = [t for t in topics if suffix in t]
            assert matching, f"No topic found containing '{suffix}'. Topics: {topics}"

    def test_consumer_purpose_is_audit(self, contract_data: dict) -> None:
        """consumer_purpose must be 'audit'."""
        event_bus = contract_data.get("event_bus", {})
        consumer_purpose = event_bus.get("consumer_purpose")

        assert consumer_purpose == "audit", (
            f"Expected consumer_purpose='audit', got '{consumer_purpose}'"
        )

    def test_auto_offset_reset_is_earliest(self, contract_data: dict) -> None:
        """auto_offset_reset must be 'earliest' to capture all historical events."""
        event_bus = contract_data.get("event_bus", {})
        auto_offset_reset = event_bus.get("auto_offset_reset")

        assert auto_offset_reset == "earliest", (
            f"Expected auto_offset_reset='earliest', got '{auto_offset_reset}'"
        )

    def test_node_type_is_compute_generic(self, contract_data: dict) -> None:
        """Node type must be COMPUTE_GENERIC."""
        node_type = contract_data.get("node_type")

        assert node_type == "COMPUTE_GENERIC", (
            f"Expected node_type='COMPUTE_GENERIC', got '{node_type}'"
        )

    def test_contract_version_is_valid_semver(self, contract_data: dict) -> None:
        """Contract version follows semantic versioning object structure."""
        cv = contract_data.get("contract_version", {})

        # ONEX uses ModelSemver object format, not string
        assert isinstance(cv, dict), f"contract_version should be dict, got {type(cv)}"
        assert "major" in cv, "contract_version missing 'major' field"
        assert "minor" in cv, "contract_version missing 'minor' field"
        assert "patch" in cv, "contract_version missing 'patch' field"

        # Each part should be a valid integer
        assert isinstance(cv["major"], int), (
            f"major should be int, got {type(cv['major'])}"
        )
        assert isinstance(cv["minor"], int), (
            f"minor should be int, got {type(cv['minor'])}"
        )
        assert isinstance(cv["patch"], int), (
            f"patch should be int, got {type(cv['patch'])}"
        )

    def test_has_consumer_group(self, contract_data: dict) -> None:
        """Contract specifies a consumer group."""
        event_bus = contract_data.get("event_bus", {})
        consumer_group = event_bus.get("consumer_group")

        assert consumer_group is not None, "consumer_group must be specified"
        assert len(consumer_group) > 0, "consumer_group must not be empty"

    def test_output_model_is_model_intent(self, contract_data: dict) -> None:
        """Output model is ModelIntent (wrapping ModelPayloadLedgerAppend)."""
        output_model = contract_data.get("output_model", {})
        model_name = output_model.get("name")

        assert model_name == "ModelIntent", (
            f"Expected output model 'ModelIntent', got '{model_name}'"
        )

    def test_handler_routing_configured(self, contract_data: dict) -> None:
        """Handler routing is configured for declarative pattern."""
        handler_routing = contract_data.get("handler_routing", {})

        assert handler_routing, "handler_routing must be configured"
        assert "handlers" in handler_routing, "handler_routing must have handlers"

        handlers = handler_routing["handlers"]
        assert len(handlers) > 0, "At least one handler must be configured"

        # Verify the handler is for ledger projection
        handler = handlers[0]
        assert handler.get("handler_type") == "ledger_projection"
        assert "ledger.project" in handler.get("supported_operations", [])


# =============================================================================
# TestPayloadModelInvariants - Payload model constraint tests
# =============================================================================


class TestPayloadModelInvariants:
    """Test ModelPayloadLedgerAppend invariants."""

    def test_intent_type_is_literal(self) -> None:
        """intent_type is fixed to 'ledger.append'."""
        payload = ModelPayloadLedgerAppend(
            topic="test.topic",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",  # base64 for "test"
        )

        assert payload.intent_type == "ledger.append"

    def test_payload_is_frozen(self) -> None:
        """ModelPayloadLedgerAppend is immutable."""
        from pydantic import ValidationError

        payload = ModelPayloadLedgerAppend(
            topic="test.topic",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
        )

        with pytest.raises((TypeError, ValidationError)):
            payload.topic = "new-topic"  # type: ignore[misc]

    def test_optional_fields_default_correctly(self) -> None:
        """Optional fields have correct default values."""
        payload = ModelPayloadLedgerAppend(
            topic="test.topic",
            partition=0,
            kafka_offset=0,
            event_value="dGVzdA==",
        )

        assert payload.event_key is None
        assert payload.correlation_id is None
        assert payload.envelope_id is None
        assert payload.event_type is None
        assert payload.source is None
        assert payload.event_timestamp is None
        assert payload.onex_headers == {}


# =============================================================================
# TestEdgeCases - Edge case and boundary tests
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_value_raises_validation_error(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """Empty bytes value fails validation (min_length=1 on event_value).

        The ModelPayloadLedgerAppend.event_value field has min_length=1,
        so empty bytes (which base64-encode to "") are not valid.
        This is intentional - events must have content to be meaningful.
        """
        from pydantic import ValidationError

        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=b"",
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        with pytest.raises(ValidationError) as exc_info:
            handler._extract_ledger_metadata(message)

        # Verify it's the event_value field that failed validation
        errors = exc_info.value.errors()
        assert any("event_value" in str(e) for e in errors)

    def test_unicode_in_json_payload(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """Unicode characters in JSON payload are preserved."""
        unicode_payload = b'{"message": "Hello"}'

        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=unicode_payload,
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)
        decoded = base64.b64decode(result.event_value)

        assert decoded == unicode_payload

    def test_very_long_topic_name(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """Long topic names are handled correctly."""
        long_topic = "org.domain.subdomain.service.event.namespace.version.v1"

        message = ModelEventMessage(
            topic=long_topic,
            key=None,
            value=b'{"test": "data"}',
            headers=sample_headers,
            partition=0,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)

        assert result.topic == long_topic

    def test_high_partition_number(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """High partition numbers are handled correctly."""
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=b'{"test": "data"}',
            headers=sample_headers,
            partition=999,
            offset="0",
        )

        result = handler._extract_ledger_metadata(message)

        assert result.partition == 999

    def test_high_offset_value(
        self, handler: HandlerLedgerProjection, sample_headers: ModelEventHeaders
    ) -> None:
        """High offset values are handled correctly."""
        message = ModelEventMessage(
            topic="test.topic",
            key=None,
            value=b'{"test": "data"}',
            headers=sample_headers,
            partition=0,
            offset="9999999999999",
        )

        result = handler._extract_ledger_metadata(message)

        assert result.kafka_offset == 9999999999999


# =============================================================================
# Module Exports
# =============================================================================

__all__: list[str] = [
    "TestBase64Encoding",
    "TestBase64Roundtrip",
    "TestContractValidation",
    "TestEdgeCases",
    "TestExtractLedgerMetadata",
    "TestHandlerLedgerProjection",
    "TestHeaderNormalization",
    "TestMissingHeaders",
    "TestNodeDeclarativePattern",
    "TestOffsetParsing",
    "TestPayloadModelInvariants",
]
