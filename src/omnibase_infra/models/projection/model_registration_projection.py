# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Projection Model.

Provides the Pydantic model for registration projections stored in PostgreSQL.
Used by orchestrators to query current registration state and make workflow
decisions without scanning Kafka topics.

Concurrency Safety:
    This model is mutable (frozen=False) to allow updates during projection
    persistence. Callers should ensure safe concurrent access when updating
    (e.g., using asyncio.Lock for coroutine-safety or threading.Lock for
    thread-safety, depending on the execution context).

Related Tickets:
    - OMN-1006: Add last_heartbeat_at for liveness expired event reporting
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumContractType, EnumRegistrationState
from omnibase_infra.models.projection.model_sequence_info import ModelSequenceInfo
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.utils import validate_contract_type_value

# Valid contract types for nodes (excludes runtime_host which is not a contract type)
# Derived from EnumContractType.valid_type_values() for type consistency
ContractType = Literal["effect", "compute", "reducer", "orchestrator"]
VALID_CONTRACT_TYPES: tuple[str, ...] = EnumContractType.valid_type_values()

# Contract type including 'unknown' for backfill/migration scenarios.
# The 'unknown' value is allowed at the model layer but rejected at the persistence
# layer unless allow_unknown_backfill=True is explicitly passed.
# See: EnumContractType.UNKNOWN documentation and persist_state_transition()
ContractTypeWithUnknown = Literal[
    "effect", "compute", "reducer", "orchestrator", "unknown"
]


class ModelRegistrationProjection(BaseModel):
    """Registration projection for orchestrator state queries.

    Stores materialized registration state that orchestrators read to make
    workflow decisions. This is the canonical read model for registration state.

    Design Notes (per F1 requirements):
        - current_state: FSM state for workflow decisions
        - ack_deadline/liveness_deadline: Timeout deadlines for C2 (durable timeout)
        - last_applied_event_id: message_id for idempotency (B3)
        - last_applied_offset: Kafka offset for ordering (canonical)
        - last_applied_sequence: Generic sequence (non-Kafka transports)

    Primary Key:
        (entity_id, domain) - composite key for multi-domain support

    Ordering Invariant:
        Per-entity monotonic ordering based on (partition, offset) or sequence.
        Projector rejects stale updates where incoming sequence <= current.

    Attributes:
        entity_id: Node UUID (partition key, registration identity)
        domain: Domain namespace for multi-domain support (default: "registration")
        current_state: Current FSM state for workflow decisions
        node_type: ONEX node type (effect, compute, reducer, orchestrator)
        node_version: Semantic version of the registered node
        capabilities: Node capabilities snapshot at registration time
        ack_deadline: Deadline for node acknowledgment (nullable)
        liveness_deadline: Deadline for next heartbeat (nullable)
        last_heartbeat_at: Timestamp of last received heartbeat (nullable)
        ack_timeout_emitted_at: Marker for ack timeout event deduplication (C2)
        liveness_timeout_emitted_at: Marker for liveness timeout deduplication (C2)
        last_applied_event_id: message_id of last applied event (idempotency)
        last_applied_offset: Kafka offset of last applied event (ordering)
        last_applied_sequence: Sequence of last applied event (non-Kafka)
        last_applied_partition: Kafka partition of last applied event
        registered_at: Timestamp of initial registration
        updated_at: Timestamp of last projection update
        correlation_id: Correlation ID for distributed tracing

    Example:
        >>> from datetime import datetime, UTC
        >>> from uuid import uuid4
        >>> from omnibase_core.enums import EnumNodeKind
        >>> now = datetime.now(UTC)
        >>> projection = ModelRegistrationProjection(
        ...     entity_id=uuid4(),
        ...     current_state=EnumRegistrationState.ACTIVE,
        ...     node_type=EnumNodeKind.EFFECT,
        ...     last_applied_event_id=uuid4(),
        ...     last_applied_offset=12345,
        ...     registered_at=now,
        ...     updated_at=now,
        ... )

    Note:
        When serialized to JSON via ``model_dump(mode="json")``, the ``node_type``
        field is serialized as its string value (e.g., ``"effect"``), not the
        enum member name. This is Pydantic's default enum serialization behavior.
        When deserializing, both ``EnumNodeKind.EFFECT`` and ``"effect"`` are
        accepted due to Pydantic's automatic coercion.
    """

    model_config = ConfigDict(
        frozen=False,  # Mutable for updates during projection persistence
        extra="forbid",
        from_attributes=True,
    )

    # Identity (composite primary key: entity_id + domain)
    entity_id: UUID = Field(
        ...,
        description="Node UUID (partition key for per-entity ordering)",
    )
    domain: str = Field(
        default="registration",
        min_length=1,
        max_length=128,
        description="Domain namespace for multi-domain support",
    )

    # FSM State
    current_state: EnumRegistrationState = Field(
        ...,
        description="Current FSM state for workflow decisions",
    )

    # Node Information (snapshot at registration time)
    node_type: EnumNodeKind = Field(
        ...,
        description="ONEX node type",
    )
    node_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Semantic version of the registered node",
    )

    @field_validator("node_version", mode="before")
    @classmethod
    def parse_node_version(cls, v: ModelSemVer | str) -> ModelSemVer:
        """Parse node_version from string or ModelSemVer.

        Args:
            v: Either a ModelSemVer instance or a semver string.

        Returns:
            Validated ModelSemVer instance.

        Raises:
            ValueError: If the string is not a valid semantic version.
        """
        if isinstance(v, str):
            try:
                return ModelSemVer.parse(v)
            except Exception as e:
                raise ValueError(f"node_version: {e!s}") from e
        return v

    capabilities: ModelNodeCapabilities = Field(
        default_factory=ModelNodeCapabilities,
        description="Node capabilities snapshot at registration",
    )

    # Capability fields for fast discovery queries (OMN-1134)
    # These are denormalized from capabilities for GIN-indexed queries
    # Uses ContractTypeWithUnknown to allow 'unknown' for backfill scenarios.
    # The 'unknown' value is accepted here but rejected at persistence layer
    # unless allow_unknown_backfill=True is explicitly passed.
    contract_type: ContractTypeWithUnknown | None = Field(
        default=None,
        description=(
            "Contract type for the node. Valid values: 'effect', 'compute', "
            "'reducer', 'orchestrator'. 'unknown' is allowed for backfill but "
            "rejected at persistence unless explicitly permitted. None indicates "
            "unspecified (never processed)."
        ),
    )

    @field_validator("contract_type", mode="before")
    @classmethod
    def validate_contract_type(cls, v: str | None) -> str | None:
        """Validate contract_type is a valid node contract type.

        Delegates to shared utility for consistent validation across all models.

        Note:
            The 'unknown' value is accepted here to allow model construction for
            backfill scenarios. However, `persist_state_transition()` will reject
            'unknown' unless `allow_unknown_backfill=True` is explicitly passed.
        """
        return validate_contract_type_value(v)

    intent_types: list[str] = Field(
        default_factory=list,
        description="Intent types this node handles",
    )
    protocols: list[str] = Field(
        default_factory=list,
        description="Protocols this node implements",
    )
    capability_tags: list[str] = Field(
        default_factory=list,
        description="Capability tags for discovery",
    )
    contract_version: str | None = Field(
        default=None,
        description="Contract version string",
    )

    # Timeout Deadlines (for C2 durable timeout handling)
    ack_deadline: datetime | None = Field(
        default=None,
        description="Deadline for node acknowledgment (nullable)",
    )
    liveness_deadline: datetime | None = Field(
        default=None,
        description="Deadline for next heartbeat (nullable)",
    )
    last_heartbeat_at: datetime | None = Field(
        default=None,
        description="Timestamp of last received heartbeat (nullable)",
    )

    # Timeout Emission Markers (for C2 deduplication)
    # These prevent emitting duplicate timeout events during replay
    ack_timeout_emitted_at: datetime | None = Field(
        default=None,
        description="Marker: ack timeout event already emitted at this time",
    )
    liveness_timeout_emitted_at: datetime | None = Field(
        default=None,
        description="Marker: liveness timeout event already emitted at this time",
    )

    # Idempotency and Ordering (per F1 requirements)
    last_applied_event_id: UUID = Field(
        ...,
        description="message_id of last applied event (idempotency)",
    )
    last_applied_offset: int = Field(
        default=0,
        ge=0,
        description="Kafka offset of last applied event (canonical ordering)",
    )
    last_applied_sequence: int | None = Field(
        default=None,
        ge=0,
        description="Sequence number for non-Kafka transports",
    )
    last_applied_partition: str | None = Field(
        default=None,
        description="Kafka partition of last applied event",
    )

    # Timestamps
    registered_at: datetime = Field(
        ...,
        description="Timestamp of initial registration",
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp of last projection update",
    )

    # Tracing
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing",
    )

    def get_sequence_info(self) -> ModelSequenceInfo:
        """Build sequence info from projection fields.

        Constructs a ModelSequenceInfo from the idempotency tracking fields
        for use in staleness comparisons.

        Returns:
            ModelSequenceInfo with sequence/partition/offset from projection

        Example:
            >>> proj = ModelRegistrationProjection(...)
            >>> seq_info = proj.get_sequence_info()
            >>> seq_info.sequence
            12345
        """
        # Sequence selection: prefer last_applied_sequence when explicitly set,
        # otherwise fall back to last_applied_offset for Kafka-based ordering.
        # Note: sequence=0 is valid; we check "is not None" not truthiness.
        sequence = (
            self.last_applied_sequence
            if self.last_applied_sequence is not None
            else self.last_applied_offset
        )

        # Offset is only meaningful when partition is present (Kafka-based mode).
        # For non-Kafka transports, partition is None and offset should also be None
        # to signal that ordering uses the generic sequence field instead.
        offset = self.last_applied_offset if self.last_applied_partition else None

        return ModelSequenceInfo(
            sequence=sequence,
            partition=self.last_applied_partition,
            offset=offset,
        )

    def is_stale(self, incoming_sequence: ModelSequenceInfo) -> bool:
        """Check if incoming event is stale compared to current projection state.

        An incoming event is stale if its sequence is older than or equal to
        the current projection's sequence. Stale updates should be rejected
        by the projector to maintain ordering correctness.

        Args:
            incoming_sequence: Sequence info from the incoming event

        Returns:
            True if incoming sequence is stale, False if it should be applied

        Example:
            >>> proj = ModelRegistrationProjection(last_applied_offset=100, ...)
            >>> old_seq = ModelSequenceInfo(sequence=50)
            >>> proj.is_stale(old_seq)
            True
            >>> new_seq = ModelSequenceInfo(sequence=150)
            >>> proj.is_stale(new_seq)
            False
        """
        current = self.get_sequence_info()
        return incoming_sequence.is_stale_compared_to(current)

    def has_ack_deadline_passed(self, now: datetime) -> bool:
        """Check if ack deadline has passed.

        Used by orchestrators during timeout scans (C2) to find
        registrations that need ack timeout events emitted.

        Args:
            now: Current time (injected by runtime)

        Returns:
            True if ack_deadline exists and has passed, False otherwise

        Example:
            >>> from datetime import datetime, UTC, timedelta
            >>> proj = ModelRegistrationProjection(
            ...     ack_deadline=datetime.now(UTC) - timedelta(minutes=5),
            ...     ...
            ... )
            >>> proj.has_ack_deadline_passed(datetime.now(UTC))
            True
        """
        if self.ack_deadline is None:
            return False
        return now > self.ack_deadline

    def has_liveness_deadline_passed(self, now: datetime) -> bool:
        """Check if liveness deadline has passed.

        Used by orchestrators during timeout scans (C2) to find
        active registrations that need liveness timeout events emitted.

        Args:
            now: Current time (injected by runtime)

        Returns:
            True if liveness_deadline exists and has passed, False otherwise
        """
        if self.liveness_deadline is None:
            return False
        return now > self.liveness_deadline

    def needs_ack_timeout_event(self, now: datetime) -> bool:
        """Check if ack timeout event should be emitted.

        Returns True if:
        - ack_deadline has passed
        - ack_timeout_emitted_at is None (not yet emitted)
        - current_state requires ack (ACCEPTED or AWAITING_ACK)

        Used by orchestrators during RuntimeTick processing (C2).

        Args:
            now: Current time (injected by runtime)

        Returns:
            True if ack timeout event should be emitted, False otherwise
        """
        if not self.has_ack_deadline_passed(now):
            return False
        if self.ack_timeout_emitted_at is not None:
            return False
        return self.current_state.requires_ack()

    def needs_liveness_timeout_event(self, now: datetime) -> bool:
        """Check if liveness timeout event should be emitted.

        Returns True if:
        - liveness_deadline has passed
        - liveness_timeout_emitted_at is None (not yet emitted)
        - current_state is ACTIVE

        Used by orchestrators during RuntimeTick processing (C2).

        Args:
            now: Current time (injected by runtime)

        Returns:
            True if liveness timeout event should be emitted, False otherwise
        """
        if not self.has_liveness_deadline_passed(now):
            return False
        if self.liveness_timeout_emitted_at is not None:
            return False
        return self.current_state.is_active()


__all__: list[str] = [
    "ContractType",
    "ContractTypeWithUnknown",
    "ModelRegistrationProjection",
    "VALID_CONTRACT_TYPES",
]
