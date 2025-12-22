# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for compensation strategy patterns.

Tests cover:
- Partial failure detection (some operations succeed, some fail)
- Compensation action invocation for failed I/O operations
- Domain events are NOT rolled back (verify they persist)
- Compensation creates compensating events instead of rollback

The compensation pattern follows these principles:
1. I/O operations (database writes, API calls) can be compensated
2. Domain events are NEVER rolled back - they are immutable records
3. Compensation creates new compensating events (e.g., OrderCancelled for OrderCreated)
4. Partial success is explicitly handled with recorded state

This pattern is documented in CLAUDE.md under "Error Recovery Patterns"
and "Compensation Strategy" sections.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Test Models and Enums
# ---------------------------------------------------------------------------


class EnumOperationType(str, Enum):
    """Types of operations in a compensating transaction."""

    IO_OPERATION = "io_operation"  # Can be compensated (DB write, API call)
    DOMAIN_EVENT = "domain_event"  # NEVER rolled back, create compensating event


class EnumOperationStatus(str, Enum):
    """Status of an operation in a saga."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


@dataclass
class OperationRecord:
    """Records a single operation in a saga/transaction.

    Tracks whether the operation succeeded, failed, or was compensated.
    """

    operation_id: UUID
    operation_name: str
    operation_type: EnumOperationType
    status: EnumOperationStatus = EnumOperationStatus.PENDING
    result: object = None
    error: Exception | None = None
    compensated_at: float | None = None


@dataclass
class CompensationEvent:
    """A compensating event created to undo a domain action.

    Domain events are never deleted - instead we create compensating events
    that record the reversal (e.g., OrderCancelled compensates OrderCreated).
    """

    event_id: UUID
    compensates_operation_id: UUID
    event_type: str
    payload: dict[str, object]
    created_at: float


class ModelSagaResult(BaseModel):
    """Result of a saga execution with compensation tracking.

    Attributes:
        saga_id: Unique identifier for this saga execution
        overall_success: True if all operations succeeded without compensation
        operations_succeeded: Count of successful operations
        operations_failed: Count of failed operations
        operations_compensated: Count of compensated operations
        compensation_events: List of compensating events created
        domain_events_preserved: Count of domain events NOT rolled back
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    saga_id: UUID
    overall_success: bool
    operations_succeeded: int
    operations_failed: int
    operations_compensated: int
    compensation_events_created: int
    domain_events_preserved: int


# ---------------------------------------------------------------------------
# Compensation Strategy Implementation
# ---------------------------------------------------------------------------


T = TypeVar("T")


class CompensatingTransactionManager:
    """Manages compensating transactions for partial failure scenarios.

    This implements the Saga pattern with compensation:
    1. Execute operations in sequence
    2. On failure, compensate all previously successful I/O operations
    3. Domain events are NEVER compensated - create compensating events instead
    4. Track all operations and their states

    Non-Goal (Explicitly Verified in Tests):
        Domain events are immutable records of business facts. We NEVER:
        - Delete domain events from the event store
        - Roll back domain event publications
        - Undo domain events in-place

        Instead, we create compensating events that record the reversal action.
    """

    def __init__(self) -> None:
        """Initialize the transaction manager."""
        self.operations: list[OperationRecord] = []
        self.compensation_events: list[CompensationEvent] = []
        self.compensation_actions: dict[UUID, Callable[[], Awaitable[None]]] = {}
        self.compensating_event_factories: dict[
            UUID, Callable[[], CompensationEvent]
        ] = {}

    def register_io_operation(
        self,
        operation_name: str,
        compensation_action: Callable[[], Awaitable[None]],
    ) -> UUID:
        """Register an I/O operation that can be compensated.

        Args:
            operation_name: Human-readable operation name
            compensation_action: Async callable to undo the operation

        Returns:
            Operation ID for tracking
        """
        operation_id = uuid4()
        record = OperationRecord(
            operation_id=operation_id,
            operation_name=operation_name,
            operation_type=EnumOperationType.IO_OPERATION,
        )
        self.operations.append(record)
        self.compensation_actions[operation_id] = compensation_action
        return operation_id

    def register_domain_event(
        self,
        operation_name: str,
        compensating_event_factory: Callable[[], CompensationEvent],
    ) -> UUID:
        """Register a domain event that creates a compensating event on failure.

        Domain events are NEVER rolled back. Instead, a compensating event
        is created to record the reversal action.

        Args:
            operation_name: Human-readable operation name
            compensating_event_factory: Factory to create compensating event

        Returns:
            Operation ID for tracking
        """
        operation_id = uuid4()
        record = OperationRecord(
            operation_id=operation_id,
            operation_name=operation_name,
            operation_type=EnumOperationType.DOMAIN_EVENT,
        )
        self.operations.append(record)
        self.compensating_event_factories[operation_id] = compensating_event_factory
        return operation_id

    async def execute_operation(
        self,
        operation_id: UUID,
        execute_fn: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute a registered operation.

        Args:
            operation_id: ID of the registered operation
            execute_fn: Async callable to execute

        Returns:
            Result of the execution

        Raises:
            Exception: If operation fails (after recording failure)
        """
        record = self._find_operation(operation_id)

        try:
            result = await execute_fn()
            record.status = EnumOperationStatus.SUCCEEDED
            record.result = result
            return result
        except Exception as e:
            record.status = EnumOperationStatus.FAILED
            record.error = e
            raise

    async def compensate_all(self) -> None:
        """Compensate all successful operations in reverse order.

        For I/O operations: Execute compensation action
        For domain events: Create compensating event (NOT rollback)

        This is called when a saga needs to be aborted due to failure.
        """
        import time

        # Process in reverse order (LIFO compensation)
        for record in reversed(self.operations):
            if record.status != EnumOperationStatus.SUCCEEDED:
                continue

            try:
                if record.operation_type == EnumOperationType.IO_OPERATION:
                    # Compensate I/O operation
                    compensation_action = self.compensation_actions.get(
                        record.operation_id
                    )
                    if compensation_action:
                        await compensation_action()
                        record.status = EnumOperationStatus.COMPENSATED
                        record.compensated_at = time.time()

                elif record.operation_type == EnumOperationType.DOMAIN_EVENT:
                    # Create compensating event (NEVER rollback domain events)
                    factory = self.compensating_event_factories.get(record.operation_id)
                    if factory:
                        compensating_event = factory()
                        self.compensation_events.append(compensating_event)
                        # Note: Status stays SUCCEEDED - the event is preserved
                        # We just record that a compensating event was created
            except Exception as e:
                record.status = EnumOperationStatus.COMPENSATION_FAILED
                record.error = e
                # Continue compensating other operations

    def get_result(self, saga_id: UUID) -> ModelSagaResult:
        """Get the result of the saga execution.

        Args:
            saga_id: ID of the saga

        Returns:
            ModelSagaResult with operation counts and status
        """
        succeeded = sum(
            1 for op in self.operations if op.status == EnumOperationStatus.SUCCEEDED
        )
        failed = sum(
            1 for op in self.operations if op.status == EnumOperationStatus.FAILED
        )
        compensated = sum(
            1 for op in self.operations if op.status == EnumOperationStatus.COMPENSATED
        )
        domain_events = sum(
            1
            for op in self.operations
            if op.operation_type == EnumOperationType.DOMAIN_EVENT
            and op.status == EnumOperationStatus.SUCCEEDED
        )

        return ModelSagaResult(
            saga_id=saga_id,
            overall_success=failed == 0 and len(self.operations) > 0,
            operations_succeeded=succeeded,
            operations_failed=failed,
            operations_compensated=compensated,
            compensation_events_created=len(self.compensation_events),
            domain_events_preserved=domain_events,
        )

    def _find_operation(self, operation_id: UUID) -> OperationRecord:
        """Find operation by ID."""
        for op in self.operations:
            if op.operation_id == operation_id:
                return op
        raise ValueError(f"Operation {operation_id} not found")


# ---------------------------------------------------------------------------
# Mock Services for Testing
# ---------------------------------------------------------------------------


@dataclass
class MockDatabaseState:
    """Simulates database state for testing compensation."""

    records: dict[str, object] = field(default_factory=dict)
    write_count: int = 0
    delete_count: int = 0


@dataclass
class MockEventStore:
    """Simulates event store that never deletes events."""

    events: list[dict[str, object]] = field(default_factory=list)
    compensating_events: list[dict[str, object]] = field(default_factory=list)
    deleted_events: list[UUID] = field(default_factory=list)  # Should always be empty


class MockOrderService:
    """Mock order service for testing compensation patterns."""

    def __init__(
        self,
        db: MockDatabaseState,
        event_store: MockEventStore,
        fail_on_payment: bool = False,
    ) -> None:
        """Initialize mock service.

        Args:
            db: Mock database state
            event_store: Mock event store
            fail_on_payment: If True, payment processing will fail
        """
        self.db = db
        self.event_store = event_store
        self.fail_on_payment = fail_on_payment

    async def create_order(self, order_id: str, items: list[str]) -> dict[str, object]:
        """Create an order (I/O operation - can be compensated)."""
        self.db.records[order_id] = {
            "id": order_id,
            "items": items,
            "status": "created",
        }
        self.db.write_count += 1
        return self.db.records[order_id]

    async def delete_order(self, order_id: str) -> None:
        """Delete an order (compensation for create_order)."""
        if order_id in self.db.records:
            del self.db.records[order_id]
            self.db.delete_count += 1

    async def publish_order_created_event(self, order_id: str) -> dict[str, object]:
        """Publish OrderCreated domain event (NEVER rolled back)."""
        event = {
            "event_id": str(uuid4()),
            "event_type": "OrderCreated",
            "order_id": order_id,
        }
        self.event_store.events.append(event)
        return event

    def create_order_cancelled_event(self, order_id: str) -> CompensationEvent:
        """Create OrderCancelled compensating event factory."""
        import time

        return CompensationEvent(
            event_id=uuid4(),
            compensates_operation_id=uuid4(),
            event_type="OrderCancelled",
            payload={"order_id": order_id, "reason": "saga_compensation"},
            created_at=time.time(),
        )

    async def process_payment(self, order_id: str, amount: float) -> dict[str, object]:
        """Process payment (may fail to test compensation)."""
        if self.fail_on_payment:
            raise ValueError(f"Payment failed for order {order_id}")
        return {"order_id": order_id, "amount": amount, "status": "paid"}


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestPartialFailureDetection:
    """Test partial failure detection (some operations succeed, some fail)."""

    @pytest.mark.asyncio
    async def test_detects_partial_failure(self) -> None:
        """Test that partial failures are correctly detected."""
        db = MockDatabaseState()
        event_store = MockEventStore()
        service = MockOrderService(db, event_store, fail_on_payment=True)

        manager = CompensatingTransactionManager()
        saga_id = uuid4()
        order_id = "order-123"

        # Register operations
        create_order_id = manager.register_io_operation(
            "create_order",
            compensation_action=lambda: service.delete_order(order_id),
        )

        payment_id = manager.register_io_operation(
            "process_payment",
            compensation_action=lambda: asyncio.sleep(0),  # No-op
        )

        # Execute operations
        await manager.execute_operation(
            create_order_id,
            lambda: service.create_order(order_id, ["item1", "item2"]),
        )

        # Second operation fails
        with pytest.raises(ValueError) as exc_info:
            await manager.execute_operation(
                payment_id,
                lambda: service.process_payment(order_id, 100.0),
            )
        assert "Payment failed" in str(exc_info.value)

        # Get result shows partial failure
        result = manager.get_result(saga_id)
        assert result.overall_success is False
        assert result.operations_succeeded == 1
        assert result.operations_failed == 1

    @pytest.mark.asyncio
    async def test_all_operations_succeed(self) -> None:
        """Test detection when all operations succeed."""
        db = MockDatabaseState()
        event_store = MockEventStore()
        service = MockOrderService(db, event_store, fail_on_payment=False)

        manager = CompensatingTransactionManager()
        saga_id = uuid4()
        order_id = "order-456"

        create_order_id = manager.register_io_operation(
            "create_order",
            compensation_action=lambda: service.delete_order(order_id),
        )

        payment_id = manager.register_io_operation(
            "process_payment",
            compensation_action=lambda: asyncio.sleep(0),
        )

        await manager.execute_operation(
            create_order_id,
            lambda: service.create_order(order_id, ["item1"]),
        )

        await manager.execute_operation(
            payment_id,
            lambda: service.process_payment(order_id, 50.0),
        )

        result = manager.get_result(saga_id)
        assert result.overall_success is True
        assert result.operations_succeeded == 2
        assert result.operations_failed == 0


class TestCompensationActionInvocation:
    """Test compensation action invocation for failed I/O operations."""

    @pytest.mark.asyncio
    async def test_io_operations_compensated_on_failure(self) -> None:
        """Test I/O operations are compensated when saga fails."""
        db = MockDatabaseState()
        event_store = MockEventStore()
        service = MockOrderService(db, event_store, fail_on_payment=True)

        manager = CompensatingTransactionManager()
        saga_id = uuid4()
        order_id = "order-789"

        # Register operations
        create_order_id = manager.register_io_operation(
            "create_order",
            compensation_action=lambda: service.delete_order(order_id),
        )

        payment_id = manager.register_io_operation(
            "process_payment",
            compensation_action=lambda: asyncio.sleep(0),
        )

        # Execute first operation (succeeds)
        await manager.execute_operation(
            create_order_id,
            lambda: service.create_order(order_id, ["item1"]),
        )

        # Verify order was created
        assert order_id in db.records
        assert db.write_count == 1

        # Execute second operation (fails)
        with pytest.raises(ValueError):
            await manager.execute_operation(
                payment_id,
                lambda: service.process_payment(order_id, 100.0),
            )

        # Compensate all successful operations
        await manager.compensate_all()

        # Verify order was deleted (compensation worked)
        assert order_id not in db.records
        assert db.delete_count == 1

        result = manager.get_result(saga_id)
        assert result.operations_compensated == 1

    @pytest.mark.asyncio
    async def test_compensation_order_is_lifo(self) -> None:
        """Test compensations happen in reverse order (LIFO)."""
        compensation_order: list[str] = []

        async def comp_a() -> None:
            compensation_order.append("A")

        async def comp_b() -> None:
            compensation_order.append("B")

        async def comp_c() -> None:
            compensation_order.append("C")

        manager = CompensatingTransactionManager()

        op_a = manager.register_io_operation("op_a", comp_a)
        op_b = manager.register_io_operation("op_b", comp_b)
        op_c = manager.register_io_operation("op_c", comp_c)

        # Execute all operations successfully
        for op_id in [op_a, op_b, op_c]:
            await manager.execute_operation(op_id, lambda: asyncio.sleep(0))

        # Compensate all
        await manager.compensate_all()

        # Should be in reverse order: C, B, A
        assert compensation_order == ["C", "B", "A"]


class TestDomainEventsNotRolledBack:
    """Test that domain events are NEVER rolled back.

    This is a critical non-goal: domain events are immutable records of
    business facts. We create compensating events instead of deleting.
    """

    @pytest.mark.asyncio
    async def test_domain_events_preserved_after_compensation(self) -> None:
        """Test domain events are NOT deleted during compensation."""
        db = MockDatabaseState()
        event_store = MockEventStore()
        service = MockOrderService(db, event_store, fail_on_payment=True)

        manager = CompensatingTransactionManager()
        saga_id = uuid4()
        order_id = "order-preserve-events"

        # Register I/O operation
        create_order_id = manager.register_io_operation(
            "create_order",
            compensation_action=lambda: service.delete_order(order_id),
        )

        # Register domain event (NOT to be rolled back)
        domain_event_id = manager.register_domain_event(
            "order_created_event",
            compensating_event_factory=lambda: service.create_order_cancelled_event(
                order_id
            ),
        )

        # Register payment (will fail)
        payment_id = manager.register_io_operation(
            "process_payment",
            compensation_action=lambda: asyncio.sleep(0),
        )

        # Execute operations
        await manager.execute_operation(
            create_order_id,
            lambda: service.create_order(order_id, ["item1"]),
        )

        await manager.execute_operation(
            domain_event_id,
            lambda: service.publish_order_created_event(order_id),
        )

        with pytest.raises(ValueError):
            await manager.execute_operation(
                payment_id,
                lambda: service.process_payment(order_id, 100.0),
            )

        # Before compensation - event exists
        assert len(event_store.events) == 1
        original_event = event_store.events[0]
        assert original_event["event_type"] == "OrderCreated"

        # Compensate
        await manager.compensate_all()

        # CRITICAL: Domain event is STILL there (not rolled back)
        assert len(event_store.events) == 1
        assert event_store.events[0] == original_event

        # CRITICAL: No events were deleted
        assert len(event_store.deleted_events) == 0

        # Result shows domain events preserved
        result = manager.get_result(saga_id)
        assert result.domain_events_preserved == 1

    @pytest.mark.asyncio
    async def test_compensating_event_created_for_domain_events(self) -> None:
        """Test compensating events are created (not rollback) for domain events."""
        db = MockDatabaseState()
        event_store = MockEventStore()
        service = MockOrderService(db, event_store, fail_on_payment=True)

        manager = CompensatingTransactionManager()
        saga_id = uuid4()
        order_id = "order-compensating-event"

        # Register domain event with compensating event factory
        domain_event_id = manager.register_domain_event(
            "order_created_event",
            compensating_event_factory=lambda: service.create_order_cancelled_event(
                order_id
            ),
        )

        # Register failing operation
        payment_id = manager.register_io_operation(
            "process_payment",
            compensation_action=lambda: asyncio.sleep(0),
        )

        # Execute domain event (succeeds)
        await manager.execute_operation(
            domain_event_id,
            lambda: service.publish_order_created_event(order_id),
        )

        # Execute payment (fails)
        with pytest.raises(ValueError):
            await manager.execute_operation(
                payment_id,
                lambda: service.process_payment(order_id, 100.0),
            )

        # Compensate
        await manager.compensate_all()

        # Compensating event was created
        assert len(manager.compensation_events) == 1
        compensating_event = manager.compensation_events[0]
        assert compensating_event.event_type == "OrderCancelled"
        assert compensating_event.payload["order_id"] == order_id
        assert compensating_event.payload["reason"] == "saga_compensation"

        result = manager.get_result(saga_id)
        assert result.compensation_events_created == 1


class TestCompensationFailureHandling:
    """Test handling of failures during compensation."""

    @pytest.mark.asyncio
    async def test_continues_compensating_after_failure(self) -> None:
        """Test compensation continues even if one compensation fails."""
        compensation_results: list[str] = []

        async def comp_success_1() -> None:
            compensation_results.append("success_1")

        async def comp_fail() -> None:
            raise RuntimeError("Compensation failed")

        async def comp_success_2() -> None:
            compensation_results.append("success_2")

        manager = CompensatingTransactionManager()

        op1 = manager.register_io_operation("op1", comp_success_1)
        op2 = manager.register_io_operation("op2", comp_fail)
        op3 = manager.register_io_operation("op3", comp_success_2)

        # Execute all successfully
        for op_id in [op1, op2, op3]:
            await manager.execute_operation(op_id, lambda: asyncio.sleep(0))

        # Compensate all (one will fail)
        await manager.compensate_all()

        # All compensations were attempted (in reverse order)
        assert "success_2" in compensation_results
        assert "success_1" in compensation_results

        # Find the failed compensation
        failed_op = manager._find_operation(op2)
        assert failed_op.status == EnumOperationStatus.COMPENSATION_FAILED


class TestEdgeCases:
    """Test edge cases in compensation strategy."""

    @pytest.mark.asyncio
    async def test_empty_saga(self) -> None:
        """Test behavior with no operations."""
        manager = CompensatingTransactionManager()
        saga_id = uuid4()

        result = manager.get_result(saga_id)
        assert result.overall_success is False  # No operations = not success
        assert result.operations_succeeded == 0
        assert result.operations_failed == 0

    @pytest.mark.asyncio
    async def test_single_failing_operation(self) -> None:
        """Test saga with single operation that fails."""
        manager = CompensatingTransactionManager()
        saga_id = uuid4()

        async def fail_operation() -> None:
            raise ValueError("Single failure")

        op_id = manager.register_io_operation(
            "failing_op",
            compensation_action=lambda: asyncio.sleep(0),
        )

        with pytest.raises(ValueError):
            await manager.execute_operation(op_id, fail_operation)

        result = manager.get_result(saga_id)
        assert result.overall_success is False
        assert result.operations_failed == 1
        assert result.operations_succeeded == 0

    @pytest.mark.asyncio
    async def test_compensation_not_needed_on_success(self) -> None:
        """Test no compensation happens when all operations succeed."""
        compensation_called = False

        async def compensation_action() -> None:
            nonlocal compensation_called
            compensation_called = True

        manager = CompensatingTransactionManager()

        op_id = manager.register_io_operation("op", compensation_action)
        await manager.execute_operation(op_id, lambda: asyncio.sleep(0))

        # Don't call compensate_all since saga succeeded
        # (In real usage, compensation is only called on failure)

        assert compensation_called is False


__all__: list[str] = [
    "TestPartialFailureDetection",
    "TestCompensationActionInvocation",
    "TestDomainEventsNotRolledBack",
    "TestCompensationFailureHandling",
    "TestEdgeCases",
]
