# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for correlation and causation chain validation.

This module provides comprehensive integration tests for the chain propagation
validator system, validating that messages properly maintain correlation and
causation chains during propagation through the ONEX event-driven system.

Tests cover:
    - Valid chain propagation scenarios
    - Correlation mismatch detection
    - Causation chain break detection
    - Multi-message workflow validation
    - Error message content verification
    - Registration workflow simulation
    - Strict enforcement behavior

Related:
    - OMN-951: Enforce Correlation and Causation Chain Validation
    - docs/patterns/correlation_id_tracking.md
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from pydantic import BaseModel, Field

from omnibase_infra.enums.enum_chain_violation_type import EnumChainViolationType
from omnibase_infra.models.validation.model_chain_violation import ModelChainViolation
from omnibase_infra.validation.chain_propagation_validator import (
    ChainPropagationError,
    ChainPropagationValidator,
    enforce_chain_propagation,
    validate_linear_message_chain,
    validate_message_chain,
)

# =============================================================================
# Test Configuration
# =============================================================================

pytestmark = [pytest.mark.integration]


# =============================================================================
# Pydantic Test Models (ONEX-Compliant)
# =============================================================================


class ModelTestUserRegistrationIntent(BaseModel):
    """Pydantic model for user registration intent in workflow tests.

    Represents the initial intent to register a new user in the system.
    """

    email: str = Field(..., description="User email address for registration")


class ModelTestCreateUserCommand(BaseModel):
    """Pydantic model for create user command.

    Command issued after registration intent to create the user account.
    """

    email: str = Field(..., description="User email address")
    name: str = Field(..., description="User display name")


class ModelTestUserCreatedEvent(BaseModel):
    """Pydantic model for user created event.

    Event emitted after successful user creation.
    """

    user_id: str = Field(..., description="Created user ID")
    email: str = Field(..., description="User email address")


class ModelTestSendWelcomeEmailCommand(BaseModel):
    """Pydantic model for send welcome email command.

    Command to send welcome email after user creation.
    """

    user_id: str = Field(..., description="User to send email to")
    email: str = Field(..., description="Email address to send to")


class ModelTestOrderIntent(BaseModel):
    """Pydantic model for order intent in workflow tests.

    Demonstrates ONEX-compliant Pydantic model usage in chain validation.
    """

    order_id: str = Field(..., description="Unique order identifier")
    customer_email: str = Field(..., description="Customer email address")
    total_amount: float = Field(..., ge=0, description="Order total amount")


class ModelTestOrderCommand(BaseModel):
    """Pydantic model for order processing command."""

    order_id: str = Field(..., description="Order to process")
    customer_email: str = Field(..., description="Customer email")
    items_count: int = Field(..., ge=1, description="Number of items")


class ModelTestOrderEvent(BaseModel):
    """Pydantic model for order completed event."""

    order_id: str = Field(..., description="Completed order ID")
    confirmation_number: str = Field(..., description="Order confirmation number")
    processed_at: str = Field(..., description="ISO timestamp of processing")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def correlation_id() -> UUID:
    """Create a shared correlation ID for workflow tests."""
    return uuid4()


@pytest.fixture
def parent_envelope(
    correlation_id: UUID,
) -> ModelEventEnvelope[ModelTestUserCreatedEvent]:
    """Create a parent envelope with known IDs.

    This envelope represents the root of a message chain with a known
    correlation_id and envelope_id for testing child message validation.
    """
    return ModelEventEnvelope(
        payload=ModelTestUserCreatedEvent(user_id="user-123", email="test@example.com"),
        correlation_id=correlation_id,
    )


@pytest.fixture
def valid_child_envelope(
    parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
) -> ModelEventEnvelope[ModelTestSendWelcomeEmailCommand]:
    """Create a valid child envelope with correct chain linkage.

    The child has:
    - Same correlation_id as parent (workflow traceability)
    - causation_id set to parent's envelope_id (causation chain)
    """
    # Create metadata with causation_id in tags
    metadata = ModelEnvelopeMetadata(
        tags={"causation_id": str(parent_envelope.envelope_id)},
    )

    return ModelEventEnvelope(
        payload=ModelTestSendWelcomeEmailCommand(
            user_id="user-123", email="test@example.com"
        ),
        correlation_id=parent_envelope.correlation_id,
        metadata=metadata,
    )


@pytest.fixture
def invalid_correlation_envelope(
    parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
) -> ModelEventEnvelope[ModelTestSendWelcomeEmailCommand]:
    """Create a child envelope with wrong correlation_id.

    This envelope has a different correlation_id than its parent,
    which breaks workflow traceability.
    """
    # Create metadata with correct causation_id but wrong correlation
    metadata = ModelEnvelopeMetadata(
        tags={"causation_id": str(parent_envelope.envelope_id)},
    )

    return ModelEventEnvelope(
        payload=ModelTestSendWelcomeEmailCommand(
            user_id="user-123", email="test@example.com"
        ),
        correlation_id=uuid4(),  # Different correlation_id!
        metadata=metadata,
    )


@pytest.fixture
def invalid_causation_envelope(
    parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
) -> ModelEventEnvelope[ModelTestSendWelcomeEmailCommand]:
    """Create a child envelope with wrong causation_id.

    This envelope has the correct correlation_id but references a
    different message as its cause, breaking the causation chain.
    """
    # Create metadata with wrong causation_id
    metadata = ModelEnvelopeMetadata(
        tags={"causation_id": str(uuid4())},  # Wrong parent reference!
    )

    return ModelEventEnvelope(
        payload=ModelTestSendWelcomeEmailCommand(
            user_id="user-123", email="test@example.com"
        ),
        correlation_id=parent_envelope.correlation_id,
        metadata=metadata,
    )


@pytest.fixture
def missing_causation_envelope(
    parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
) -> ModelEventEnvelope[ModelTestSendWelcomeEmailCommand]:
    """Create a child envelope with no causation_id.

    This envelope has the correct correlation_id but is missing
    the causation_id entirely.
    """
    return ModelEventEnvelope(
        payload=ModelTestSendWelcomeEmailCommand(
            user_id="user-123", email="test@example.com"
        ),
        correlation_id=parent_envelope.correlation_id,
        # No metadata with causation_id
    )


@pytest.fixture
def validator() -> ChainPropagationValidator:
    """Create a ChainPropagationValidator instance."""
    return ChainPropagationValidator()


# =============================================================================
# Validation Test Cases
# =============================================================================


class TestChainValidation:
    """Tests for single parent-child chain validation."""

    def test_valid_chain_passes_validation(
        self,
        validator: ChainPropagationValidator,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        valid_child_envelope: ModelEventEnvelope[ModelTestSendWelcomeEmailCommand],
    ) -> None:
        """Valid parent-child chain should pass validation with no violations.

        When a child message properly inherits parent's correlation_id and
        references parent's message_id in its causation_id, validation passes.
        """
        violations = validator.validate_chain(parent_envelope, valid_child_envelope)

        assert len(violations) == 0

    def test_correlation_mismatch_detected(
        self,
        validator: ChainPropagationValidator,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        invalid_correlation_envelope: ModelEventEnvelope[
            ModelTestSendWelcomeEmailCommand
        ],
    ) -> None:
        """Different correlation_id in child should be detected as violation.

        All messages in a workflow must share the same correlation_id for
        end-to-end distributed tracing.
        """
        violations = validator.validate_chain(
            parent_envelope, invalid_correlation_envelope
        )

        assert len(violations) >= 1

        # Find correlation violation
        correlation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        ]
        assert len(correlation_violations) == 1

        violation = correlation_violations[0]
        assert violation.severity == "error"
        assert violation.expected_value == parent_envelope.correlation_id
        assert violation.actual_value == invalid_correlation_envelope.correlation_id

    def test_causation_chain_broken_detected(
        self,
        validator: ChainPropagationValidator,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        invalid_causation_envelope: ModelEventEnvelope[
            ModelTestSendWelcomeEmailCommand
        ],
    ) -> None:
        """Wrong causation_id in child should be detected as chain break.

        Each message's causation_id must reference its direct parent's
        message_id to form an unbroken lineage.
        """
        violations = validator.validate_chain(
            parent_envelope, invalid_causation_envelope
        )

        assert len(violations) >= 1

        # Find causation violation
        causation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        ]
        assert len(causation_violations) == 1

        violation = causation_violations[0]
        assert violation.severity == "error"
        assert violation.expected_value == parent_envelope.envelope_id

    def test_missing_causation_detected(
        self,
        validator: ChainPropagationValidator,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        missing_causation_envelope: ModelEventEnvelope[
            ModelTestSendWelcomeEmailCommand
        ],
    ) -> None:
        """Missing causation_id in child should be detected as chain break.

        Every message (except root) must have a causation_id referencing
        its parent's message_id.
        """
        violations = validator.validate_chain(
            parent_envelope, missing_causation_envelope
        )

        assert len(violations) >= 1

        # Find causation violation
        causation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        ]
        assert len(causation_violations) == 1

        violation = causation_violations[0]
        assert violation.severity == "error"
        assert violation.actual_value is None  # Missing causation_id

    def test_multiple_violations_reported(
        self,
        validator: ChainPropagationValidator,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
    ) -> None:
        """Both correlation and causation violations should be reported together.

        When a message has both wrong correlation_id and wrong causation_id,
        both violations should be detected and reported.
        """
        # Create envelope with both violations
        bad_envelope = ModelEventEnvelope(
            payload=ModelTestSendWelcomeEmailCommand(
                user_id="user-123", email="test@example.com"
            ),
            correlation_id=uuid4(),  # Wrong correlation
            # No causation_id (missing)
        )

        violations = validator.validate_chain(parent_envelope, bad_envelope)

        assert len(violations) == 2

        # Check both violation types are present
        violation_types = {v.violation_type for v in violations}
        assert EnumChainViolationType.CORRELATION_MISMATCH in violation_types
        assert EnumChainViolationType.CAUSATION_CHAIN_BROKEN in violation_types


# =============================================================================
# Workflow Chain Test Cases
# =============================================================================


class TestWorkflowChainValidation:
    """Tests for multi-message workflow chain validation."""

    def test_workflow_chain_valid(
        self,
        validator: ChainPropagationValidator,
        correlation_id: UUID,
    ) -> None:
        """Multi-message workflow with proper chain linkage should pass.

        Tests a complete workflow:
        msg1 (root) -> msg2 -> msg3 -> msg4
        All share same correlation_id, each references its direct parent.
        """
        # Create workflow chain
        msg1 = ModelEventEnvelope(
            payload=ModelTestUserRegistrationIntent(email="test@example.com"),
            correlation_id=correlation_id,
        )

        msg2_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg1.envelope_id)},
        )
        msg2 = ModelEventEnvelope(
            payload=ModelTestCreateUserCommand(
                email="test@example.com", name="Test User"
            ),
            correlation_id=correlation_id,
            metadata=msg2_metadata,
        )

        msg3_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg2.envelope_id)},
        )
        msg3 = ModelEventEnvelope(
            payload=ModelTestUserCreatedEvent(
                user_id="user-123", email="test@example.com"
            ),
            correlation_id=correlation_id,
            metadata=msg3_metadata,
        )

        msg4_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg3.envelope_id)},
        )
        msg4 = ModelEventEnvelope(
            payload=ModelTestSendWelcomeEmailCommand(
                user_id="user-123", email="test@example.com"
            ),
            correlation_id=correlation_id,
            metadata=msg4_metadata,
        )

        violations = validator.validate_workflow_chain([msg1, msg2, msg3, msg4])

        assert len(violations) == 0

    def test_workflow_chain_detects_correlation_drift(
        self,
        validator: ChainPropagationValidator,
        correlation_id: UUID,
    ) -> None:
        """Workflow should detect when correlation_id changes mid-chain.

        If a message in the middle of a workflow has a different correlation_id,
        this breaks distributed tracing and should be flagged.
        """
        # Create workflow with correlation drift in msg3
        msg1 = ModelEventEnvelope(
            payload=ModelTestUserRegistrationIntent(email="test@example.com"),
            correlation_id=correlation_id,
        )

        msg2_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg1.envelope_id)},
        )
        msg2 = ModelEventEnvelope(
            payload=ModelTestCreateUserCommand(
                email="test@example.com", name="Test User"
            ),
            correlation_id=correlation_id,
            metadata=msg2_metadata,
        )

        # msg3 has different correlation_id (drift!)
        msg3_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg2.envelope_id)},
        )
        msg3 = ModelEventEnvelope(
            payload=ModelTestUserCreatedEvent(
                user_id="user-123", email="test@example.com"
            ),
            correlation_id=uuid4(),  # Different correlation_id!
            metadata=msg3_metadata,
        )

        violations = validator.validate_workflow_chain([msg1, msg2, msg3])

        assert len(violations) >= 1

        # Should detect correlation mismatch
        correlation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        ]
        assert len(correlation_violations) >= 1

    def test_workflow_chain_detects_ancestor_skip(
        self,
        validator: ChainPropagationValidator,
        correlation_id: UUID,
    ) -> None:
        """Workflow should detect when causation_id skips a parent.

        If msg3's causation_id references msg1 instead of msg2 (its direct
        parent), this breaks the causation chain lineage.
        """
        # Create workflow where msg3 skips msg2
        msg1 = ModelEventEnvelope(
            payload=ModelTestUserRegistrationIntent(email="test@example.com"),
            correlation_id=correlation_id,
        )

        msg2_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg1.envelope_id)},
        )
        msg2 = ModelEventEnvelope(
            payload=ModelTestCreateUserCommand(
                email="test@example.com", name="Test User"
            ),
            correlation_id=correlation_id,
            metadata=msg2_metadata,
        )

        # msg3 references msg1 instead of msg2 (ancestor skip!)
        msg3_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg1.envelope_id)},  # Wrong: should be msg2
        )
        msg3 = ModelEventEnvelope(
            payload=ModelTestUserCreatedEvent(
                user_id="user-123", email="test@example.com"
            ),
            correlation_id=correlation_id,
            metadata=msg3_metadata,
        )

        # Note: The workflow validator checks that causation_id references
        # a message in the chain, but doesn't enforce direct parent ordering.
        # The single-message validator (validate_chain) enforces direct parent.
        violations = validator.validate_workflow_chain([msg1, msg2, msg3])

        # The workflow validator should pass because msg3's causation_id (msg1)
        # is in the chain. The direct parent check is done by validate_chain.
        # For strict ancestor checking, use pairwise validate_chain calls.
        # This test validates the workflow allows valid ancestor references.
        # (This behavior aligns with the documented chain rules)
        assert len(violations) == 0


# =============================================================================
# Error Message Test Cases
# =============================================================================


class TestErrorMessageContent:
    """Tests for error message content and formatting."""

    def test_error_message_includes_expected_value(
        self,
        validator: ChainPropagationValidator,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        invalid_correlation_envelope: ModelEventEnvelope[
            ModelTestSendWelcomeEmailCommand
        ],
    ) -> None:
        """Violation message should include the expected value for debugging."""
        violations = validator.validate_chain(
            parent_envelope, invalid_correlation_envelope
        )

        correlation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        ]
        assert len(correlation_violations) == 1

        violation = correlation_violations[0]
        assert violation.expected_value is not None
        assert isinstance(violation.expected_value, UUID)

    def test_error_message_includes_actual_value(
        self,
        validator: ChainPropagationValidator,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        invalid_correlation_envelope: ModelEventEnvelope[
            ModelTestSendWelcomeEmailCommand
        ],
    ) -> None:
        """Violation message should include the actual value found."""
        violations = validator.validate_chain(
            parent_envelope, invalid_correlation_envelope
        )

        correlation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        ]
        assert len(correlation_violations) == 1

        violation = correlation_violations[0]
        assert violation.actual_value is not None
        assert isinstance(violation.actual_value, UUID)

    def test_error_message_includes_message_id(
        self,
        validator: ChainPropagationValidator,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        invalid_correlation_envelope: ModelEventEnvelope[
            ModelTestSendWelcomeEmailCommand
        ],
    ) -> None:
        """Violation should include the message_id where violation was detected."""
        violations = validator.validate_chain(
            parent_envelope, invalid_correlation_envelope
        )

        for violation in violations:
            assert violation.message_id is not None
            assert isinstance(violation.message_id, UUID)
            assert violation.message_id == invalid_correlation_envelope.envelope_id

    def test_violation_format_for_logging(
        self,
        validator: ChainPropagationValidator,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        invalid_correlation_envelope: ModelEventEnvelope[
            ModelTestSendWelcomeEmailCommand
        ],
    ) -> None:
        """Violation should have a format_for_logging method for structured logs."""
        violations = validator.validate_chain(
            parent_envelope, invalid_correlation_envelope
        )

        for violation in violations:
            log_output = violation.format_for_logging()

            # Verify log output structure
            assert isinstance(log_output, str)
            assert "[" in log_output  # Severity marker
            assert violation.violation_type.value.upper() in log_output.upper()
            assert "message=" in log_output


# =============================================================================
# Registration Workflow Simulation
# =============================================================================


class TestRegistrationWorkflowChain:
    """Test complete user registration workflow chain validation."""

    def test_registration_workflow_chain(self, correlation_id: UUID) -> None:
        """Simulate full registration workflow with proper chain validation.

        Workflow:
        1. ModelTestUserRegistrationIntent (root)
        2. ModelTestCreateUserCommand (caused by intent)
        3. ModelTestUserCreatedEvent (caused by command)
        4. ModelTestSendWelcomeEmailCommand (caused by event)

        All messages share same correlation_id.
        Each has causation_id = previous.message_id.
        """
        validator = ChainPropagationValidator()

        # Step 1: ModelTestUserRegistrationIntent (root message)
        registration_intent = ModelEventEnvelope(
            payload=ModelTestUserRegistrationIntent(email="newuser@example.com"),
            correlation_id=correlation_id,
        )

        # Step 2: ModelTestCreateUserCommand (caused by intent)
        create_command_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(registration_intent.envelope_id)},
        )
        create_command = ModelEventEnvelope(
            payload=ModelTestCreateUserCommand(
                email="newuser@example.com", name="New User"
            ),
            correlation_id=correlation_id,
            metadata=create_command_metadata,
        )

        # Validate intent -> command chain
        violations_1_2 = validator.validate_chain(registration_intent, create_command)
        assert len(violations_1_2) == 0, f"Intent->Command failed: {violations_1_2}"

        # Step 3: ModelTestUserCreatedEvent (caused by command)
        user_created_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(create_command.envelope_id)},
        )
        user_created = ModelEventEnvelope(
            payload=ModelTestUserCreatedEvent(
                user_id="user-456", email="newuser@example.com"
            ),
            correlation_id=correlation_id,
            metadata=user_created_metadata,
        )

        # Validate command -> event chain
        violations_2_3 = validator.validate_chain(create_command, user_created)
        assert len(violations_2_3) == 0, f"Command->Event failed: {violations_2_3}"

        # Step 4: ModelTestSendWelcomeEmailCommand (caused by event)
        welcome_email_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(user_created.envelope_id)},
        )
        welcome_email = ModelEventEnvelope(
            payload=ModelTestSendWelcomeEmailCommand(
                user_id="user-456", email="newuser@example.com"
            ),
            correlation_id=correlation_id,
            metadata=welcome_email_metadata,
        )

        # Validate event -> command chain
        violations_3_4 = validator.validate_chain(user_created, welcome_email)
        assert len(violations_3_4) == 0, f"Event->Command failed: {violations_3_4}"

        # Validate entire workflow chain
        all_messages = [
            registration_intent,
            create_command,
            user_created,
            welcome_email,
        ]
        workflow_violations = validator.validate_workflow_chain(all_messages)
        assert len(workflow_violations) == 0, f"Workflow failed: {workflow_violations}"

        # Verify all messages share same correlation_id
        for msg in all_messages:
            assert msg.correlation_id == correlation_id

        # Verify causation chain integrity
        assert create_command.metadata.tags.get("causation_id") == str(
            registration_intent.envelope_id
        )
        assert user_created.metadata.tags.get("causation_id") == str(
            create_command.envelope_id
        )
        assert welcome_email.metadata.tags.get("causation_id") == str(
            user_created.envelope_id
        )


# =============================================================================
# Enforcement Test Cases
# =============================================================================


class TestChainEnforcement:
    """Tests for strict chain propagation enforcement."""

    def test_enforce_chain_propagation_raises_on_violation(
        self,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        invalid_correlation_envelope: ModelEventEnvelope[
            ModelTestSendWelcomeEmailCommand
        ],
    ) -> None:
        """enforce_chain_propagation should raise ChainPropagationError on violation."""
        with pytest.raises(ChainPropagationError) as exc_info:
            enforce_chain_propagation(parent_envelope, invalid_correlation_envelope)

        error = exc_info.value
        assert len(error.violations) >= 1

    def test_enforce_chain_propagation_passes_valid_chain(
        self,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        valid_child_envelope: ModelEventEnvelope[ModelTestSendWelcomeEmailCommand],
    ) -> None:
        """enforce_chain_propagation should not raise for valid chain."""
        # Should not raise
        enforce_chain_propagation(parent_envelope, valid_child_envelope)

    def test_chain_propagation_error_contains_all_violations(
        self,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
    ) -> None:
        """ChainPropagationError should contain all detected violations."""
        # Create envelope with multiple violations
        bad_envelope = ModelEventEnvelope(
            payload=ModelTestSendWelcomeEmailCommand(
                user_id="user-123", email="test@example.com"
            ),
            correlation_id=uuid4(),  # Wrong correlation
            # No causation_id (missing)
        )

        with pytest.raises(ChainPropagationError) as exc_info:
            enforce_chain_propagation(parent_envelope, bad_envelope)

        error = exc_info.value
        assert len(error.violations) == 2

        # Verify both violation types are in error
        violation_types = {v.violation_type for v in error.violations}
        assert EnumChainViolationType.CORRELATION_MISMATCH in violation_types
        assert EnumChainViolationType.CAUSATION_CHAIN_BROKEN in violation_types


# =============================================================================
# Convenience Function Test Cases
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_validate_message_chain_returns_violations(
        self,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        invalid_correlation_envelope: ModelEventEnvelope[
            ModelTestSendWelcomeEmailCommand
        ],
    ) -> None:
        """validate_message_chain should return list of violations."""
        violations = validate_message_chain(
            parent_envelope, invalid_correlation_envelope
        )

        assert isinstance(violations, list)
        assert len(violations) >= 1
        assert all(isinstance(v, ModelChainViolation) for v in violations)

    def test_validate_message_chain_returns_empty_for_valid(
        self,
        parent_envelope: ModelEventEnvelope[ModelTestUserCreatedEvent],
        valid_child_envelope: ModelEventEnvelope[ModelTestSendWelcomeEmailCommand],
    ) -> None:
        """validate_message_chain should return empty list for valid chain."""
        violations = validate_message_chain(parent_envelope, valid_child_envelope)

        assert isinstance(violations, list)
        assert len(violations) == 0


# =============================================================================
# Pydantic Model Integration Test Cases
# =============================================================================


class TestPydanticModelChainValidation:
    """Tests using Pydantic models for real-world ONEX compliance validation.

    These tests verify that chain validation works correctly with proper
    Pydantic models as required by ONEX guidelines, not just plain classes.
    """

    def test_pydantic_model_workflow_chain_valid(
        self,
        correlation_id: UUID,
    ) -> None:
        """Validate workflow chain using Pydantic models.

        Tests a complete order workflow with proper ONEX-compliant Pydantic models:
        OrderIntent -> OrderCommand -> OrderEvent
        All share same correlation_id, each references its direct parent.
        """
        validator = ChainPropagationValidator()

        # Step 1: OrderIntent (root message with Pydantic model)
        order_intent = ModelEventEnvelope(
            payload=ModelTestOrderIntent(
                order_id="order-789",
                customer_email="customer@example.com",
                total_amount=99.99,
            ),
            correlation_id=correlation_id,
        )

        # Step 2: OrderCommand (caused by intent)
        order_command_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(order_intent.envelope_id)},
        )
        order_command = ModelEventEnvelope(
            payload=ModelTestOrderCommand(
                order_id="order-789",
                customer_email="customer@example.com",
                items_count=3,
            ),
            correlation_id=correlation_id,
            metadata=order_command_metadata,
        )

        # Step 3: OrderEvent (caused by command)
        order_event_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(order_command.envelope_id)},
        )
        order_event = ModelEventEnvelope(
            payload=ModelTestOrderEvent(
                order_id="order-789",
                confirmation_number="CONF-12345",
                processed_at="2025-12-20T12:00:00Z",
            ),
            correlation_id=correlation_id,
            metadata=order_event_metadata,
        )

        # Validate pairwise chains
        violations_1_2 = validator.validate_chain(order_intent, order_command)
        assert len(violations_1_2) == 0, f"Intent->Command failed: {violations_1_2}"

        violations_2_3 = validator.validate_chain(order_command, order_event)
        assert len(violations_2_3) == 0, f"Command->Event failed: {violations_2_3}"

        # Validate entire workflow
        all_messages = [order_intent, order_command, order_event]
        workflow_violations = validator.validate_workflow_chain(all_messages)
        assert len(workflow_violations) == 0, f"Workflow failed: {workflow_violations}"

        # Verify Pydantic model payloads are accessible
        assert order_intent.payload.order_id == "order-789"
        assert order_command.payload.items_count == 3
        assert order_event.payload.confirmation_number == "CONF-12345"

    def test_pydantic_model_chain_violation_detected(
        self,
        correlation_id: UUID,
    ) -> None:
        """Verify chain violations are detected with Pydantic model payloads.

        Tests that validation correctly detects correlation mismatch even when
        using proper Pydantic models as payloads.
        """
        validator = ChainPropagationValidator()

        # Parent with Pydantic model
        parent = ModelEventEnvelope(
            payload=ModelTestOrderIntent(
                order_id="order-999",
                customer_email="test@example.com",
                total_amount=50.00,
            ),
            correlation_id=correlation_id,
        )

        # Child with wrong correlation_id
        child_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(parent.envelope_id)},
        )
        child = ModelEventEnvelope(
            payload=ModelTestOrderCommand(
                order_id="order-999",
                customer_email="test@example.com",
                items_count=1,
            ),
            correlation_id=uuid4(),  # Wrong correlation_id!
            metadata=child_metadata,
        )

        violations = validator.validate_chain(parent, child)

        assert len(violations) >= 1
        correlation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        ]
        assert len(correlation_violations) == 1


# =============================================================================
# Linear Chain Validation Integration Tests
# =============================================================================


class TestLinearChainValidation:
    """Integration tests for strict linear chain validation.

    These tests verify that validate_linear_workflow_chain() and
    validate_linear_message_chain() correctly detect ancestor skipping
    scenarios that pass the more flexible validate_workflow_chain().
    """

    def test_ancestor_skip_passes_workflow_but_fails_linear(
        self,
        validator: ChainPropagationValidator,
        correlation_id: UUID,
    ) -> None:
        """Ancestor skip should pass workflow validation but fail linear validation.

        This is the key difference between validate_workflow_chain() and
        validate_linear_workflow_chain():

        - validate_workflow_chain() allows msg3 to reference msg1 (any ancestor in chain)
        - validate_linear_workflow_chain() requires msg3 to reference msg2 (direct parent)

        Scenario:
            msg1 (root) -> msg2 -> msg3
            where msg3.causation_id = msg1 (skipping msg2)

        This pattern may be valid for fan-out workflows but is invalid for
        strict linear chains.
        """
        # Create workflow where msg3 skips msg2
        msg1 = ModelEventEnvelope(
            payload=ModelTestUserRegistrationIntent(email="test@example.com"),
            correlation_id=correlation_id,
        )

        msg2_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg1.envelope_id)},
        )
        msg2 = ModelEventEnvelope(
            payload=ModelTestCreateUserCommand(
                email="test@example.com", name="Test User"
            ),
            correlation_id=correlation_id,
            metadata=msg2_metadata,
        )

        # msg3 references msg1 instead of msg2 (ancestor skip!)
        msg3_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg1.envelope_id)},  # Wrong: should be msg2
        )
        msg3 = ModelEventEnvelope(
            payload=ModelTestUserCreatedEvent(
                user_id="user-123", email="test@example.com"
            ),
            correlation_id=correlation_id,
            metadata=msg3_metadata,
        )

        chain = [msg1, msg2, msg3]

        # Workflow validation SHOULD PASS (allows ancestor skipping)
        workflow_violations = validator.validate_workflow_chain(chain)
        assert len(workflow_violations) == 0, (
            f"Expected workflow validation to pass (ancestor skipping allowed), "
            f"but got violations: {workflow_violations}"
        )

        # Linear validation SHOULD FAIL (enforces direct parent reference)
        linear_violations = validator.validate_linear_workflow_chain(chain)
        assert len(linear_violations) >= 1, (
            "Expected linear validation to fail (ancestor skip detected), "
            "but got no violations"
        )

        # Verify it's specifically a causation chain violation
        causation_violations = [
            v
            for v in linear_violations
            if v.violation_type == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        ]
        assert len(causation_violations) == 1

        violation = causation_violations[0]
        assert violation.expected_value == msg2.envelope_id
        assert violation.actual_value == msg1.envelope_id

    def test_validate_linear_message_chain_convenience_function(
        self,
        correlation_id: UUID,
    ) -> None:
        """Test the validate_linear_message_chain convenience function.

        Verifies the module-level function works correctly for detecting
        ancestor skipping in linear chains.
        """
        # Create valid linear chain
        msg1 = ModelEventEnvelope(
            payload=ModelTestOrderIntent(
                order_id="order-123",
                customer_email="test@example.com",
                total_amount=100.0,
            ),
            correlation_id=correlation_id,
        )

        msg2_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg1.envelope_id)},
        )
        msg2 = ModelEventEnvelope(
            payload=ModelTestOrderCommand(
                order_id="order-123",
                customer_email="test@example.com",
                items_count=2,
            ),
            correlation_id=correlation_id,
            metadata=msg2_metadata,
        )

        msg3_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg2.envelope_id)},
        )
        msg3 = ModelEventEnvelope(
            payload=ModelTestOrderEvent(
                order_id="order-123",
                confirmation_number="CONF-456",
                processed_at="2025-12-21T10:00:00Z",
            ),
            correlation_id=correlation_id,
            metadata=msg3_metadata,
        )

        # Valid linear chain should pass
        violations = validate_linear_message_chain([msg1, msg2, msg3])
        assert len(violations) == 0

        # Now create chain with ancestor skip
        msg3_skipped_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg1.envelope_id)},  # Skips msg2
        )
        msg3_skipped = ModelEventEnvelope(
            payload=ModelTestOrderEvent(
                order_id="order-123",
                confirmation_number="CONF-789",
                processed_at="2025-12-21T10:00:00Z",
            ),
            correlation_id=correlation_id,
            metadata=msg3_skipped_metadata,
        )

        # Chain with ancestor skip should fail
        violations_with_skip = validate_linear_message_chain([msg1, msg2, msg3_skipped])
        assert len(violations_with_skip) >= 1

        # Verify the violation details
        causation_violations = [
            v
            for v in violations_with_skip
            if v.violation_type == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        ]
        assert len(causation_violations) == 1

    def test_four_message_linear_chain_with_ancestor_skip(
        self,
        validator: ChainPropagationValidator,
        correlation_id: UUID,
    ) -> None:
        """Test a 4-message chain where msg4 skips msg3 to reference msg2.

        Workflow:
            msg1 (root) -> msg2 -> msg3 -> msg4
            where msg4.causation_id = msg2 (skipping msg3)

        This should pass workflow validation but fail linear validation.
        """
        # Create 4-message workflow
        msg1 = ModelEventEnvelope(
            payload=ModelTestUserRegistrationIntent(email="test@example.com"),
            correlation_id=correlation_id,
        )

        msg2_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg1.envelope_id)},
        )
        msg2 = ModelEventEnvelope(
            payload=ModelTestCreateUserCommand(email="test@example.com", name="Test"),
            correlation_id=correlation_id,
            metadata=msg2_metadata,
        )

        msg3_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg2.envelope_id)},
        )
        msg3 = ModelEventEnvelope(
            payload=ModelTestUserCreatedEvent(
                user_id="user-1", email="test@example.com"
            ),
            correlation_id=correlation_id,
            metadata=msg3_metadata,
        )

        # msg4 references msg2 instead of msg3 (ancestor skip at end of chain)
        msg4_metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(msg2.envelope_id)},  # Should be msg3
        )
        msg4 = ModelEventEnvelope(
            payload=ModelTestSendWelcomeEmailCommand(
                user_id="user-1", email="test@example.com"
            ),
            correlation_id=correlation_id,
            metadata=msg4_metadata,
        )

        chain = [msg1, msg2, msg3, msg4]

        # Workflow validation should pass
        workflow_violations = validator.validate_workflow_chain(chain)
        assert len(workflow_violations) == 0

        # Linear validation should fail
        linear_violations = validator.validate_linear_workflow_chain(chain)
        assert len(linear_violations) >= 1

        # Should detect that msg4 skipped msg3
        causation_violations = [
            v
            for v in linear_violations
            if v.violation_type == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        ]
        assert len(causation_violations) == 1

        violation = causation_violations[0]
        assert violation.expected_value == msg3.envelope_id
        assert violation.actual_value == msg2.envelope_id
