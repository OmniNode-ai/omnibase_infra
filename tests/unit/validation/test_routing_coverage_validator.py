# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for the Routing Coverage Validator.

Validates that:
- Message type discovery correctly identifies Event, Command, Intent, Projection classes
- Route registration discovery identifies registered routes
- RoutingCoverageValidator accurately computes coverage
- Startup fail-fast integration works correctly
- CI gate returns appropriate results and violations
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.enums.enum_execution_shape_violation import (
    EnumExecutionShapeViolation,
)
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
from omnibase_infra.validation.routing_coverage_validator import (
    RoutingCoverageError,
    RoutingCoverageValidator,
    check_routing_coverage_ci,
    discover_message_types,
    discover_registered_routes,
    validate_routing_coverage_on_startup,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_source_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test source files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_event_file(temp_source_dir: Path) -> Path:
    """Create a sample file with Event classes."""
    file_path = temp_source_dir / "models" / "events.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        '''
"""Sample event classes for testing."""

from pydantic import BaseModel


class OrderCreatedEvent(BaseModel):
    """Event fired when an order is created."""
    order_id: str
    customer_id: str


class PaymentReceivedEvent(BaseModel):
    """Event fired when payment is received."""
    payment_id: str
    amount: float


class UserRegisteredEvent(BaseModel):
    """Event fired when a user registers."""
    user_id: str
    email: str
'''
    )
    return file_path


@pytest.fixture
def sample_command_file(temp_source_dir: Path) -> Path:
    """Create a sample file with Command classes."""
    file_path = temp_source_dir / "models" / "commands.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        '''
"""Sample command classes for testing."""

from pydantic import BaseModel


class CreateOrderCommand(BaseModel):
    """Command to create an order."""
    customer_id: str
    items: list[str]


class ProcessPaymentCommand(BaseModel):
    """Command to process a payment."""
    order_id: str
    payment_method: str
'''
    )
    return file_path


@pytest.fixture
def sample_intent_file(temp_source_dir: Path) -> Path:
    """Create a sample file with Intent classes."""
    file_path = temp_source_dir / "models" / "intents.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        '''
"""Sample intent classes for testing."""

from pydantic import BaseModel


class CheckoutIntent(BaseModel):
    """User intent to checkout cart."""
    cart_id: str
    user_id: str


class SubscriptionIntent(BaseModel):
    """User intent to subscribe to a plan."""
    plan_id: str
    user_id: str
'''
    )
    return file_path


@pytest.fixture
def sample_projection_file(temp_source_dir: Path) -> Path:
    """Create a sample file with Projection classes."""
    file_path = temp_source_dir / "models" / "projections.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        '''
"""Sample projection classes for testing."""

from pydantic import BaseModel


class OrderSummaryProjection(BaseModel):
    """Projection for order summary read model."""
    order_id: str
    total_amount: float
    status: str


class UserProfileProjection(BaseModel):
    """Projection for user profile read model."""
    user_id: str
    display_name: str
'''
    )
    return file_path


@pytest.fixture
def sample_registration_file(temp_source_dir: Path) -> Path:
    """Create a sample file with route registrations."""
    file_path = temp_source_dir / "routing" / "registrations.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        '''
"""Sample route registrations for testing."""


def register_routes(registry):
    """Register message type routes."""
    registry.register("OrderCreatedEvent", OrderEventHandler)
    registry.register("PaymentReceivedEvent", PaymentEventHandler)
    registry.register("CreateOrderCommand", CreateOrderHandler)

    # Handler map registration style
    handler_map["CheckoutIntent"] = CheckoutHandler

    # Subscribe style registration
    bus.subscribe("orders.events", "OrderSummaryProjection")
'''
    )
    return file_path


@pytest.fixture
def sample_decorator_file(temp_source_dir: Path) -> Path:
    """Create a sample file with decorator-based message types."""
    file_path = temp_source_dir / "models" / "decorated.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        '''
"""Sample decorated message types for testing."""

from pydantic import BaseModel


@message_type
class GenericMessage(BaseModel):
    """Generic message with decorator."""
    data: str


@event_type
class DecoratedEvent(BaseModel):
    """Decorated event class."""
    event_data: str


@command_type()
class DecoratedCommand(BaseModel):
    """Decorated command class."""
    command_data: str
'''
    )
    return file_path


@pytest.fixture
def sample_inheritance_file(temp_source_dir: Path) -> Path:
    """Create a sample file with inheritance-based message types."""
    file_path = temp_source_dir / "models" / "inheritance.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        '''
"""Sample inheritance-based message types for testing."""


class BaseEvent:
    """Base class for events."""
    pass


class ModelEvent:
    """Model event base class."""
    pass


class InheritedFromBaseEvent(BaseEvent):
    """Inherits from BaseEvent."""
    pass


class InheritedFromModelEvent(ModelEvent):
    """Inherits from ModelEvent."""
    pass
'''
    )
    return file_path


@pytest.fixture
def sample_mixed_file(temp_source_dir: Path) -> Path:
    """Create a sample file with mixed classes (not all message types)."""
    file_path = temp_source_dir / "models" / "mixed.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        '''
"""Sample file with mixed classes."""

from pydantic import BaseModel


class RegularModel(BaseModel):
    """Not a message type."""
    value: str


class MyService:
    """A service class, not a message type."""
    pass


class OrderCreatedEventHandler:
    """Handler class, not a message type (but ends in Event)."""
    # Note: This might be detected due to suffix pattern
    # Real-world code would exclude *Handler classes
    pass


class AnotherEvent(BaseModel):
    """This is a message type."""
    data: str
'''
    )
    return file_path


# =============================================================================
# Test RoutingCoverageError
# =============================================================================


class TestRoutingCoverageError:
    """Tests for RoutingCoverageError exception."""

    def test_error_message_with_unmapped_types(self) -> None:
        """Verify error message includes unmapped types."""
        error = RoutingCoverageError(
            unmapped_types={"OrderEvent", "PaymentCommand"},
            total_types=10,
            registered_types=8,
        )
        assert "OrderEvent" in str(error)
        assert "PaymentCommand" in str(error)
        assert "2 unmapped" in str(error)

    def test_error_coverage_percentage(self) -> None:
        """Verify coverage percentage is calculated correctly."""
        error = RoutingCoverageError(
            unmapped_types={"OrderEvent"},
            total_types=10,
            registered_types=9,
        )
        assert error.coverage_percent == 90.0

    def test_error_with_zero_total_types(self) -> None:
        """Verify error handles zero total types gracefully."""
        error = RoutingCoverageError(
            unmapped_types=set(),
            total_types=0,
            registered_types=0,
        )
        assert error.coverage_percent == 0.0

    def test_error_unmapped_types_attribute(self) -> None:
        """Verify unmapped_types attribute is set correctly."""
        unmapped = {"TypeA", "TypeB", "TypeC"}
        error = RoutingCoverageError(unmapped_types=unmapped)
        assert error.unmapped_types == unmapped

    def test_error_inherits_from_runtime_host_error(self) -> None:
        """Verify RoutingCoverageError is a RuntimeHostError."""
        from omnibase_infra.errors import RuntimeHostError

        error = RoutingCoverageError(unmapped_types={"Test"})
        assert isinstance(error, RuntimeHostError)


# =============================================================================
# Test Message Type Discovery
# =============================================================================


class TestDiscoverMessageTypes:
    """Tests for discover_message_types function."""

    def test_discover_event_classes(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify Event suffix classes are discovered."""
        types = discover_message_types(temp_source_dir)
        assert "OrderCreatedEvent" in types
        assert "PaymentReceivedEvent" in types
        assert "UserRegisteredEvent" in types
        assert types["OrderCreatedEvent"] == EnumMessageCategory.EVENT

    def test_discover_command_classes(
        self, temp_source_dir: Path, sample_command_file: Path
    ) -> None:
        """Verify Command suffix classes are discovered."""
        types = discover_message_types(temp_source_dir)
        assert "CreateOrderCommand" in types
        assert "ProcessPaymentCommand" in types
        assert types["CreateOrderCommand"] == EnumMessageCategory.COMMAND

    def test_discover_intent_classes(
        self, temp_source_dir: Path, sample_intent_file: Path
    ) -> None:
        """Verify Intent suffix classes are discovered."""
        types = discover_message_types(temp_source_dir)
        assert "CheckoutIntent" in types
        assert "SubscriptionIntent" in types
        assert types["CheckoutIntent"] == EnumMessageCategory.INTENT

    def test_discover_projection_classes(
        self, temp_source_dir: Path, sample_projection_file: Path
    ) -> None:
        """Verify Projection suffix classes are discovered.

        Note: PROJECTION uses EnumNodeOutputType (not EnumMessageCategory)
        because projections are node outputs, not routed messages.
        """
        types = discover_message_types(temp_source_dir)
        assert "OrderSummaryProjection" in types
        assert "UserProfileProjection" in types
        assert types["OrderSummaryProjection"] == EnumNodeOutputType.PROJECTION

    def test_discover_decorated_classes(
        self, temp_source_dir: Path, sample_decorator_file: Path
    ) -> None:
        """Verify decorated message types are discovered."""
        types = discover_message_types(temp_source_dir)
        # DecoratedEvent and DecoratedCommand have category-specific decorators
        assert "DecoratedEvent" in types
        assert "DecoratedCommand" in types
        assert types["DecoratedEvent"] == EnumMessageCategory.EVENT
        assert types["DecoratedCommand"] == EnumMessageCategory.COMMAND

    def test_discover_inherited_classes(
        self, temp_source_dir: Path, sample_inheritance_file: Path
    ) -> None:
        """Verify inheritance-based message types are discovered."""
        types = discover_message_types(temp_source_dir)
        assert "InheritedFromBaseEvent" in types
        assert "InheritedFromModelEvent" in types
        assert types["InheritedFromBaseEvent"] == EnumMessageCategory.EVENT
        assert types["InheritedFromModelEvent"] == EnumMessageCategory.EVENT

    def test_discover_excludes_test_files(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify test files are excluded by default."""
        # Create a test file
        test_file = temp_source_dir / "tests" / "test_events.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            '''
class TestEventInTestFile(BaseModel):
    """This should be excluded."""
    pass
'''
        )
        types = discover_message_types(temp_source_dir)
        assert "TestEventInTestFile" not in types

    def test_discover_empty_directory(self, temp_source_dir: Path) -> None:
        """Verify empty directory returns empty dict."""
        types = discover_message_types(temp_source_dir)
        assert types == {}

    def test_discover_with_syntax_error(self, temp_source_dir: Path) -> None:
        """Verify syntax errors are handled gracefully."""
        bad_file = temp_source_dir / "bad_syntax.py"
        bad_file.write_text("class MissingColon")  # Syntax error

        # Should not raise, just skip the file
        types = discover_message_types(temp_source_dir)
        assert isinstance(types, dict)


# =============================================================================
# Test Route Registration Discovery
# =============================================================================


class TestDiscoverRegisteredRoutes:
    """Tests for discover_registered_routes function."""

    def test_discover_registry_register_pattern(
        self, temp_source_dir: Path, sample_registration_file: Path
    ) -> None:
        """Verify registry.register() pattern is detected."""
        routes = discover_registered_routes(source_directory=temp_source_dir)
        assert "OrderCreatedEvent" in routes
        assert "PaymentReceivedEvent" in routes
        assert "CreateOrderCommand" in routes

    def test_discover_handler_map_pattern(
        self, temp_source_dir: Path, sample_registration_file: Path
    ) -> None:
        """Verify handler_map[] pattern is detected."""
        routes = discover_registered_routes(source_directory=temp_source_dir)
        assert "CheckoutIntent" in routes

    def test_discover_subscribe_pattern(
        self, temp_source_dir: Path, sample_registration_file: Path
    ) -> None:
        """Verify subscribe() pattern is detected."""
        routes = discover_registered_routes(source_directory=temp_source_dir)
        assert "OrderSummaryProjection" in routes

    def test_discover_empty_directory(self, temp_source_dir: Path) -> None:
        """Verify empty directory returns empty set."""
        routes = discover_registered_routes(source_directory=temp_source_dir)
        assert routes == set()


# =============================================================================
# Test RoutingCoverageValidator
# =============================================================================


class TestRoutingCoverageValidator:
    """Tests for RoutingCoverageValidator class."""

    def test_validate_coverage_with_unmapped_types(
        self,
        temp_source_dir: Path,
        sample_event_file: Path,
        sample_command_file: Path,
    ) -> None:
        """Verify violations are returned for unmapped types."""
        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        violations = validator.validate_coverage()

        # All message types should be reported as unmapped (no registrations)
        assert len(violations) > 0
        violation_messages = [v.message for v in violations]
        assert any("OrderCreatedEvent" in m for m in violation_messages)

    def test_validate_coverage_violation_format(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify violations have correct format."""
        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        violations = validator.validate_coverage()

        assert len(violations) > 0
        violation = violations[0]
        assert (
            violation.violation_type
            == EnumExecutionShapeViolation.UNMAPPED_MESSAGE_ROUTE
        )
        # handler_type is None because routing coverage is not handler-specific
        assert violation.handler_type is None
        assert violation.severity == "error"
        assert "not registered" in violation.message

    def test_get_unmapped_types(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify get_unmapped_types returns correct set."""
        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        unmapped = validator.get_unmapped_types()

        assert "OrderCreatedEvent" in unmapped
        assert "PaymentReceivedEvent" in unmapped
        assert "UserRegisteredEvent" in unmapped

    def test_get_coverage_report(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify coverage report contains expected fields."""
        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        report = validator.get_coverage_report()

        assert "total_types" in report
        assert "registered_types" in report
        assert "unmapped_types" in report
        assert "coverage_percent" in report
        assert report["total_types"] == 3  # 3 events in sample file

    def test_fail_fast_on_unmapped_raises(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify fail_fast_on_unmapped raises RoutingCoverageError."""
        validator = RoutingCoverageValidator(source_directory=temp_source_dir)

        with pytest.raises(RoutingCoverageError) as exc_info:
            validator.fail_fast_on_unmapped()

        assert "OrderCreatedEvent" in exc_info.value.unmapped_types

    def test_fail_fast_on_unmapped_succeeds_when_all_mapped(
        self, temp_source_dir: Path
    ) -> None:
        """Verify fail_fast_on_unmapped succeeds when all types are mapped."""
        # Create matching events and registrations
        events_file = temp_source_dir / "events.py"
        events_file.write_text("class OrderCreatedEvent: pass\n")

        reg_file = temp_source_dir / "registrations.py"
        reg_file.write_text('registry.register("OrderCreatedEvent", handler)\n')

        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        # Should not raise
        validator.fail_fast_on_unmapped()

    def test_refresh_clears_cache(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify refresh clears cached discovery results."""
        validator = RoutingCoverageValidator(source_directory=temp_source_dir)

        # First call populates cache
        unmapped1 = validator.get_unmapped_types()
        assert len(unmapped1) > 0

        # Refresh clears cache
        validator.refresh()
        assert validator._discovered_types is None
        assert validator._registered_routes is None


# =============================================================================
# Test Startup Integration
# =============================================================================


class TestValidateRoutingCoverageOnStartup:
    """Tests for validate_routing_coverage_on_startup function."""

    def test_returns_false_when_unmapped(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify returns False when types are unmapped."""
        result = validate_routing_coverage_on_startup(
            source_directory=temp_source_dir,
            fail_on_unmapped=False,
        )
        assert result is False

    def test_returns_true_when_all_mapped(self, temp_source_dir: Path) -> None:
        """Verify returns True when all types are mapped."""
        # Create matching events and registrations
        events_file = temp_source_dir / "events.py"
        events_file.write_text("class OrderCreatedEvent: pass\n")

        reg_file = temp_source_dir / "registrations.py"
        reg_file.write_text('registry.register("OrderCreatedEvent", handler)\n')

        result = validate_routing_coverage_on_startup(
            source_directory=temp_source_dir,
            fail_on_unmapped=False,
        )
        assert result is True

    def test_raises_when_fail_on_unmapped_true(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify raises RoutingCoverageError when fail_on_unmapped is True."""
        with pytest.raises(RoutingCoverageError):
            validate_routing_coverage_on_startup(
                source_directory=temp_source_dir,
                fail_on_unmapped=True,
            )


# =============================================================================
# Test CI Integration
# =============================================================================


class TestCheckRoutingCoverageCi:
    """Tests for check_routing_coverage_ci function."""

    def test_returns_false_with_violations(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify returns (False, violations) when types are unmapped."""
        passed, violations = check_routing_coverage_ci(temp_source_dir)
        assert passed is False
        assert len(violations) > 0

    def test_returns_true_with_empty_violations(self, temp_source_dir: Path) -> None:
        """Verify returns (True, []) when all types are mapped."""
        # Create matching events and registrations
        events_file = temp_source_dir / "events.py"
        events_file.write_text("class OrderCreatedEvent: pass\n")

        reg_file = temp_source_dir / "registrations.py"
        reg_file.write_text('registry.register("OrderCreatedEvent", handler)\n')

        passed, violations = check_routing_coverage_ci(temp_source_dir)
        assert passed is True
        assert violations == []

    def test_violations_have_ci_format(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify violations can be formatted for CI output."""
        _passed, violations = check_routing_coverage_ci(temp_source_dir)

        for violation in violations:
            ci_format = violation.format_for_ci()
            assert "::" in ci_format
            assert "error" in ci_format or "warning" in ci_format


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_source_directory(self, temp_source_dir: Path) -> None:
        """Verify empty directory is handled correctly."""
        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        violations = validator.validate_coverage()
        assert violations == []

    def test_non_python_files_ignored(self, temp_source_dir: Path) -> None:
        """Verify non-Python files are ignored."""
        txt_file = temp_source_dir / "readme.txt"
        txt_file.write_text("class OrderEvent: pass")

        json_file = temp_source_dir / "config.json"
        json_file.write_text('{"class": "OrderEvent"}')

        types = discover_message_types(temp_source_dir)
        assert "OrderEvent" not in types

    def test_unicode_file_handling(self, temp_source_dir: Path) -> None:
        """Verify Unicode content is handled correctly."""
        unicode_file = temp_source_dir / "unicode_events.py"
        unicode_file.write_text(
            '''
class OrderCreatedEvent:
    """Order created event."""
    pass
''',
            encoding="utf-8",
        )

        types = discover_message_types(temp_source_dir)
        assert "OrderCreatedEvent" in types

    def test_deeply_nested_files(self, temp_source_dir: Path) -> None:
        """Verify deeply nested files are discovered."""
        deep_file = temp_source_dir / "a" / "b" / "c" / "d" / "events.py"
        deep_file.parent.mkdir(parents=True, exist_ok=True)
        deep_file.write_text("class DeepEvent: pass\n")

        types = discover_message_types(temp_source_dir)
        assert "DeepEvent" in types

    def test_thread_safety(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify validator is thread-safe."""
        import threading

        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        results: list[set[str]] = []
        errors: list[Exception] = []

        def get_unmapped() -> None:
            try:
                unmapped = validator.get_unmapped_types()
                results.append(unmapped)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_unmapped) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All results should be the same
        assert all(r == results[0] for r in results)


# =============================================================================
# Test Coverage Report Accuracy
# =============================================================================


class TestCoverageReportAccuracy:
    """Test coverage report calculations."""

    def test_100_percent_coverage(self, temp_source_dir: Path) -> None:
        """Verify 100% coverage is correctly calculated."""
        events_file = temp_source_dir / "events.py"
        events_file.write_text(
            """
class OrderCreatedEvent: pass
class PaymentReceivedEvent: pass
"""
        )

        reg_file = temp_source_dir / "registrations.py"
        reg_file.write_text(
            """
registry.register("OrderCreatedEvent", handler)
registry.register("PaymentReceivedEvent", handler)
"""
        )

        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        report = validator.get_coverage_report()
        assert report["coverage_percent"] == 100.0

    def test_50_percent_coverage(self, temp_source_dir: Path) -> None:
        """Verify 50% coverage is correctly calculated."""
        events_file = temp_source_dir / "events.py"
        events_file.write_text(
            """
class OrderCreatedEvent: pass
class PaymentReceivedEvent: pass
"""
        )

        reg_file = temp_source_dir / "registrations.py"
        reg_file.write_text(
            """
registry.register("OrderCreatedEvent", handler)
"""
        )

        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        report = validator.get_coverage_report()
        assert report["coverage_percent"] == 50.0

    def test_0_percent_coverage(
        self, temp_source_dir: Path, sample_event_file: Path
    ) -> None:
        """Verify 0% coverage is correctly calculated."""
        validator = RoutingCoverageValidator(source_directory=temp_source_dir)
        report = validator.get_coverage_report()
        assert report["coverage_percent"] == 0.0
