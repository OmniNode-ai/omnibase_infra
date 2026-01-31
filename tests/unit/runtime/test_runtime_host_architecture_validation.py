# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RuntimeHostProcess architecture validation.

Tests for the architecture validation wiring in RuntimeHostProcess,
specifically the integration with HandlerArchitectureValidation
at startup (OMN-1138, refactored in OMN-1726).

Test Categories:
    - Validation skipped when no rules configured
    - ERROR severity violations block startup
    - WARNING severity violations log but don't block
    - Validation runs BEFORE other startup logic
    - Handler instantiation with configured rules (OMN-1726)

Handler Semantics - Two Distinct Concepts:
    RuntimeHostProcess has TWO INDEPENDENT handler-related checks that serve
    different purposes. Understanding this distinction is critical for reading
    these tests correctly:

    1. **Fail-fast startup check** (process._handlers - handler INSTANCES):
       - Purpose: Ensure the runtime has at least one handler instance to
         process events. A runtime without handlers is useless.
       - How tests satisfy it: Call seed_mock_handlers(process) to inject a
         mock handler instance into process._handlers.
       - When it runs: After architecture validation, during start() step 4.1.

    2. **Architecture validation** (handler registry - handler CLASSES):
       - Purpose: Validate that handler CLASSES conform to architecture rules
         (e.g., NO_HANDLER_PUBLISHING, PURE_REDUCERS).
       - How tests control it: Mock registry.list_protocols() return value.
         - Empty list [] = no classes to validate = validation passes trivially
         - Non-empty list = classes are validated against configured rules
       - When it runs: Before startup, during start() step 1.

    Key Test Scenarios:
        | Registry (Classes)     | _handlers (Instances) | Outcome                    |
        |------------------------|-----------------------|----------------------------|
        | Empty []               | seed_mock_handlers()  | Startup OK (no validation) |
        | [MockClass] + passing  | seed_mock_handlers()  | Startup OK (rules pass)    |
        | [MockClass] + failing  | Not reached           | ArchitectureViolationError |

    Why "No Handlers to Validate" Tests Still Seed Handlers:
        When we test "empty registry skips validation", we still call
        seed_mock_handlers() because:
        1. Architecture validation (empty registry) passes trivially
        2. Startup continues to the SEPARATE fail-fast check
        3. Fail-fast check requires at least one handler INSTANCE
        4. Without seeding, the test would fail for the wrong reason

        This is intentional - we're testing architecture validation behavior,
        not the fail-fast startup check.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.errors import ArchitectureViolationError
from omnibase_infra.nodes.architecture_validator import (
    EnumValidationSeverity,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
    ModelRuleCheckResult,
)

# Import RuntimeHostProcess (should always be available)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from tests.conftest import make_runtime_config, seed_mock_handlers

# =============================================================================
# Quick Reference: Handler Semantics (see module docstring for full details)
# =============================================================================
# Two independent checks - don't confuse them:
#
# | Check                    | What It Checks          | How Tests Control It         |
# |--------------------------|-------------------------|------------------------------|
# | Architecture validation  | Handler CLASSES         | mock_registry.list_protocols |
# | Fail-fast startup        | Handler INSTANCES       | seed_mock_handlers()         |
#
# "No handlers to validate" = empty registry (classes), NOT empty _handlers (instances)
# =============================================================================


# =============================================================================
# Mock Rule Implementation
# =============================================================================


class MockArchitectureRule:
    """Mock architecture rule for testing.

    Can be configured to pass or fail with specific severity.
    Uses valid rule IDs from SUPPORTED_RULE_IDS to pass validation.
    """

    def __init__(
        self,
        rule_id: str = "NO_HANDLER_PUBLISHING",  # Use valid rule from SUPPORTED_RULE_IDS
        name: str = "Mock Rule",
        description: str = "A mock rule for testing",
        severity: EnumValidationSeverity = EnumValidationSeverity.ERROR,
        should_pass: bool = True,
    ) -> None:
        """Initialize mock rule.

        Args:
            rule_id: Unique identifier for this rule.
            name: Human-readable name.
            description: Rule description.
            severity: Severity level for violations.
            should_pass: Whether check() should return passed=True.
        """
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.severity = severity
        self._should_pass = should_pass

    def check(self, target: object) -> ModelRuleCheckResult:
        """Check the target against this rule.

        Args:
            target: Node or handler to validate.

        Returns:
            ModelRuleCheckResult indicating pass/fail.
        """
        if self._should_pass:
            return ModelRuleCheckResult(passed=True, rule_id=self.rule_id)
        return ModelRuleCheckResult(
            passed=False,
            rule_id=self.rule_id,
            message=f"Mock violation for {target}",
        )


# =============================================================================
# Test: No Rules Configured
# =============================================================================


class TestNoRulesConfigured:
    """Tests when no architecture rules are provided."""

    @pytest.mark.asyncio
    async def test_validation_skipped_when_no_rules(self) -> None:
        """Validation is skipped when no rules are configured."""
        process = RuntimeHostProcess(config=make_runtime_config())

        # Mock the rest of start() to prevent actual startup
        with (
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.service_runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
        ):
            # Seed handler instances to satisfy fail-fast startup check
            seed_mock_handlers(process)
            # Should not raise - no architecture rules configured
            await process.start()
            assert process.is_running

    @pytest.mark.asyncio
    async def test_validation_skipped_with_empty_rules_tuple(self) -> None:
        """Validation is skipped when empty rules tuple is provided."""
        process = RuntimeHostProcess(
            architecture_rules=(), config=make_runtime_config()
        )

        with (
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.service_runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
        ):
            # Seed handler instances to satisfy fail-fast startup check
            seed_mock_handlers(process)
            await process.start()
            assert process.is_running


# =============================================================================
# Test: ERROR Severity Blocks Startup
# =============================================================================


class TestErrorSeverityBlocksStartup:
    """Tests that ERROR severity violations prevent startup."""

    @pytest.mark.asyncio
    async def test_empty_registry_skips_architecture_validation(self) -> None:
        """Empty handler registry means no classes to validate against rules.

        This test verifies that when the handler REGISTRY contains no handler
        classes (list_protocols() returns []), architecture validation passes
        because there's nothing to check rules against.

        Note: We still call seed_mock_handlers() to populate process._handlers
        with a mock instance. This satisfies the SEPARATE fail-fast startup check
        that requires at least one handler instance for event processing.

        The two checks are independent:
        - Fail-fast: process._handlers must not be empty (handler INSTANCES)
        - Architecture validation: registry.list_protocols() classes validated
        """
        # Create a rule that always fails with ERROR severity
        # Use valid rule ID from SUPPORTED_RULE_IDS
        failing_rule = MockArchitectureRule(
            rule_id="NO_HANDLER_PUBLISHING",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )

        mock_container = MagicMock()

        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(failing_rule,),
            config=make_runtime_config(),
        )

        # Mock the handler registry to return NO handler classes for validation.
        # The failing_rule would fail if it had any handlers to check, but
        # with list_protocols() returning [], there are no classes to validate.
        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.service_runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
        ):
            mock_registry = MagicMock()
            # Empty registry = no handler CLASSES for architecture validation
            mock_registry.list_protocols.return_value = []
            mock_get_registry.return_value = mock_registry

            # Seed handler INSTANCES in _handlers to satisfy fail-fast startup
            # (see seed_mock_handlers docstring for detailed explanation)
            seed_mock_handlers(process)

            # Should NOT raise - empty registry means no classes to validate
            await process.start()
            assert process.is_running

    @pytest.mark.asyncio
    async def test_error_violation_contains_all_violations(self) -> None:
        """ArchitectureViolationError contains all blocking violations."""

        class MockHandlerClass:
            """Mock handler class for testing."""

        # Create multiple failing rules
        # Use valid rule IDs from SUPPORTED_RULE_IDS
        rule1 = MockArchitectureRule(
            rule_id="NO_HANDLER_PUBLISHING",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )
        rule2 = MockArchitectureRule(
            rule_id="PURE_REDUCERS",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(rule1, rule2),
            config=make_runtime_config(),
        )

        # Mock handler registry to return one handler class
        with patch.object(process, "_get_handler_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = ["mock"]
            mock_registry.get.return_value = MockHandlerClass
            mock_get_registry.return_value = mock_registry

            with pytest.raises(ArchitectureViolationError) as exc_info:
                await process.start()

            # Should have 2 violations (one from each rule)
            assert len(exc_info.value.violations) == 2
            violation_rule_ids = {v.rule_id for v in exc_info.value.violations}
            assert "NO_HANDLER_PUBLISHING" in violation_rule_ids
            assert "PURE_REDUCERS" in violation_rule_ids

    @pytest.mark.asyncio
    async def test_event_bus_not_started_on_validation_failure(self) -> None:
        """Event bus is NOT started when validation fails."""

        class MockHandlerClass:
            pass

        # Use valid rule ID from SUPPORTED_RULE_IDS
        failing_rule = MockArchitectureRule(
            rule_id="NO_FSM_IN_ORCHESTRATORS",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )

        process = RuntimeHostProcess(
            architecture_rules=(failing_rule,),
            config=make_runtime_config(),
        )

        event_bus_start = AsyncMock()
        process._event_bus.start = event_bus_start

        with patch.object(process, "_get_handler_registry") as mock_get_registry:
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = ["mock"]
            mock_registry.get.return_value = MockHandlerClass
            mock_get_registry.return_value = mock_registry

            with pytest.raises(ArchitectureViolationError):
                await process.start()

            # Event bus should NOT have been started
            event_bus_start.assert_not_called()


# =============================================================================
# Test: WARNING Severity Logs but Doesn't Block
# =============================================================================


class TestWarningSeverityDoesntBlock:
    """Tests that WARNING severity violations don't block startup."""

    @pytest.mark.asyncio
    async def test_warning_violations_dont_block(self) -> None:
        """WARNING severity violations log but allow startup."""

        class MockHandlerClass:
            pass

        warning_rule = MockArchitectureRule(
            rule_id="NO_DIRECT_HANDLER_DISPATCH",  # Use valid WARNING-severity rule
            severity=EnumValidationSeverity.WARNING,
            should_pass=False,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(warning_rule,),
            config=make_runtime_config(),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.service_runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
        ):
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = ["mock"]
            mock_registry.get.return_value = MockHandlerClass
            mock_get_registry.return_value = mock_registry

            # Seed handler instances to satisfy fail-fast startup check
            seed_mock_handlers(process)
            # Should NOT raise - warnings don't block
            await process.start()
            assert process.is_running

    @pytest.mark.asyncio
    async def test_warning_violations_are_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WARNING severity violations are logged."""

        class MockHandlerClass:
            pass

        warning_rule = MockArchitectureRule(
            rule_id="NO_DIRECT_HANDLER_DISPATCH",  # Use valid WARNING-severity rule
            name="Warning Rule",
            severity=EnumValidationSeverity.WARNING,
            should_pass=False,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(warning_rule,),
            config=make_runtime_config(),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.service_runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
            caplog.at_level(logging.WARNING),
        ):
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = ["mock"]
            mock_registry.get.return_value = MockHandlerClass
            mock_get_registry.return_value = mock_registry

            # Seed handler instances to satisfy fail-fast startup check
            seed_mock_handlers(process)
            await process.start()

            # Check that warning was logged
            warning_logs = [r for r in caplog.records if r.levelname == "WARNING"]
            assert len(warning_logs) > 0
            assert any("Architecture warning" in r.message for r in warning_logs)


# =============================================================================
# Test: Validation Runs Before Other Startup Logic
# =============================================================================


class TestValidationOrder:
    """Tests that validation runs before other startup steps."""

    @pytest.mark.asyncio
    async def test_validation_runs_first(self) -> None:
        """Architecture validation runs before event bus starts."""

        class MockHandlerClass:
            pass

        failing_rule = MockArchitectureRule(
            rule_id="NO_WORKFLOW_IN_REDUCERS",  # Use valid ERROR-severity rule
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )

        process = RuntimeHostProcess(
            architecture_rules=(failing_rule,),
            config=make_runtime_config(),
        )

        # Track call order
        call_order: list[str] = []

        async def track_event_bus_start() -> None:
            call_order.append("event_bus_start")

        def track_wire_handlers() -> None:
            call_order.append("wire_handlers")

        async def track_populate_handlers() -> None:
            call_order.append("populate_handlers")

        process._event_bus.start = track_event_bus_start

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch(
                "omnibase_infra.runtime.service_runtime_host_process.wire_handlers",
                side_effect=track_wire_handlers,
            ),
            patch.object(
                process,
                "_populate_handlers_from_registry",
                side_effect=track_populate_handlers,
            ),
        ):
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = ["mock"]
            mock_registry.get.return_value = MockHandlerClass
            mock_get_registry.return_value = mock_registry

            with pytest.raises(ArchitectureViolationError):
                await process.start()

            # None of the other startup steps should have been called
            assert "event_bus_start" not in call_order
            assert "wire_handlers" not in call_order
            assert "populate_handlers" not in call_order


# =============================================================================
# Test: Handler Instantiation (OMN-1726 refactoring)
# =============================================================================


class TestContainerHandling:
    """Tests for handler instantiation with architecture rules.

    Note: After OMN-1726 refactoring, architecture validation uses
    HandlerArchitectureValidation instead of NodeArchitectureValidatorCompute.
    The handler takes only `rules` parameter (no container injection).

    These tests verify that the handler is correctly instantiated with
    the configured architecture rules.
    """

    @pytest.mark.asyncio
    async def test_handler_instantiated_with_configured_rules(self) -> None:
        """Handler is instantiated with the configured architecture rules."""
        mock_container = MagicMock()

        # Rule that passes
        passing_rule = MockArchitectureRule(
            rule_id="NO_LOCAL_ONLY_PATHS",  # Use valid rule from SUPPORTED_RULE_IDS
            should_pass=True,
        )

        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(passing_rule,),
            config=make_runtime_config(),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.service_runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
            patch(
                "omnibase_infra.nodes.architecture_validator.HandlerArchitectureValidation"
            ) as mock_handler_cls,
        ):
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = []
            mock_get_registry.return_value = mock_registry

            mock_handler = MagicMock()
            mock_handler.validate_architecture.return_value = (
                ModelArchitectureValidationResult(
                    violations=(),
                    rules_checked=("NO_LOCAL_ONLY_PATHS",),
                    nodes_checked=0,
                    handlers_checked=0,
                )
            )
            mock_handler_cls.return_value = mock_handler

            # Seed handler instances to satisfy fail-fast startup check
            seed_mock_handlers(process)
            await process.start()

            # Verify handler was instantiated with the configured rules
            mock_handler_cls.assert_called_once()
            call_kwargs = mock_handler_cls.call_args[1]
            assert call_kwargs["rules"] == (passing_rule,)

            # Verify validate_architecture was called
            mock_handler.validate_architecture.assert_called_once()

    @pytest.mark.asyncio
    async def test_validation_works_without_container(self) -> None:
        """Validation works when no container is provided to RuntimeHostProcess.

        After OMN-1726, architecture validation uses HandlerArchitectureValidation
        which does not require a container. This test verifies that validation
        works correctly regardless of container state.
        """
        passing_rule = MockArchitectureRule(
            rule_id="NO_LOCAL_ONLY_PATHS",  # Use valid rule from SUPPORTED_RULE_IDS
            should_pass=True,
        )

        # No container provided
        process = RuntimeHostProcess(
            architecture_rules=(passing_rule,),
            config=make_runtime_config(),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.service_runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
            patch(
                "omnibase_infra.nodes.architecture_validator.HandlerArchitectureValidation"
            ) as mock_handler_cls,
        ):
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = []
            mock_get_registry.return_value = mock_registry

            mock_handler = MagicMock()
            mock_handler.validate_architecture.return_value = (
                ModelArchitectureValidationResult(
                    violations=(),
                    rules_checked=("NO_LOCAL_ONLY_PATHS",),
                    nodes_checked=0,
                    handlers_checked=0,
                )
            )
            mock_handler_cls.return_value = mock_handler

            # Seed handler instances to satisfy fail-fast startup check
            seed_mock_handlers(process)
            await process.start()

            # Verify handler was instantiated with rules (no container dependency)
            mock_handler_cls.assert_called_once()
            call_kwargs = mock_handler_cls.call_args[1]
            assert call_kwargs["rules"] == (passing_rule,)
            # No container in call_kwargs - handler doesn't use containers
            assert "container" not in call_kwargs

            # Verify startup completed successfully
            assert process.is_running


# =============================================================================
# Test: Passing Validation
# =============================================================================


class TestPassingValidation:
    """Tests for successful validation scenarios."""

    @pytest.mark.asyncio
    async def test_passing_rules_allow_startup(self) -> None:
        """Passing rules allow normal startup."""
        passing_rule = MockArchitectureRule(
            rule_id="NO_LOCAL_ONLY_PATHS",  # Use valid rule from SUPPORTED_RULE_IDS
            should_pass=True,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(passing_rule,),
            config=make_runtime_config(),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.service_runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
        ):
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = []
            mock_get_registry.return_value = mock_registry

            # Seed handler instances to satisfy fail-fast startup check
            seed_mock_handlers(process)
            await process.start()
            assert process.is_running

    @pytest.mark.asyncio
    async def test_success_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Successful validation is logged."""
        passing_rule = MockArchitectureRule(
            rule_id="NO_LOCAL_ONLY_PATHS",  # Use valid rule from SUPPORTED_RULE_IDS
            should_pass=True,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(passing_rule,),
            config=make_runtime_config(),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.service_runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
            caplog.at_level(logging.INFO),
        ):
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = []
            mock_get_registry.return_value = mock_registry

            # Seed handler instances to satisfy fail-fast startup check
            seed_mock_handlers(process)
            await process.start()

            info_logs = [r for r in caplog.records if r.levelname == "INFO"]
            assert any("Architecture validation passed" in r.message for r in info_logs)


# =============================================================================
# Test: ArchitectureViolationError
# =============================================================================


class TestArchitectureViolationError:
    """Tests for ArchitectureViolationError class."""

    def test_error_contains_violations(self) -> None:
        """Error stores violations for inspection."""
        violations = (
            ModelArchitectureViolation(
                rule_id="RULE_1",
                rule_name="Rule 1",
                severity=EnumValidationSeverity.ERROR,
                target_type="handler",
                target_name="MyHandler",
                message="Violation message",
            ),
        )

        error = ArchitectureViolationError(
            message="Test error",
            violations=violations,
        )

        assert error.violations == violations
        assert len(error.violations) == 1

    def test_format_violations(self) -> None:
        """format_violations() returns formatted string."""
        violations = (
            ModelArchitectureViolation(
                rule_id="RULE_1",
                rule_name="Rule 1",
                severity=EnumValidationSeverity.ERROR,
                target_type="handler",
                target_name="MyHandler",
                message="Test violation",
            ),
            ModelArchitectureViolation(
                rule_id="RULE_2",
                rule_name="Rule 2",
                severity=EnumValidationSeverity.ERROR,
                target_type="node",
                target_name="MyNode",
                message="Another violation",
            ),
        )

        error = ArchitectureViolationError(
            message="Test error",
            violations=violations,
        )

        formatted = error.format_violations()
        assert "RULE_1" in formatted
        assert "RULE_2" in formatted
        assert "MyHandler" in formatted
        assert "MyNode" in formatted

    def test_error_context_includes_violation_info(self) -> None:
        """Error context includes violation count and rule IDs."""
        violations = (
            ModelArchitectureViolation(
                rule_id="RULE_1",
                rule_name="Rule 1",
                severity=EnumValidationSeverity.ERROR,
                target_type="handler",
                target_name="MyHandler",
                message="Violation",
            ),
        )

        error = ArchitectureViolationError(
            message="Test error",
            violations=violations,
        )

        # Check that the error model has the context
        assert error.model.context.get("violation_count") == 1
        assert "RULE_1" in error.model.context.get("violation_rule_ids", ())
