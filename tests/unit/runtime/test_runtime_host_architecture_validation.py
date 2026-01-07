# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RuntimeHostProcess architecture validation.

Tests for the architecture validation wiring in RuntimeHostProcess,
specifically the integration with NodeArchitectureValidatorCompute
at startup (OMN-1138).

Test Categories:
    - Validation skipped when no rules configured
    - ERROR severity violations block startup
    - WARNING severity violations log but don't block
    - Validation runs BEFORE other startup logic
    - Container injection vs minimal container creation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.errors import ArchitectureViolationError
from omnibase_infra.nodes.architecture_validator import (
    EnumValidationSeverity,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
    ModelRuleCheckResult,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.architecture_validator import ProtocolArchitectureRule

# Import RuntimeHostProcess (should always be available)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

# =============================================================================
# Mock Rule Implementation
# =============================================================================


class MockArchitectureRule:
    """Mock architecture rule for testing.

    Can be configured to pass or fail with specific severity.
    """

    def __init__(
        self,
        rule_id: str = "MOCK_RULE",
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
        process = RuntimeHostProcess()

        # Mock the rest of start() to prevent actual startup
        with (
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
        ):
            # Should not raise - no validation occurs
            await process.start()
            assert process.is_running

    @pytest.mark.asyncio
    async def test_validation_skipped_with_empty_rules_tuple(self) -> None:
        """Validation is skipped when empty rules tuple is provided."""
        process = RuntimeHostProcess(architecture_rules=())

        with (
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
        ):
            await process.start()
            assert process.is_running


# =============================================================================
# Test: ERROR Severity Blocks Startup
# =============================================================================


class TestErrorSeverityBlocksStartup:
    """Tests that ERROR severity violations prevent startup."""

    @pytest.mark.asyncio
    async def test_error_violation_blocks_startup_no_handlers(self) -> None:
        """With no handlers, validation passes (nothing to validate)."""
        # Create a rule that always fails with ERROR severity
        failing_rule = MockArchitectureRule(
            rule_id="ALWAYS_FAILS",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )

        mock_container = MagicMock()

        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(failing_rule,),
        )

        # When no handlers are registered, validation passes because
        # there's nothing to validate
        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.runtime_host_process.wire_handlers"),
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

            # Should NOT raise - no handlers to validate means nothing fails
            await process.start()
            assert process.is_running

    @pytest.mark.asyncio
    async def test_error_violation_contains_all_violations(self) -> None:
        """ArchitectureViolationError contains all blocking violations."""

        class MockHandlerClass:
            """Mock handler class for testing."""

        # Create multiple failing rules
        rule1 = MockArchitectureRule(
            rule_id="RULE_1",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )
        rule2 = MockArchitectureRule(
            rule_id="RULE_2",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(rule1, rule2),
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
            assert "RULE_1" in violation_rule_ids
            assert "RULE_2" in violation_rule_ids

    @pytest.mark.asyncio
    async def test_event_bus_not_started_on_validation_failure(self) -> None:
        """Event bus is NOT started when validation fails."""

        class MockHandlerClass:
            pass

        failing_rule = MockArchitectureRule(
            rule_id="BLOCKER",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )

        process = RuntimeHostProcess(
            architecture_rules=(failing_rule,),
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
            rule_id="WARNING_RULE",
            severity=EnumValidationSeverity.WARNING,
            should_pass=False,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(warning_rule,),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.runtime_host_process.wire_handlers"),
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
            rule_id="WARNING_RULE",
            name="Warning Rule",
            severity=EnumValidationSeverity.WARNING,
            should_pass=False,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(warning_rule,),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.runtime_host_process.wire_handlers"),
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
            rule_id="BLOCKER",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )

        process = RuntimeHostProcess(
            architecture_rules=(failing_rule,),
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
                "omnibase_infra.runtime.runtime_host_process.wire_handlers",
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
# Test: Container Handling
# =============================================================================


class TestContainerHandling:
    """Tests for container injection and minimal container creation."""

    @pytest.mark.asyncio
    async def test_uses_injected_container(self) -> None:
        """Uses injected container for validation."""
        mock_container = MagicMock()

        # Rule that passes
        passing_rule = MockArchitectureRule(
            rule_id="PASS",
            should_pass=True,
        )

        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(passing_rule,),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
            patch(
                "omnibase_infra.nodes.architecture_validator.NodeArchitectureValidatorCompute"
            ) as mock_validator_cls,
        ):
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = []
            mock_get_registry.return_value = mock_registry

            mock_validator = MagicMock()
            mock_validator.compute.return_value = ModelArchitectureValidationResult(
                violations=(),
                rules_checked=("PASS",),
                nodes_checked=0,
                handlers_checked=0,
            )
            mock_validator_cls.return_value = mock_validator

            await process.start()

            # Verify injected container was used
            mock_validator_cls.assert_called_once()
            call_kwargs = mock_validator_cls.call_args[1]
            assert call_kwargs["container"] is mock_container

    @pytest.mark.asyncio
    async def test_creates_container_if_none_provided(self) -> None:
        """Creates container if none was injected."""
        passing_rule = MockArchitectureRule(
            rule_id="PASS",
            should_pass=True,
        )

        # No container provided
        process = RuntimeHostProcess(
            architecture_rules=(passing_rule,),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.runtime_host_process.wire_handlers"),
            patch.object(
                process, "_populate_handlers_from_registry", new_callable=AsyncMock
            ),
            patch.object(
                process, "_initialize_idempotency_store", new_callable=AsyncMock
            ),
            patch.object(process._event_bus, "subscribe", new_callable=AsyncMock),
            # Patch where it's imported inside _get_or_create_container
            patch(
                "omnibase_core.models.container.model_onex_container.ModelONEXContainer"
            ) as mock_container_cls,
            patch(
                "omnibase_infra.nodes.architecture_validator.NodeArchitectureValidatorCompute"
            ) as mock_validator_cls,
        ):
            mock_registry = MagicMock()
            mock_registry.list_protocols.return_value = []
            mock_get_registry.return_value = mock_registry

            mock_container = MagicMock()
            mock_container_cls.return_value = mock_container

            mock_validator = MagicMock()
            mock_validator.compute.return_value = ModelArchitectureValidationResult(
                violations=(),
                rules_checked=("PASS",),
                nodes_checked=0,
                handlers_checked=0,
            )
            mock_validator_cls.return_value = mock_validator

            await process.start()

            # Verify container was created
            mock_container_cls.assert_called_once()


# =============================================================================
# Test: Passing Validation
# =============================================================================


class TestPassingValidation:
    """Tests for successful validation scenarios."""

    @pytest.mark.asyncio
    async def test_passing_rules_allow_startup(self) -> None:
        """Passing rules allow normal startup."""
        passing_rule = MockArchitectureRule(
            rule_id="PASS",
            should_pass=True,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(passing_rule,),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.runtime_host_process.wire_handlers"),
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

            await process.start()
            assert process.is_running

    @pytest.mark.asyncio
    async def test_success_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Successful validation is logged."""
        passing_rule = MockArchitectureRule(
            rule_id="PASS",
            should_pass=True,
        )

        mock_container = MagicMock()
        process = RuntimeHostProcess(
            container=mock_container,
            architecture_rules=(passing_rule,),
        )

        with (
            patch.object(process, "_get_handler_registry") as mock_get_registry,
            patch.object(process._event_bus, "start", new_callable=AsyncMock),
            patch("omnibase_infra.runtime.runtime_host_process.wire_handlers"),
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
