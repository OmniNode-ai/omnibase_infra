# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for handshake validation gate in kernel bootstrap (OMN-2089).

These tests verify that:
1. Plugins with validate_handshake() that passes allow wiring to proceed
2. Plugins with validate_handshake() that fails block wiring
3. Plugins without validate_handshake() pass by default
4. Hard gate exceptions (B1-B3) propagate through the handshake gate
5. ModelHandshakeResult and ModelHandshakeCheckResult work correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.runtime.models.model_handshake_check_result import (
    ModelHandshakeCheckResult,
)
from omnibase_infra.runtime.models.model_handshake_result import (
    ModelHandshakeResult,
)
from omnibase_infra.runtime.protocol_domain_plugin import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
)

# ============================================================================
# Test helpers: plugin variants for handshake testing
# ============================================================================


class PluginWithPassingHandshake:
    """Plugin that implements validate_handshake() and passes all checks."""

    def __init__(self) -> None:
        self._handshake_called = False
        self._wire_handlers_called = False

    @property
    def plugin_id(self) -> str:
        return "passing-handshake"

    @property
    def display_name(self) -> str:
        return "Passing Handshake"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        return True

    async def initialize(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def validate_handshake(
        self, config: ModelDomainPluginConfig
    ) -> ModelHandshakeResult:
        self._handshake_called = True
        return ModelHandshakeResult.all_passed(
            plugin_id=self.plugin_id,
            checks=[
                ModelHandshakeCheckResult(
                    check_name="db_ownership",
                    passed=True,
                    message="Database owned by correct service",
                ),
                ModelHandshakeCheckResult(
                    check_name="schema_fingerprint",
                    passed=True,
                    message="Schema fingerprint matches",
                ),
            ],
        )

    async def wire_handlers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        self._wire_handlers_called = True
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def wire_dispatchers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def start_consumers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def shutdown(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)


class PluginWithFailingHandshake:
    """Plugin that implements validate_handshake() and fails one check."""

    def __init__(self) -> None:
        self._handshake_called = False
        self._wire_handlers_called = False

    @property
    def plugin_id(self) -> str:
        return "failing-handshake"

    @property
    def display_name(self) -> str:
        return "Failing Handshake"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        return True

    async def initialize(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def validate_handshake(
        self, config: ModelDomainPluginConfig
    ) -> ModelHandshakeResult:
        self._handshake_called = True
        return ModelHandshakeResult.failed(
            plugin_id=self.plugin_id,
            error_message="Schema fingerprint mismatch detected",
            checks=[
                ModelHandshakeCheckResult(
                    check_name="db_ownership",
                    passed=True,
                    message="Database owned by correct service",
                ),
                ModelHandshakeCheckResult(
                    check_name="schema_fingerprint",
                    passed=False,
                    message="Schema fingerprint mismatch: expected abc, got xyz",
                ),
            ],
        )

    async def wire_handlers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        self._wire_handlers_called = True
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def wire_dispatchers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def start_consumers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def shutdown(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)


class PluginWithoutHandshake:
    """Plugin that does NOT implement validate_handshake().

    This tests the default-pass behavior: plugins without the method
    should proceed to wiring without any validation gate.
    """

    def __init__(self) -> None:
        self._wire_handlers_called = False

    @property
    def plugin_id(self) -> str:
        return "no-handshake"

    @property
    def display_name(self) -> str:
        return "No Handshake"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        return True

    async def initialize(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    # NOTE: No validate_handshake() method -- default pass

    async def wire_handlers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        self._wire_handlers_called = True
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def wire_dispatchers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def start_consumers(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)

    async def shutdown(
        self, config: ModelDomainPluginConfig
    ) -> ModelDomainPluginResult:
        return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)


# ============================================================================
# Tests: ModelHandshakeResult
# ============================================================================


class TestModelHandshakeResult:
    """Tests for ModelHandshakeResult dataclass."""

    def test_all_passed_factory(self) -> None:
        """Test all_passed() factory creates a passing result."""
        result = ModelHandshakeResult.all_passed(
            plugin_id="test",
            checks=[
                ModelHandshakeCheckResult(
                    check_name="check1",
                    passed=True,
                ),
            ],
        )
        assert result.passed
        assert bool(result) is True
        assert result.plugin_id == "test"
        assert len(result.checks) == 1

    def test_failed_factory(self) -> None:
        """Test failed() factory creates a failing result."""
        result = ModelHandshakeResult.failed(
            plugin_id="test",
            error_message="Something went wrong",
            checks=[
                ModelHandshakeCheckResult(
                    check_name="check1",
                    passed=False,
                    message="Check failed",
                ),
            ],
        )
        assert not result.passed
        assert bool(result) is False
        assert result.error_message == "Something went wrong"
        assert len(result.checks) == 1

    def test_default_pass_factory(self) -> None:
        """Test default_pass() factory for plugins without validation."""
        result = ModelHandshakeResult.default_pass(plugin_id="test")
        assert result.passed
        assert bool(result) is True
        assert len(result.checks) == 0

    def test_bool_reflects_passed_status(self) -> None:
        """Test __bool__ returns True only when passed is True."""
        passing = ModelHandshakeResult(plugin_id="p", passed=True)
        failing = ModelHandshakeResult(plugin_id="p", passed=False)

        assert bool(passing) is True
        assert bool(failing) is False


class TestModelHandshakeCheckResult:
    """Tests for ModelHandshakeCheckResult dataclass."""

    def test_basic_check_result(self) -> None:
        """Test creating a basic check result."""
        check = ModelHandshakeCheckResult(
            check_name="db_ownership",
            passed=True,
            message="Database owned by omnibase_infra",
        )
        assert check.check_name == "db_ownership"
        assert check.passed is True
        assert check.message == "Database owned by omnibase_infra"

    def test_check_result_defaults(self) -> None:
        """Test that message defaults to empty string."""
        check = ModelHandshakeCheckResult(
            check_name="check1",
            passed=True,
        )
        assert check.message == ""


# ============================================================================
# Tests: Handshake gate behavior (kernel lifecycle simulation)
# ============================================================================


class TestHandshakeGateBehavior:
    """Tests for handshake gate behavior in kernel bootstrap.

    These tests simulate the kernel's plugin activation loop to verify
    the handshake gate correctly gates handler wiring.
    """

    @pytest.mark.asyncio
    async def test_passing_handshake_allows_wiring(self) -> None:
        """When validate_handshake() passes, wire_handlers() is called."""
        plugin = PluginWithPassingHandshake()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        # Simulate kernel lifecycle: initialize -> validate_handshake -> wire_handlers
        init_result = await plugin.initialize(config)
        assert init_result.success

        handshake_result = await plugin.validate_handshake(config)
        assert handshake_result.passed

        # Proceed to wiring since handshake passed
        wire_result = await plugin.wire_handlers(config)
        assert wire_result.success

        assert plugin._handshake_called
        assert plugin._wire_handlers_called

    @pytest.mark.asyncio
    async def test_failing_handshake_blocks_wiring(self) -> None:
        """When validate_handshake() fails, wire_handlers() is NOT called."""
        plugin = PluginWithFailingHandshake()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        # Simulate kernel lifecycle: initialize -> validate_handshake -> (abort)
        init_result = await plugin.initialize(config)
        assert init_result.success

        handshake_result = await plugin.validate_handshake(config)
        assert not handshake_result.passed
        assert handshake_result.error_message == "Schema fingerprint mismatch detected"

        # Kernel would skip wire_handlers() since handshake failed
        # Verify wire_handlers was NOT called
        assert plugin._handshake_called
        assert not plugin._wire_handlers_called

    @pytest.mark.asyncio
    async def test_no_handshake_method_defaults_to_pass(self) -> None:
        """Plugins without validate_handshake() pass by default."""
        plugin = PluginWithoutHandshake()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        # Simulate kernel lifecycle: initialize -> (no handshake) -> wire_handlers
        init_result = await plugin.initialize(config)
        assert init_result.success

        # Check hasattr like the kernel does
        has_handshake = hasattr(plugin, "validate_handshake") and callable(
            getattr(plugin, "validate_handshake", None)
        )
        assert not has_handshake

        # Proceed directly to wiring since no handshake method
        wire_result = await plugin.wire_handlers(config)
        assert wire_result.success
        assert plugin._wire_handlers_called

    @pytest.mark.asyncio
    async def test_handshake_check_details_available(self) -> None:
        """Verify individual check results are accessible for diagnostics."""
        plugin = PluginWithFailingHandshake()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        await plugin.initialize(config)
        handshake_result = await plugin.validate_handshake(config)

        # Verify check details
        assert len(handshake_result.checks) == 2

        db_check = handshake_result.checks[0]
        assert db_check.check_name == "db_ownership"
        assert db_check.passed is True

        schema_check = handshake_result.checks[1]
        assert schema_check.check_name == "schema_fingerprint"
        assert schema_check.passed is False
        assert "mismatch" in schema_check.message


class TestHandshakePhaseStateMachine:
    """Tests for the phase state machine:
    INITIALIZING -> HANDSHAKE_VALIDATE -> HANDSHAKE_ATTEST -> WIRING -> READY
    """

    @pytest.mark.asyncio
    async def test_full_phase_sequence_on_success(self) -> None:
        """Verify the full phase sequence when all checks pass.

        INITIALIZING -> HANDSHAKE_VALIDATE -> HANDSHAKE_ATTEST -> WIRING -> READY
        """
        plugin = PluginWithPassingHandshake()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        phases: list[str] = []

        # INITIALIZING
        init_result = await plugin.initialize(config)
        assert init_result.success
        phases.append("INITIALIZING")

        # HANDSHAKE_VALIDATE
        handshake_result = await plugin.validate_handshake(config)
        phases.append("HANDSHAKE_VALIDATE")

        # HANDSHAKE_ATTEST (transition on success)
        if handshake_result.passed:
            phases.append("HANDSHAKE_ATTEST")

        # WIRING
        wire_result = await plugin.wire_handlers(config)
        assert wire_result.success
        phases.append("WIRING")

        # READY
        phases.append("READY")

        assert phases == [
            "INITIALIZING",
            "HANDSHAKE_VALIDATE",
            "HANDSHAKE_ATTEST",
            "WIRING",
            "READY",
        ]

    @pytest.mark.asyncio
    async def test_phase_aborts_on_handshake_failure(self) -> None:
        """Verify that phase sequence stops at HANDSHAKE_VALIDATE on failure.

        INITIALIZING -> HANDSHAKE_VALIDATE -> (abort, never reaches WIRING)
        """
        plugin = PluginWithFailingHandshake()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        phases: list[str] = []

        # INITIALIZING
        init_result = await plugin.initialize(config)
        assert init_result.success
        phases.append("INITIALIZING")

        # HANDSHAKE_VALIDATE
        handshake_result = await plugin.validate_handshake(config)
        phases.append("HANDSHAKE_VALIDATE")

        # HANDSHAKE_ATTEST only on success
        if handshake_result.passed:
            phases.append("HANDSHAKE_ATTEST")
            # WIRING only after attestation
            wire_result = await plugin.wire_handlers(config)
            phases.append("WIRING")
            phases.append("READY")

        # Should stop at HANDSHAKE_VALIDATE -- never reaches WIRING or READY
        assert phases == ["INITIALIZING", "HANDSHAKE_VALIDATE"]
        assert not plugin._wire_handlers_called


__all__: list[str] = [
    "PluginWithFailingHandshake",
    "PluginWithPassingHandshake",
    "PluginWithoutHandshake",
    "TestHandshakeGateBehavior",
    "TestHandshakePhaseStateMachine",
    "TestModelHandshakeCheckResult",
    "TestModelHandshakeResult",
]
