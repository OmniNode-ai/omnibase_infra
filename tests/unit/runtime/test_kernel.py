# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the ONEX runtime kernel.

Tests the contract-driven bootstrap entrypoint including:
- Configuration loading from contracts
- Environment variable overrides
- RuntimeHostProcess integration
- Signal handler setup
- Graceful shutdown
"""

from __future__ import annotations

from collections.abc import Coroutine
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from omnibase_core.container import ModelONEXContainer

from omnibase_infra.errors import ProtocolConfigurationError

# Import shared service registry availability check
from tests.conftest import check_service_registry_available

# Check if service_registry is available (circular import bug in omnibase_core 0.6.2)
_SERVICE_REGISTRY_AVAILABLE = check_service_registry_available()
_SKIP_REASON = (
    "service_registry is None due to circular import bug in omnibase_core 0.6.2. "
    "Upgrade to omnibase_core >= 0.6.3 to run these tests."
)
from omnibase_infra.runtime.kernel import (
    DEFAULT_GROUP_ID,
    DEFAULT_INPUT_TOPIC,
    DEFAULT_OUTPUT_TOPIC,
    MAX_PORT,
    MIN_PORT,
    bootstrap,
    configure_logging,
    load_runtime_config,
    main,
)
from omnibase_infra.runtime.models import ModelRuntimeConfig

if TYPE_CHECKING:
    from collections.abc import Generator


class TestLoadRuntimeConfig:
    """Tests for load_runtime_config function."""

    def test_load_config_from_file(self, tmp_path: Path) -> None:
        """Test loading config from a valid YAML file."""
        # Create contracts directory structure
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"

        # Write test config
        test_config = {
            "input_topic": "test-requests",
            "output_topic": "test-responses",
            "group_id": "test-group",
        }
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(test_config, f)

        # Load config
        config = load_runtime_config(tmp_path)

        assert isinstance(config, ModelRuntimeConfig)
        assert config.input_topic == "test-requests"
        assert config.output_topic == "test-responses"
        assert config.consumer_group == "test-group"

    def test_load_config_file_not_found_uses_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that missing config file returns ModelRuntimeConfig with defaults."""
        # Clear env vars to ensure we test true defaults
        monkeypatch.delenv("ONEX_INPUT_TOPIC", raising=False)
        monkeypatch.delenv("ONEX_OUTPUT_TOPIC", raising=False)
        monkeypatch.delenv("ONEX_GROUP_ID", raising=False)

        config = load_runtime_config(tmp_path)

        assert isinstance(config, ModelRuntimeConfig)
        assert config.input_topic == DEFAULT_INPUT_TOPIC
        assert config.output_topic == DEFAULT_OUTPUT_TOPIC
        assert config.consumer_group == DEFAULT_GROUP_ID

    def test_load_config_with_env_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that environment variables override defaults when no config file."""
        monkeypatch.setenv("ONEX_INPUT_TOPIC", "env-requests")
        monkeypatch.setenv("ONEX_OUTPUT_TOPIC", "env-responses")
        monkeypatch.setenv("ONEX_GROUP_ID", "env-group")

        config = load_runtime_config(tmp_path)

        assert isinstance(config, ModelRuntimeConfig)
        assert config.input_topic == "env-requests"
        assert config.output_topic == "env-responses"
        assert config.consumer_group == "env-group"

    def test_load_config_empty_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading from empty YAML file returns ModelRuntimeConfig with defaults."""
        # Clear env vars to ensure we test true defaults from ModelRuntimeConfig
        monkeypatch.delenv("ONEX_INPUT_TOPIC", raising=False)
        monkeypatch.delenv("ONEX_OUTPUT_TOPIC", raising=False)
        monkeypatch.delenv("ONEX_GROUP_ID", raising=False)

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"

        # Write empty file
        config_file.write_text("")

        config = load_runtime_config(tmp_path)

        # Empty YAML parses as None (empty dict), ModelRuntimeConfig uses defaults
        assert isinstance(config, ModelRuntimeConfig)
        assert config.input_topic == DEFAULT_INPUT_TOPIC
        assert config.output_topic == DEFAULT_OUTPUT_TOPIC
        assert config.consumer_group == DEFAULT_GROUP_ID

    def test_load_config_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid YAML raises ProtocolConfigurationError."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"

        # Write invalid YAML
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path)

        assert "Failed to parse runtime config YAML" in str(exc_info.value)

    def test_load_config_contract_validation_fails(self, tmp_path: Path) -> None:
        """Test that contract validation errors raise ProtocolConfigurationError."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"

        # Write config with invalid topic name (spaces not allowed)
        test_config = {
            "input_topic": "invalid topic with spaces",
        }
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(test_config, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path)

        assert "Contract validation failed" in str(exc_info.value)
        assert "input_topic" in str(exc_info.value)

    def test_load_config_contract_validation_multiple_errors(
        self, tmp_path: Path
    ) -> None:
        """Test that multiple contract validation errors are reported with full error list."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"

        # Write config with multiple invalid fields
        test_config = {
            "input_topic": "invalid topic",
            "output_topic": "also invalid",
            "event_bus": {"type": "unknown-type"},
        }
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(test_config, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path)

        error = exc_info.value
        assert "Contract validation failed" in str(error)
        # Should report error count
        assert "3 error(s)" in str(error)

        # Verify structured error context is available via model.context
        # The error includes validation_errors and error_count in kwargs
        assert hasattr(error, "model")
        error_context = error.model.context
        assert error_context is not None
        # validation_errors contains full list (not truncated)
        assert "validation_errors" in error_context
        assert isinstance(error_context["validation_errors"], list)
        assert len(error_context["validation_errors"]) == 3
        # error_count is available for structured access
        assert "error_count" in error_context
        assert error_context["error_count"] == 3

    def test_load_config_contract_validation_more_than_three_errors(
        self, tmp_path: Path
    ) -> None:
        """Test that 4+ errors show 'and N more...' in message but full list in context."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"

        # Write config with 4+ invalid fields to trigger "(and N more...)"
        test_config = {
            "input_topic": "invalid topic",
            "output_topic": "also invalid",
            "consumer_group": "bad group name",
            "event_bus": {"type": "unknown-type"},
        }
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(test_config, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path)

        error = exc_info.value
        error_message = str(error)

        # Should have 4 errors total
        assert "4 error(s)" in error_message
        # Message should include "(and 1 more...)" since only first 3 are shown
        assert "(and 1 more...)" in error_message

        # Verify full error list is in structured context
        assert hasattr(error, "model")
        error_context = error.model.context
        assert error_context is not None
        # validation_errors contains ALL errors, not truncated
        assert "validation_errors" in error_context
        assert isinstance(error_context["validation_errors"], list)
        assert len(error_context["validation_errors"]) == 4
        # error_count matches full count
        assert "error_count" in error_context
        assert error_context["error_count"] == 4


@pytest.mark.skipif(not _SERVICE_REGISTRY_AVAILABLE, reason=_SKIP_REASON)
class TestBootstrap:
    """Tests for the bootstrap function."""

    @pytest.fixture
    def mock_runtime_host(self) -> Generator[MagicMock, None, None]:
        """Create a mock RuntimeHostProcess.

        Uses side_effect with async no-op functions to ensure coroutines
        created by AsyncMock are properly awaited and cleaned up, avoiding
        'coroutine was never awaited' warnings when asyncio.wait_for wraps
        the stop() call.
        """

        async def noop_start() -> None:
            """Async no-op for start that completes immediately."""

        async def noop_stop() -> None:
            """Async no-op for stop that completes immediately."""

        with patch("omnibase_infra.runtime.kernel.RuntimeHostProcess") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock(side_effect=noop_start)
            mock_instance.stop = AsyncMock(side_effect=noop_stop)
            mock_instance.input_topic = "requests"
            mock_instance.output_topic = "responses"
            mock_cls.return_value = mock_instance
            yield mock_cls

    @pytest.fixture
    def mock_event_bus(self) -> Generator[MagicMock, None, None]:
        """Create a mock InMemoryEventBus."""
        with patch("omnibase_infra.runtime.kernel.InMemoryEventBus") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            yield mock_cls

    @pytest.fixture
    def mock_health_server(self) -> Generator[MagicMock, None, None]:
        """Create a mock ServiceHealth.

        Uses side_effect with async no-op functions to ensure coroutines
        are properly awaited and cleaned up.
        """

        async def noop_start() -> None:
            """Async no-op for start that completes immediately."""

        async def noop_stop() -> None:
            """Async no-op for stop that completes immediately."""

        with patch("omnibase_infra.services.service_health.ServiceHealth") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock(side_effect=noop_start)
            mock_instance.stop = AsyncMock(side_effect=noop_stop)
            mock_instance.is_running = True
            mock_cls.return_value = mock_instance
            yield mock_cls

    async def test_bootstrap_starts_and_stops_runtime(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
    ) -> None:
        """Test that bootstrap starts runtime and handles shutdown."""
        mock_instance = mock_runtime_host.return_value
        mock_health_instance = mock_health_server.return_value

        # Create a task that will set shutdown after a short delay
        async def delayed_shutdown() -> int:
            # Create a patched bootstrap that returns quickly
            with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
                event_instance = MagicMock()
                # Make wait() return immediately
                event_instance.wait = AsyncMock(return_value=None)
                mock_event.return_value = event_instance

                return await bootstrap()

        exit_code = await delayed_shutdown()

        assert exit_code == 0
        mock_instance.start.assert_called_once()
        mock_instance.stop.assert_called_once()
        mock_health_instance.start.assert_called_once()
        mock_health_instance.stop.assert_called_once()

    async def test_bootstrap_returns_error_on_unexpected_exception(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
    ) -> None:
        """Test that bootstrap returns 1 on unexpected exception."""
        mock_instance = mock_runtime_host.return_value
        mock_instance.start = AsyncMock(side_effect=Exception("Test error"))

        # Patch the shutdown event wait to avoid hanging
        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            exit_code = await bootstrap()

        assert exit_code == 1
        # Cleanup attempted via finally block
        mock_instance.stop.assert_called_once()

    async def test_bootstrap_returns_error_on_config_error(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
    ) -> None:
        """Test that bootstrap returns 1 on ProtocolConfigurationError."""
        # Force config load to raise ProtocolConfigurationError
        with patch(
            "omnibase_infra.runtime.kernel.load_runtime_config",
            side_effect=ProtocolConfigurationError("Config error"),
        ):
            exit_code = await bootstrap()

        assert exit_code == 1
        # Runtime was never created, so stop should not be called
        mock_runtime_host.return_value.stop.assert_not_called()

    async def test_bootstrap_creates_event_bus_with_environment(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that bootstrap creates event bus with correct environment."""
        monkeypatch.setenv("ONEX_ENVIRONMENT", "test-env")
        # Ensure InMemoryEventBus is used by unsetting KAFKA_BOOTSTRAP_SERVERS
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            await bootstrap()

        # Verify event bus was created with environment
        mock_event_bus.assert_called_once()
        call_kwargs = mock_event_bus.call_args[1]
        assert call_kwargs["environment"] == "test-env"

    async def test_bootstrap_creates_kafka_event_bus_when_configured(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that bootstrap creates KafkaEventBus when KAFKA_BOOTSTRAP_SERVERS is set."""
        monkeypatch.setenv("ONEX_ENVIRONMENT", "test-env")
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

        with (
            patch("omnibase_infra.runtime.kernel.KafkaEventBus") as mock_kafka_bus,
            patch(
                "omnibase_infra.runtime.kernel.InMemoryEventBus"
            ) as mock_inmemory_bus,
            patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event,
        ):
            mock_kafka_instance = MagicMock()
            mock_kafka_bus.return_value = mock_kafka_instance

            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            await bootstrap()

        # Verify KafkaEventBus was created, not InMemoryEventBus
        mock_kafka_bus.assert_called_once()
        mock_inmemory_bus.assert_not_called()

        # Verify config was passed with correct parameters
        call_kwargs = mock_kafka_bus.call_args[1]
        config = call_kwargs["config"]
        assert config.bootstrap_servers == "kafka:9092"
        assert config.environment == "test-env"

    async def test_bootstrap_uses_contracts_dir_from_env(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test that bootstrap uses ONEX_CONTRACTS_DIR from environment.

        Uses pytest's tmp_path fixture for automatic temporary directory cleanup.
        """
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(tmp_path))
        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            exit_code = await bootstrap()

        assert exit_code == 0

    async def test_bootstrap_handles_windows_signal_setup(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
    ) -> None:
        """Test that bootstrap sets up signal handlers on Windows."""
        import signal

        with patch("omnibase_infra.runtime.kernel.sys.platform", "win32"):
            with patch("omnibase_infra.runtime.kernel.signal.signal") as mock_signal:
                with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
                    event_instance = MagicMock()
                    event_instance.wait = AsyncMock(return_value=None)
                    mock_event.return_value = event_instance

                    exit_code = await bootstrap()

                assert exit_code == 0
                # Verify signal.signal was called for SIGINT on Windows
                mock_signal.assert_called_once()
                call_args = mock_signal.call_args
                assert call_args[0][0] == signal.SIGINT

    async def test_bootstrap_shutdown_timeout_logs_warning(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
    ) -> None:
        """Test that shutdown timeout logs warning and continues gracefully."""
        import asyncio

        mock_instance = mock_runtime_host.return_value

        # Make stop() hang indefinitely (simulating a stuck shutdown)
        async def never_complete() -> None:
            await asyncio.sleep(100)  # Will be cancelled by timeout

        mock_instance.stop = AsyncMock(side_effect=never_complete)

        # Create config with very short grace period for testing
        from omnibase_infra.runtime.models import (
            ModelRuntimeConfig,
            ModelShutdownConfig,
        )

        test_config = ModelRuntimeConfig(
            shutdown=ModelShutdownConfig(
                grace_period_seconds=0
            ),  # 0 second timeout for instant timeout
        )

        with patch(
            "omnibase_infra.runtime.kernel.load_runtime_config",
            return_value=test_config,
        ):
            with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
                event_instance = MagicMock()
                event_instance.wait = AsyncMock(return_value=None)
                mock_event.return_value = event_instance

                with patch("omnibase_infra.runtime.kernel.logger") as mock_logger:
                    exit_code = await bootstrap()

        # Should still exit successfully despite timeout
        assert exit_code == 0
        # Verify warning was logged about timeout
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "timed out" in str(call).lower()
        ]
        assert len(warning_calls) == 1
        # The warning uses %s formatting, so check for the format string and arg
        call_args = warning_calls[0][0]  # positional args tuple
        assert "timed out" in call_args[0].lower()
        assert call_args[1] == 0  # grace_period_seconds value

    async def test_bootstrap_uses_config_grace_period(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
    ) -> None:
        """Test that bootstrap uses grace_period_seconds from config."""
        # Create config with custom grace period
        from omnibase_infra.runtime.models import (
            ModelRuntimeConfig,
            ModelShutdownConfig,
        )

        test_config = ModelRuntimeConfig(
            shutdown=ModelShutdownConfig(grace_period_seconds=45),  # Custom timeout
        )

        async def mock_wait_for_impl(
            coro: Coroutine[object, object, None], *, timeout: float
        ) -> None:
            """Mock wait_for that properly closes the coroutine argument."""
            coro.close()  # Close the coroutine to prevent RuntimeWarning

        with patch(
            "omnibase_infra.runtime.kernel.load_runtime_config",
            return_value=test_config,
        ):
            with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
                event_instance = MagicMock()
                event_instance.wait = AsyncMock(return_value=None)
                mock_event.return_value = event_instance

                with patch(
                    "omnibase_infra.runtime.kernel.asyncio.wait_for",
                    side_effect=mock_wait_for_impl,
                ) as mock_wait_for:
                    exit_code = await bootstrap()

        assert exit_code == 0
        # Verify wait_for was called with correct timeout for shutdown
        # Note: wait_for may be called multiple times (once for producer start,
        # once for shutdown), so we check that at least one call used our timeout
        assert mock_wait_for.call_count >= 1
        # Find the shutdown call that used our configured grace period
        shutdown_calls = [
            call
            for call in mock_wait_for.call_args_list
            if call[1].get("timeout") == 45
        ]
        assert len(shutdown_calls) >= 1, (
            f"Expected at least one wait_for call with timeout=45, "
            f"got calls: {mock_wait_for.call_args_list}"
        )

    async def test_bootstrap_passes_container_to_service_health(
        self,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
    ) -> None:
        """Test that bootstrap passes container to ServiceHealth.

        Verifies that the ModelONEXContainer instance created during bootstrap
        is correctly passed to the ServiceHealth constructor.
        """
        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            exit_code = await bootstrap()

        assert exit_code == 0
        # Verify ServiceHealth was called with container parameter
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args.kwargs
        assert "container" in call_kwargs, (
            "Expected 'container' parameter to be passed to ServiceHealth"
        )
        # Verify the container is a ModelONEXContainer instance
        container_arg = call_kwargs["container"]
        assert isinstance(container_arg, ModelONEXContainer), (
            f"Expected container to be ModelONEXContainer, got {type(container_arg)}"
        )

    async def test_bootstrap_passes_all_required_args_to_service_health(
        self,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
    ) -> None:
        """Test that bootstrap passes all required arguments to ServiceHealth.

        Verifies that container, runtime, port, and version are all passed
        to the ServiceHealth constructor.
        """
        from omnibase_infra.runtime.kernel import KERNEL_VERSION
        from omnibase_infra.services.service_health import DEFAULT_HTTP_PORT

        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            exit_code = await bootstrap()

        assert exit_code == 0
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args.kwargs

        # Verify all required parameters are present
        expected_params = {"container", "runtime", "port", "version"}
        actual_params = set(call_kwargs.keys())
        assert expected_params == actual_params, (
            f"Expected ServiceHealth params {expected_params}, got {actual_params}"
        )

        # Verify specific values
        assert isinstance(call_kwargs["container"], ModelONEXContainer)
        assert call_kwargs["runtime"] == mock_runtime_host.return_value
        assert call_kwargs["port"] == DEFAULT_HTTP_PORT
        assert call_kwargs["version"] == KERNEL_VERSION


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_default_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that default log level is INFO."""
        monkeypatch.delenv("ONEX_LOG_LEVEL", raising=False)
        with patch("logging.basicConfig") as mock_config:
            configure_logging()

            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["level"] == 20  # logging.INFO

    def test_configure_logging_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that log level can be set via environment."""
        monkeypatch.setenv("ONEX_LOG_LEVEL", "DEBUG")
        with patch("logging.basicConfig") as mock_config:
            configure_logging()

        mock_config.assert_called_once()
        call_kwargs = mock_config.call_args[1]
        assert call_kwargs["level"] == 10  # logging.DEBUG


class TestMain:
    """Tests for main entry point."""

    def test_main_calls_bootstrap(self) -> None:
        """Test that main runs bootstrap and exits with code."""

        def mock_asyncio_run(coro: Coroutine[object, object, int]) -> int:
            """Mock asyncio.run that properly closes the unawaited coroutine."""
            coro.close()  # Close the coroutine to prevent RuntimeWarning
            return 0

        with patch("omnibase_infra.runtime.kernel.configure_logging"):
            with patch(
                "omnibase_infra.runtime.kernel.asyncio.run",
                side_effect=mock_asyncio_run,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 0

    def test_main_exits_with_error_code(self) -> None:
        """Test that main exits with error code from bootstrap."""

        def mock_asyncio_run(coro: Coroutine[object, object, int]) -> int:
            """Mock asyncio.run that properly closes the unawaited coroutine."""
            coro.close()  # Close the coroutine to prevent RuntimeWarning
            return 1

        with patch("omnibase_infra.runtime.kernel.configure_logging"):
            with patch(
                "omnibase_infra.runtime.kernel.asyncio.run",
                side_effect=mock_asyncio_run,
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1


@pytest.mark.skipif(not _SERVICE_REGISTRY_AVAILABLE, reason=_SKIP_REASON)
class TestIntegration:
    """Integration tests for kernel with real components."""

    async def test_full_bootstrap_with_real_event_bus(
        self,
        mock_wire_infrastructure: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test bootstrap with real InMemoryEventBus but mocked wait and health server.

        Uses pytest's tmp_path fixture for automatic temporary directory cleanup.
        This test uses real components except for the shutdown wait and health server.
        Health server is mocked to avoid port conflicts in parallel tests.
        """
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(tmp_path))

        with patch(
            "omnibase_infra.services.service_health.ServiceHealth"
        ) as mock_health:
            mock_health_instance = MagicMock()
            mock_health_instance.start = AsyncMock()
            mock_health_instance.stop = AsyncMock()
            mock_health.return_value = mock_health_instance

            with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
                event_instance = MagicMock()
                event_instance.wait = AsyncMock(return_value=None)
                event_instance.set = MagicMock()
                mock_event.return_value = event_instance

                exit_code = await bootstrap()

        assert exit_code == 0

    async def test_bootstrap_passes_container_to_service_health_integration(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Integration test verifying container is passed to ServiceHealth.

        This test uses real components (InMemoryEventBus, RuntimeHostProcess)
        but mocks ServiceHealth to verify the container injection.
        """
        monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(tmp_path))

        with patch(
            "omnibase_infra.services.service_health.ServiceHealth"
        ) as mock_health:
            mock_health_instance = MagicMock()
            mock_health_instance.start = AsyncMock()
            mock_health_instance.stop = AsyncMock()
            mock_health.return_value = mock_health_instance

            with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
                event_instance = MagicMock()
                event_instance.wait = AsyncMock(return_value=None)
                event_instance.set = MagicMock()
                mock_event.return_value = event_instance

                exit_code = await bootstrap()

        assert exit_code == 0

        # Verify container was passed to ServiceHealth
        mock_health.assert_called_once()
        call_kwargs = mock_health.call_args.kwargs
        assert "container" in call_kwargs, (
            "Expected 'container' parameter to be passed to ServiceHealth"
        )
        # Verify the container is a ModelONEXContainer instance
        container_arg = call_kwargs["container"]
        assert isinstance(container_arg, ModelONEXContainer), (
            f"Expected container to be ModelONEXContainer, got {type(container_arg)}"
        )


@pytest.mark.skipif(not _SERVICE_REGISTRY_AVAILABLE, reason=_SKIP_REASON)
class TestHttpPortValidation:
    """Tests for HTTP port validation in bootstrap."""

    @pytest.fixture
    def mock_runtime_host(self) -> Generator[MagicMock, None, None]:
        """Create a mock RuntimeHostProcess.

        Uses side_effect with async no-op functions to ensure coroutines
        created by AsyncMock are properly awaited and cleaned up, avoiding
        'coroutine was never awaited' warnings when asyncio.wait_for wraps
        the stop() call.
        """

        async def noop_start() -> None:
            """Async no-op for start that completes immediately."""

        async def noop_stop() -> None:
            """Async no-op for stop that completes immediately."""

        with patch("omnibase_infra.runtime.kernel.RuntimeHostProcess") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock(side_effect=noop_start)
            mock_instance.stop = AsyncMock(side_effect=noop_stop)
            mock_instance.input_topic = "requests"
            mock_instance.output_topic = "responses"
            mock_cls.return_value = mock_instance
            yield mock_cls

    @pytest.fixture
    def mock_event_bus(self) -> Generator[MagicMock, None, None]:
        """Create a mock InMemoryEventBus."""
        with patch("omnibase_infra.runtime.kernel.InMemoryEventBus") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            yield mock_cls

    @pytest.fixture
    def mock_health_server(self) -> Generator[MagicMock, None, None]:
        """Create a mock ServiceHealth.

        Uses side_effect with async no-op functions to ensure coroutines
        are properly awaited and cleaned up.
        """

        async def noop_start() -> None:
            """Async no-op for start that completes immediately."""

        async def noop_stop() -> None:
            """Async no-op for stop that completes immediately."""

        with patch("omnibase_infra.services.service_health.ServiceHealth") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock(side_effect=noop_start)
            mock_instance.stop = AsyncMock(side_effect=noop_stop)
            mock_instance.is_running = True
            mock_cls.return_value = mock_instance
            yield mock_cls

    def test_port_constants_are_valid(self) -> None:
        """Test that port validation constants are correctly defined."""
        assert MIN_PORT == 1
        assert MAX_PORT == 65535

    async def test_bootstrap_rejects_port_zero(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that port 0 is rejected and falls back to default."""
        from omnibase_infra.services.service_health import DEFAULT_HTTP_PORT

        monkeypatch.setenv("ONEX_HTTP_PORT", "0")

        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            with patch("omnibase_infra.runtime.kernel.logger") as mock_logger:
                exit_code = await bootstrap()

        assert exit_code == 0
        # Verify ServiceHealth was created with default port
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args[1]
        assert call_kwargs["port"] == DEFAULT_HTTP_PORT

        # Verify warning was logged about port out of range
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "outside valid range" in str(call).lower()
        ]
        assert len(warning_calls) == 1

    async def test_bootstrap_rejects_port_above_max(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that port 65536 is rejected and falls back to default."""
        from omnibase_infra.services.service_health import DEFAULT_HTTP_PORT

        monkeypatch.setenv("ONEX_HTTP_PORT", "65536")

        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            with patch("omnibase_infra.runtime.kernel.logger") as mock_logger:
                exit_code = await bootstrap()

        assert exit_code == 0
        # Verify ServiceHealth was created with default port
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args[1]
        assert call_kwargs["port"] == DEFAULT_HTTP_PORT

        # Verify warning was logged about port out of range
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "outside valid range" in str(call).lower()
        ]
        assert len(warning_calls) == 1

    async def test_bootstrap_accepts_min_port(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that port 1 (MIN_PORT) is accepted."""
        monkeypatch.setenv("ONEX_HTTP_PORT", "1")

        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            exit_code = await bootstrap()

        assert exit_code == 0
        # Verify ServiceHealth was created with port 1
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args[1]
        assert call_kwargs["port"] == 1

    async def test_bootstrap_accepts_max_port(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that port 65535 (MAX_PORT) is accepted."""
        monkeypatch.setenv("ONEX_HTTP_PORT", "65535")

        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            exit_code = await bootstrap()

        assert exit_code == 0
        # Verify ServiceHealth was created with port 65535
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args[1]
        assert call_kwargs["port"] == 65535

    async def test_bootstrap_rejects_negative_port(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that negative port is rejected and falls back to default."""
        from omnibase_infra.services.service_health import DEFAULT_HTTP_PORT

        monkeypatch.setenv("ONEX_HTTP_PORT", "-1")

        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            with patch("omnibase_infra.runtime.kernel.logger") as mock_logger:
                exit_code = await bootstrap()

        assert exit_code == 0
        # Verify ServiceHealth was created with default port
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args[1]
        assert call_kwargs["port"] == DEFAULT_HTTP_PORT

        # Verify warning was logged about port out of range
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "outside valid range" in str(call).lower()
        ]
        assert len(warning_calls) == 1

    async def test_bootstrap_rejects_very_large_port(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that very large port number is rejected."""
        from omnibase_infra.services.service_health import DEFAULT_HTTP_PORT

        monkeypatch.setenv("ONEX_HTTP_PORT", "100000")

        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            exit_code = await bootstrap()

        assert exit_code == 0
        # Verify ServiceHealth was created with default port
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args[1]
        assert call_kwargs["port"] == DEFAULT_HTTP_PORT

    async def test_bootstrap_rejects_non_numeric_port(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that non-numeric port string is rejected and falls back to default."""
        from omnibase_infra.services.service_health import DEFAULT_HTTP_PORT

        monkeypatch.setenv("ONEX_HTTP_PORT", "not_a_number")

        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            with patch("omnibase_infra.runtime.kernel.logger") as mock_logger:
                exit_code = await bootstrap()

        assert exit_code == 0
        # Verify ServiceHealth was created with default port
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args[1]
        assert call_kwargs["port"] == DEFAULT_HTTP_PORT

        # Verify warning was logged about invalid port value
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "invalid" in str(call).lower() and "onex_http_port" in str(call).lower()
        ]
        assert len(warning_calls) == 1

    @pytest.mark.parametrize(
        ("invalid_port_value", "description"),
        [
            ("abc", "alphabetic string"),
            ("", "empty string"),
            ("   ", "whitespace only"),
            ("12abc", "mixed numeric and alphabetic"),
            ("8080.5", "decimal number string"),
            ("port", "word string"),
        ],
    )
    async def test_bootstrap_rejects_non_numeric_port_edge_cases(
        self,
        mock_wire_infrastructure: MagicMock,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        invalid_port_value: str,
        description: str,
    ) -> None:
        """Test that various non-numeric port strings are rejected and fall back to default.

        Tests edge cases including:
        - Alphabetic strings ("abc")
        - Empty strings ("")
        - Whitespace strings ("   ")
        - Mixed numeric/alphabetic ("12abc")
        - Decimal strings ("8080.5")
        """
        from omnibase_infra.services.service_health import DEFAULT_HTTP_PORT

        monkeypatch.setenv("ONEX_HTTP_PORT", invalid_port_value)

        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            with patch("omnibase_infra.runtime.kernel.logger") as mock_logger:
                exit_code = await bootstrap()

        assert exit_code == 0, f"Expected success for {description}"
        # Verify ServiceHealth was created with default port
        mock_health_server.assert_called_once()
        call_kwargs = mock_health_server.call_args[1]
        assert call_kwargs["port"] == DEFAULT_HTTP_PORT, (
            f"Expected default port for {description}"
        )

        # Verify warning was logged about invalid port value
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if "invalid" in str(call).lower() and "onex_http_port" in str(call).lower()
        ]
        assert len(warning_calls) == 1, (
            f"Expected exactly one warning logged for {description}"
        )
