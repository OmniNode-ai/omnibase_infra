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

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.kernel import (
    DEFAULT_GROUP_ID,
    DEFAULT_INPUT_TOPIC,
    DEFAULT_OUTPUT_TOPIC,
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
        with open(config_file, "w") as f:
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


class TestBootstrap:
    """Tests for the bootstrap function."""

    @pytest.fixture
    def mock_runtime_host(self) -> Generator[MagicMock, None, None]:
        """Create a mock RuntimeHostProcess."""
        with patch("omnibase_infra.runtime.kernel.RuntimeHostProcess") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
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
        """Create a mock HealthServer."""
        with patch("omnibase_infra.runtime.kernel.HealthServer") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_instance.stop = AsyncMock()
            mock_instance.is_running = True
            mock_cls.return_value = mock_instance
            yield mock_cls

    async def test_bootstrap_starts_and_stops_runtime(
        self,
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
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that bootstrap creates event bus with correct environment."""
        monkeypatch.setenv("ONEX_ENVIRONMENT", "test-env")
        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            await bootstrap()

        # Verify event bus was created with environment
        mock_event_bus.assert_called_once()
        call_kwargs = mock_event_bus.call_args[1]
        assert call_kwargs["environment"] == "test-env"

    async def test_bootstrap_uses_contracts_dir_from_env(
        self,
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that bootstrap uses CONTRACTS_DIR from environment."""
        with TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("CONTRACTS_DIR", tmpdir)
            with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
                event_instance = MagicMock()
                event_instance.wait = AsyncMock(return_value=None)
                mock_event.return_value = event_instance

                exit_code = await bootstrap()

        assert exit_code == 0

    async def test_bootstrap_handles_windows_signal_setup(
        self,
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
        from omnibase_infra.runtime.models import ModelRuntimeConfig

        test_config = ModelRuntimeConfig(
            shutdown={
                "grace_period_seconds": 0
            },  # 0 second timeout for instant timeout
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
        mock_runtime_host: MagicMock,
        mock_event_bus: MagicMock,
        mock_health_server: MagicMock,
    ) -> None:
        """Test that bootstrap uses grace_period_seconds from config."""
        import asyncio

        mock_instance = mock_runtime_host.return_value
        mock_instance.stop = AsyncMock()

        # Create config with custom grace period
        from omnibase_infra.runtime.models import ModelRuntimeConfig

        test_config = ModelRuntimeConfig(
            shutdown={"grace_period_seconds": 45},  # Custom timeout
        )

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
                    new_callable=AsyncMock,
                ) as mock_wait_for:
                    exit_code = await bootstrap()

        assert exit_code == 0
        # Verify wait_for was called with correct timeout
        mock_wait_for.assert_called_once()
        call_kwargs = mock_wait_for.call_args[1]
        assert call_kwargs["timeout"] == 45


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
        with patch("omnibase_infra.runtime.kernel.configure_logging"):
            with patch("omnibase_infra.runtime.kernel.asyncio.run") as mock_run:
                mock_run.return_value = 0

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 0

    def test_main_exits_with_error_code(self) -> None:
        """Test that main exits with error code from bootstrap."""
        with patch("omnibase_infra.runtime.kernel.configure_logging"):
            with patch("omnibase_infra.runtime.kernel.asyncio.run") as mock_run:
                mock_run.return_value = 1

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1


class TestIntegration:
    """Integration tests for kernel with real components."""

    async def test_full_bootstrap_with_real_event_bus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test bootstrap with real InMemoryEventBus but mocked wait and health server."""
        # This test uses real components except for the shutdown wait and health server
        # Health server is mocked to avoid port conflicts in parallel tests
        with TemporaryDirectory() as tmpdir:
            contracts_dir = Path(tmpdir)
            monkeypatch.setenv("CONTRACTS_DIR", str(contracts_dir))

            with patch("omnibase_infra.runtime.kernel.HealthServer") as mock_health:
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
