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

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from omnibase_infra.runtime.kernel import (
    DEFAULT_CONTRACTS_DIR,
    DEFAULT_GROUP_ID,
    DEFAULT_INPUT_TOPIC,
    DEFAULT_OUTPUT_TOPIC,
    bootstrap,
    configure_logging,
    load_runtime_config,
    main,
)

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

        assert config["input_topic"] == "test-requests"
        assert config["output_topic"] == "test-responses"
        assert config["group_id"] == "test-group"

    def test_load_config_file_not_found_uses_defaults(self, tmp_path: Path) -> None:
        """Test that missing config file returns defaults."""
        config = load_runtime_config(tmp_path)

        assert config["input_topic"] == DEFAULT_INPUT_TOPIC
        assert config["output_topic"] == DEFAULT_OUTPUT_TOPIC
        assert config["group_id"] == DEFAULT_GROUP_ID

    def test_load_config_with_env_overrides(self, tmp_path: Path) -> None:
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "ONEX_INPUT_TOPIC": "env-requests",
                "ONEX_OUTPUT_TOPIC": "env-responses",
                "ONEX_GROUP_ID": "env-group",
            },
        ):
            config = load_runtime_config(tmp_path)

            assert config["input_topic"] == "env-requests"
            assert config["output_topic"] == "env-responses"
            assert config["group_id"] == "env-group"

    def test_load_config_empty_yaml(self, tmp_path: Path) -> None:
        """Test loading from empty YAML file returns empty dict."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"

        # Write empty file
        config_file.write_text("")

        config = load_runtime_config(tmp_path)

        # Empty YAML parses as None, should return empty dict
        assert config == {}


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

    async def test_bootstrap_starts_and_stops_runtime(
        self, mock_runtime_host: MagicMock, mock_event_bus: MagicMock
    ) -> None:
        """Test that bootstrap starts runtime and handles shutdown."""
        mock_instance = mock_runtime_host.return_value

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

    async def test_bootstrap_returns_error_on_exception(
        self, mock_runtime_host: MagicMock, mock_event_bus: MagicMock
    ) -> None:
        """Test that bootstrap returns error code on exception."""
        mock_instance = mock_runtime_host.return_value
        mock_instance.start = AsyncMock(side_effect=Exception("Test error"))

        # Patch the shutdown event wait to avoid hanging
        with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
            event_instance = MagicMock()
            event_instance.wait = AsyncMock(return_value=None)
            mock_event.return_value = event_instance

            exit_code = await bootstrap()

        assert exit_code == 1
        mock_instance.stop.assert_called_once()  # Cleanup attempted

    async def test_bootstrap_creates_event_bus_with_environment(
        self, mock_runtime_host: MagicMock, mock_event_bus: MagicMock
    ) -> None:
        """Test that bootstrap creates event bus with correct environment."""
        with patch.dict(os.environ, {"ONEX_ENVIRONMENT": "test-env"}):
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
        self, mock_runtime_host: MagicMock, mock_event_bus: MagicMock
    ) -> None:
        """Test that bootstrap uses CONTRACTS_DIR from environment."""
        with TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"CONTRACTS_DIR": tmpdir}):
                with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
                    event_instance = MagicMock()
                    event_instance.wait = AsyncMock(return_value=None)
                    mock_event.return_value = event_instance

                    exit_code = await bootstrap()

        assert exit_code == 0


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_default_level(self) -> None:
        """Test that default log level is INFO."""
        with patch("logging.basicConfig") as mock_config:
            with patch.dict(os.environ, {}, clear=True):
                # Remove ONEX_LOG_LEVEL if present
                os.environ.pop("ONEX_LOG_LEVEL", None)
                configure_logging()

            mock_config.assert_called_once()
            # Check that level is INFO (20)
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["level"] == 20  # logging.INFO

    def test_configure_logging_from_env(self) -> None:
        """Test that log level can be set via environment."""
        with patch("logging.basicConfig") as mock_config:
            with patch.dict(os.environ, {"ONEX_LOG_LEVEL": "DEBUG"}):
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

    async def test_full_bootstrap_with_real_event_bus(self) -> None:
        """Test bootstrap with real InMemoryEventBus but mocked wait."""
        # This test uses real components except for the shutdown wait
        with TemporaryDirectory() as tmpdir:
            contracts_dir = Path(tmpdir)

            with patch.dict(os.environ, {"CONTRACTS_DIR": str(contracts_dir)}):
                with patch("omnibase_infra.runtime.kernel.asyncio.Event") as mock_event:
                    event_instance = MagicMock()
                    event_instance.wait = AsyncMock(return_value=None)
                    event_instance.set = MagicMock()
                    mock_event.return_value = event_instance

                    exit_code = await bootstrap()

        assert exit_code == 0
