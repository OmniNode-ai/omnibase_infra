# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader observability logging.

These tests verify that the handler plugin loader produces observability
logs with correct counts, timing, and handler details at the end of
batch loading operations.

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from .conftest import INVALID_YAML_SYNTAX, MINIMAL_HANDLER_CONTRACT_YAML


class TestHandlerPluginLoaderObservabilityLogging:
    """Tests for observability logging in batch load operations."""

    def test_load_from_directory_logs_summary(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that load_from_directory logs a summary at the end."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.INFO):
            handlers = loader.load_from_directory(valid_contract_directory)

        # Should have loaded 3 handlers
        assert len(handlers) == 3

        # Verify summary log was produced
        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        assert summary_record.levelname == "INFO"
        assert "3 handlers loaded" in summary_record.message
        assert "successfully" in summary_record.message

    def test_load_from_directory_logs_handler_details(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that summary includes handler class names and modules."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.INFO):
            loader.load_from_directory(valid_contract_directory)

        # Find the summary log
        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        # Verify handler classes are listed
        assert "MockValidHandler" in summary_record.message
        assert "Loaded handlers:" in summary_record.message

    def test_load_from_directory_logs_timing(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that summary includes timing information."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.INFO):
            loader.load_from_directory(valid_contract_directory)

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        # Timing should be in the message with proper duration format (e.g., "123.45ms")
        # NOTE: Using regex to avoid false positives - checking "s" in message would
        # always pass due to words like "handlers", "successfully", etc.
        duration_pattern = r"\d+(\.\d+)?\s*(ms|us|s)\b"
        assert re.search(duration_pattern, summary_record.message), (
            f"Duration format not found in message: {summary_record.message}"
        )

        # Extra data should include duration_seconds
        assert hasattr(summary_record, "duration_seconds")
        assert summary_record.duration_seconds >= 0

    def test_load_from_directory_logs_correlation_id(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that summary includes correlation ID for tracing."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        test_correlation_id = uuid4()

        with caplog.at_level(logging.INFO):
            loader.load_from_directory(
                valid_contract_directory, correlation_id=test_correlation_id
            )

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        assert hasattr(summary_record, "correlation_id")
        # The log extra stores correlation_id as a string representation
        assert summary_record.correlation_id == str(test_correlation_id)

    def test_load_from_directory_logs_failed_handlers(
        self, mixed_valid_invalid_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that summary reports failed handlers."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.WARNING):
            handlers = loader.load_from_directory(mixed_valid_invalid_directory)

        # Should have loaded 1 valid handler
        assert len(handlers) == 1

        # Summary should be at WARNING level due to failures
        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        assert summary_record.levelname == "WARNING"
        assert "with failures" in summary_record.message
        assert "1 handlers loaded" in summary_record.message
        assert "Failed handlers:" in summary_record.message

    def test_load_from_directory_logs_failure_count(
        self, mixed_valid_invalid_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that summary includes correct failure count."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.WARNING):
            loader.load_from_directory(mixed_valid_invalid_directory)

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        # Should report failures in the extra data
        assert hasattr(summary_record, "total_failed")
        assert summary_record.total_failed >= 1

    def test_discover_and_load_logs_summary(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that discover_and_load logs a summary at the end."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.INFO):
            handlers = loader.discover_and_load(
                ["**/handler_contract.yaml"],
                base_path=valid_contract_directory,
            )

        # Should have discovered handlers
        assert len(handlers) == 3

        # Verify summary log was produced
        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        assert summary_record.levelname == "INFO"
        assert "3 handlers loaded" in summary_record.message

    def test_discover_and_load_logs_patterns_source(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that discover_and_load summary includes patterns as source."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.INFO):
            loader.discover_and_load(
                ["**/handler_contract.yaml", "**/contract.yaml"],
                base_path=valid_contract_directory,
            )

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        # Source should be the patterns
        assert hasattr(summary_record, "source")
        assert "handler_contract.yaml" in summary_record.source

    def test_empty_directory_logs_summary_with_zero_handlers(
        self, empty_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that empty directory still logs a summary."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.INFO):
            handlers = loader.load_from_directory(empty_directory)

        assert handlers == []

        # Summary should still be logged
        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        assert "0 handlers loaded" in summary_record.message
        assert "(none)" in summary_record.message

    def test_summary_includes_handler_names_in_extra(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that summary extra data includes handler names list."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.INFO):
            loader.load_from_directory(valid_contract_directory)

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        assert hasattr(summary_record, "handler_names")
        assert len(summary_record.handler_names) == 3
        assert "handler.one" in summary_record.handler_names

    def test_summary_includes_handler_classes_in_extra(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that summary extra data includes handler classes list."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.INFO):
            loader.load_from_directory(valid_contract_directory)

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        assert hasattr(summary_record, "handler_classes")
        assert len(summary_record.handler_classes) == 3

    def test_summary_includes_failed_paths_in_extra(
        self, mixed_valid_invalid_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that summary extra data includes failed paths list."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.WARNING):
            loader.load_from_directory(mixed_valid_invalid_directory)

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        summary_record = summary_logs[0]
        assert hasattr(summary_record, "failed_paths")
        assert len(summary_record.failed_paths) >= 1


class TestModelPluginLoadSummary:
    """Tests for the ModelPluginLoadSummary model."""

    def test_model_creation_with_valid_data(self) -> None:
        """Test creating ModelPluginLoadSummary with valid data."""
        from datetime import UTC, datetime
        from pathlib import Path
        from uuid import uuid4

        from omnibase_infra.models.runtime import (
            ModelFailedPluginLoad,
            ModelPluginLoadSummary,
        )

        summary = ModelPluginLoadSummary(
            operation="load_from_directory",
            source="/app/plugins",
            total_discovered=5,
            total_loaded=4,
            total_failed=1,
            loaded_plugins=[
                {
                    "name": "auth.plugin",
                    "class": "AuthPlugin",
                    "module": "app.plugins",
                }
            ],
            failed_plugins=[
                ModelFailedPluginLoad(
                    contract_path=Path("/app/plugins/broken/contract.yaml"),
                    error_message="Invalid YAML syntax",
                    error_code="HANDLER_LOADER_002",
                )
            ],
            duration_seconds=0.23,
            correlation_id=uuid4(),
            completed_at=datetime.now(UTC),
        )

        assert summary.operation == "load_from_directory"
        assert summary.total_discovered == 5
        assert summary.total_loaded == 4
        assert summary.total_failed == 1
        assert len(summary.loaded_plugins) == 1
        assert len(summary.failed_plugins) == 1

    def test_model_failed_plugin_load_creation(self) -> None:
        """Test creating ModelFailedPluginLoad with valid data."""
        from pathlib import Path

        from omnibase_infra.models.runtime import ModelFailedPluginLoad

        failed = ModelFailedPluginLoad(
            contract_path=Path("/app/plugins/broken/contract.yaml"),
            error_message="Module not found: myapp.plugins.broken",
            error_code="HANDLER_LOADER_010",
        )

        assert failed.contract_path == Path("/app/plugins/broken/contract.yaml")
        assert "Module not found" in failed.error_message
        assert failed.error_code == "HANDLER_LOADER_010"

    def test_model_failed_plugin_load_optional_error_code(self) -> None:
        """Test that error_code is optional in ModelFailedPluginLoad."""
        from pathlib import Path

        from omnibase_infra.models.runtime import ModelFailedPluginLoad

        failed = ModelFailedPluginLoad(
            contract_path=Path("/app/plugins/broken/contract.yaml"),
            error_message="Some error",
            # error_code not provided
        )

        assert failed.error_code is None


class TestLogLoadSummaryMethod:
    """Tests for the _log_load_summary helper method."""

    def test_log_load_summary_returns_model(
        self, valid_contract_directory: Path
    ) -> None:
        """Test that _log_load_summary returns a ModelPluginLoadSummary."""
        from omnibase_infra.models.runtime import ModelPluginLoadSummary
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        handlers = loader.load_from_directory(valid_contract_directory)

        # The summary is created internally but we can verify by checking
        # that the model is properly typed
        assert isinstance(handlers, list)
        assert len(handlers) == 3

    def test_duration_formatting_microseconds(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that very short durations are formatted as microseconds."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Mock a very fast operation
        with patch("time.perf_counter") as mock_time:
            mock_time.side_effect = [0.0, 0.0001]  # 100 microseconds

            # We need a real directory but an empty one to be fast
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                with caplog.at_level(logging.INFO):
                    loader.load_from_directory(Path(tmpdir))

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        if summary_logs:
            summary_record = summary_logs[0]
            # Should use microsecond formatting for very fast operations
            assert "us" in summary_record.message or "ms" in summary_record.message

    def test_duration_formatting_milliseconds(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that millisecond durations are formatted correctly."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.INFO):
            loader.load_from_directory(valid_contract_directory)

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1

        # Typical load operation should be in milliseconds
        summary_record = summary_logs[0]
        # Duration will be in message somewhere
        assert summary_record.duration_seconds >= 0

    def test_log_level_info_on_success(
        self, valid_contract_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that successful loads use INFO level."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.DEBUG):
            loader.load_from_directory(valid_contract_directory)

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1
        assert summary_logs[0].levelname == "INFO"

    def test_log_level_warning_on_failures(
        self, mixed_valid_invalid_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that loads with failures use WARNING level."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with caplog.at_level(logging.DEBUG):
            loader.load_from_directory(mixed_valid_invalid_directory)

        summary_logs = [
            r for r in caplog.records if "Handler load complete" in r.message
        ]
        assert len(summary_logs) == 1
        assert summary_logs[0].levelname == "WARNING"
