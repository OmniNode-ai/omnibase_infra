# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for opt-in OpenTelemetry tracing configuration (OMN-3811)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.tracing import configure_tracing


@pytest.mark.unit
class TestConfigureTracing:
    """Tests for configure_tracing() opt-in behaviour."""

    def test_skips_when_endpoint_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tracing is silently skipped when OTEL_EXPORTER_OTLP_ENDPOINT is absent."""
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert configure_tracing() is False

    def test_skips_when_endpoint_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tracing is silently skipped when OTEL_EXPORTER_OTLP_ENDPOINT is empty."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        assert configure_tracing() is False

    def test_skips_when_endpoint_whitespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tracing is silently skipped when OTEL_EXPORTER_OTLP_ENDPOINT is whitespace."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "  ")
        assert configure_tracing() is False

    def test_skips_when_exporter_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tracing is disabled when OTEL_TRACES_EXPORTER=none."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix:6006")
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")
        assert configure_tracing() is False

    def test_configures_when_endpoint_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tracing is configured when OTEL_EXPORTER_OTLP_ENDPOINT is provided."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix:6006")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "test-service")
        monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)

        mock_provider = MagicMock()
        mock_exporter = MagicMock()
        mock_resource = MagicMock()
        mock_trace = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {
                    "opentelemetry": MagicMock(trace=mock_trace),
                    "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(
                        OTLPSpanExporter=MagicMock(return_value=mock_exporter),
                    ),
                    "opentelemetry.sdk.resources": MagicMock(
                        Resource=mock_resource,
                    ),
                    "opentelemetry.sdk.trace": MagicMock(
                        TracerProvider=MagicMock(return_value=mock_provider),
                    ),
                    "opentelemetry.sdk.trace.export": MagicMock(),
                },
            ),
        ):
            result = configure_tracing()

        assert result is True
        mock_resource.create.assert_called_once_with({"service.name": "test-service"})
        mock_trace.set_tracer_provider.assert_called_once_with(mock_provider)

    def test_defaults_service_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Service name defaults to 'onex-runtime' when not set."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix:6006")
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)

        mock_resource = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(),
                "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(),
                "opentelemetry.sdk.resources": MagicMock(Resource=mock_resource),
                "opentelemetry.sdk.trace": MagicMock(
                    TracerProvider=MagicMock(return_value=MagicMock()),
                ),
                "opentelemetry.sdk.trace.export": MagicMock(),
            },
        ):
            configure_tracing()

        mock_resource.create.assert_called_once_with({"service.name": "onex-runtime"})

    def test_returns_false_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False gracefully if OTEL setup raises an error."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix:6006")
        monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)

        # Simulate an import/setup failure
        broken_module = MagicMock()
        broken_module.Resource.create.side_effect = RuntimeError("boom")

        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(),
                "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(),
                "opentelemetry.sdk.resources": broken_module,
                "opentelemetry.sdk.trace": MagicMock(),
                "opentelemetry.sdk.trace.export": MagicMock(),
            },
        ):
            result = configure_tracing()

        assert result is False
