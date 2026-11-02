# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression for the OTLP exporter URL ``configure_tracing`` builds (OMN-22097).

Probe recorded inside the dev runtime container (opentelemetry-exporter-otlp-proto-http
1.40.0): ``OTLPSpanExporter(endpoint="http://phoenix:6006")`` targets
``http://phoenix:6006`` and Phoenix answers a POST to ``/`` with 405, while the
exporter built without ``endpoint=`` targets ``http://phoenix:6006/v1/traces``.
Spans only reach Phoenix when the exporter URL carries the signal path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_infra.runtime.tracing import configure_tracing

pytestmark = pytest.mark.unit

_BASE = "http://phoenix:6006"
_TRACES = "http://phoenix:6006/v1/traces"


class TestConfigureTracingExporterUrl:
    """Assert public constructor arguments and provider wiring, without SDK internals."""

    @pytest.mark.parametrize(
        "configured",
        [_BASE, f"{_BASE}/", _TRACES, f"{_TRACES}/"],
        ids=["bare", "trailing-slash", "already-traces-path", "traces-trailing-slash"],
    )
    def test_exporter_targets_traces_path(
        self, monkeypatch: pytest.MonkeyPatch, configured: str
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", configured)
        monkeypatch.setenv("OTEL_SERVICE_NAME", "test-service")
        monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
        with (
            patch(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
            ) as exporter,
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as processor,
            patch("opentelemetry.sdk.trace.TracerProvider") as provider,
            patch("opentelemetry.trace.set_tracer_provider") as install,
        ):
            assert configure_tracing() is True

        exporter.assert_called_once_with(endpoint=_TRACES)
        processor.assert_called_once_with(exporter.return_value)
        provider.return_value.add_span_processor.assert_called_once_with(
            processor.return_value
        )
        install.assert_called_once_with(provider.return_value)


class TestNoLanAddressInTrackedSource:
    """AC4: the Phoenix endpoint comes from the container environment, never a literal."""

    @pytest.mark.parametrize(
        "relative",
        [
            "src/omnibase_infra/runtime/tracing.py",
            "src/omnibase_infra/runtime/tracing_contract.yaml",
            "src/omnibase_infra/handlers/dispatch_span.py",
        ],
    )
    def test_source_names_no_lan_address(self, relative: str) -> None:
        text = (Path(__file__).resolve().parents[3] / relative).read_text(
            encoding="utf-8"
        )
        assert "192.168." not in text
