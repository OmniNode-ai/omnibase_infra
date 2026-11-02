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

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from omnibase_infra.runtime.tracing import configure_tracing

pytestmark = pytest.mark.unit

_BASE = "http://phoenix:6006"
_TRACES = "http://phoenix:6006/v1/traces"


@pytest.fixture
def captured_exporters(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[object]]:
    """Run ``configure_tracing`` with a real exporter and capture it, no global provider."""
    monkeypatch.setenv("OTEL_SERVICE_NAME", "test-service")
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    exporters: list[object] = []

    def _capture(exporter: object) -> object:
        exporters.append(exporter)
        return object()

    with (
        patch("opentelemetry.sdk.trace.export.BatchSpanProcessor", _capture),
        patch("opentelemetry.trace.set_tracer_provider"),
        patch("opentelemetry.sdk.trace.TracerProvider.add_span_processor"),
    ):
        yield exporters


class TestSdkEndpointSemantics:
    """The recorded probe: passing ``endpoint=`` skips the SDK's signal path."""

    def test_endpoint_keyword_is_used_verbatim(self) -> None:
        assert OTLPSpanExporter(endpoint=_BASE)._endpoint == _BASE

    def test_environment_resolution_appends_traces_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", _BASE)
        assert OTLPSpanExporter()._endpoint == _TRACES


class TestConfigureTracingExporterUrl:
    """``configure_tracing`` must target ``<endpoint>/v1/traces``."""

    @pytest.mark.parametrize(
        "configured",
        [_BASE, f"{_BASE}/", _TRACES],
        ids=["bare", "trailing-slash", "already-traces-path"],
    )
    def test_exporter_targets_traces_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        captured_exporters: list[object],
        configured: str,
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", configured)

        assert configure_tracing() is True

        (exporter,) = captured_exporters
        assert isinstance(exporter, OTLPSpanExporter)
        assert exporter._endpoint == _TRACES


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
