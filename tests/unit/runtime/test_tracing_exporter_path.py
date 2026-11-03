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

import re
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
        ("configured", "expected"),
        [
            (_BASE, _TRACES),
            (f"{_BASE}/", _TRACES),
            (_TRACES, _TRACES),
            (f"{_TRACES}/", _TRACES),
            (f"{_BASE}/proxy/phoenix", f"{_BASE}/proxy/phoenix/v1/traces"),
            (f"{_BASE}/proxy/phoenix/v1/traces", f"{_BASE}/proxy/phoenix/v1/traces"),
            (f"{_BASE}?tenant=a", f"{_TRACES}?tenant=a"),
            (f"{_BASE}/proxy/?tenant=a", f"{_BASE}/proxy/v1/traces?tenant=a"),
        ],
        ids=[
            "bare",
            "trailing-slash",
            "already-traces-path",
            "traces-trailing-slash",
            "path-prefix",
            "path-prefix-already-traces",
            "query-string",
            "path-prefix-and-query",
        ],
    )
    def test_exporter_targets_traces_path(
        self, monkeypatch: pytest.MonkeyPatch, configured: str, expected: str
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

        exporter.assert_called_once_with(endpoint=expected)
        processor.assert_called_once_with(exporter.return_value)
        provider.return_value.add_span_processor.assert_called_once_with(
            processor.return_value
        )
        install.assert_called_once_with(provider.return_value)

    @pytest.mark.parametrize("configured", ["phoenix:6006", "ftp://phoenix", "http://"])
    def test_endpoint_without_an_http_url_disables_tracing(
        self, monkeypatch: pytest.MonkeyPatch, configured: str
    ) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", configured)
        monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
        with (
            patch(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter"
            ) as exporter,
            patch("opentelemetry.trace.set_tracer_provider") as install,
        ):
            assert configure_tracing() is False

        exporter.assert_not_called()
        install.assert_not_called()


# Everything OMN-22097 adds or edits, plus the contract that supplies the endpoint. The
# repository tracks other files that legitimately name private addresses (fleet and
# routing config, unrelated tests), so the scan is scoped to this change's own files.
_CHANGE_FILES = (
    "contracts/OMN-22097.yaml",
    "src/omnibase_infra/handlers/dispatch_span.py",
    "src/omnibase_infra/runtime/message_dispatch_engine.py",
    "src/omnibase_infra/runtime/tracing.py",
    "src/omnibase_infra/runtime/tracing_contract.yaml",
    "tests/integration/runtime/test_dispatch_tracing_http_omn22097.py",
    "tests/unit/runtime/test_dispatch_engine_span.py",
    "tests/unit/runtime/test_tracing_exporter_path.py",
)
_PRIVATE_ADDRESS = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2})\b"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]


class TestNoLanAddressInChangedFiles:
    """AC4: the Phoenix endpoint comes from the container environment, never a literal."""

    @pytest.mark.parametrize("relative", _CHANGE_FILES)
    def test_file_names_no_private_address(self, relative: str) -> None:
        path = _REPO_ROOT / relative
        assert path.is_file(), f"{relative} moved or was deleted; update _CHANGE_FILES"
        hit = _PRIVATE_ADDRESS.search(path.read_text(encoding="utf-8"))
        assert hit is None, f"{relative} names a private address"

    def test_pattern_matches_private_ranges_only(self) -> None:
        # Dotted quads are assembled here so this file does not name one itself.
        def quad(*parts: str) -> str:
            return ".".join(parts)

        for address in (
            quad("192", "168", "1", "2"),
            quad("10", "0", "0", "9"),
            quad("172", "16", "4", "1"),
            quad("172", "31", "255", "1"),
        ):
            assert _PRIVATE_ADDRESS.search(f"http://{address}:6006")
        for text in (
            quad("172", "32", "0", "1"),
            quad("8", "8", "8", "8"),
            "http://phoenix:6006",
        ):
            assert _PRIVATE_ADDRESS.search(text) is None
