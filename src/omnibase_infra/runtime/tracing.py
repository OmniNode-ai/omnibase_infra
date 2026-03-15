# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Opt-in OpenTelemetry tracing configuration for the ONEX runtime.

Exports traces to an OTLP HTTP endpoint (e.g. Arize Phoenix on port 6006).
Activation is controlled entirely by environment variables — if
``OTEL_EXPORTER_OTLP_ENDPOINT`` is unset or empty, tracing is silently
skipped and the runtime operates without any OTEL overhead.

Environment Variables:
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP HTTP endpoint URL (e.g. ``http://phoenix:6006``).
        When empty or unset, tracing is disabled.
    OTEL_SERVICE_NAME: Logical service name attached to all spans.
        Defaults to ``onex-runtime``.
    OTEL_TRACES_EXPORTER: Exporter type. Only ``otlp`` (the default) is
        currently supported; set to ``none`` to explicitly disable.

OMN-3811: Initial instrumentation for Phoenix OTEL traces.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def configure_tracing() -> bool:
    """Configure OpenTelemetry tracing if an OTLP endpoint is available.

    Returns:
        ``True`` if tracing was successfully configured, ``False`` if skipped
        (no endpoint configured) or if an error occurred during setup.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        logger.debug("OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled")
        return False

    exporter_type = os.environ.get("OTEL_TRACES_EXPORTER", "otlp").strip()
    if exporter_type == "none":
        logger.info("OTEL_TRACES_EXPORTER=none — tracing explicitly disabled")
        return False

    service_name = os.environ.get("OTEL_SERVICE_NAME", "onex-runtime").strip()

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # Phoenix accepts OTLP HTTP on its main port (6006 by default).
        # The OTLPSpanExporter appends /v1/traces automatically.
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        logger.info(
            "OpenTelemetry tracing configured: endpoint=%s service=%s",
            endpoint,
            service_name,
        )
        return True

    except Exception:
        logger.warning(
            "Failed to configure OpenTelemetry tracing — continuing without traces",
            exc_info=True,
        )
        return False
