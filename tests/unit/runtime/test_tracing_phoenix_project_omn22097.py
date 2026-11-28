# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Phoenix project routing for runtime spans (OMN-22097).

Measured on the dev lane Phoenix: the runtime's spans arrive (newest within a
minute of the read) but all land in the project named ``default``, which holds
tens of millions of single-span traces, so ``projects.traceCount`` for it cannot
be read inside two minutes. Phoenix routes a span to a project by the resource
attribute ``openinference.project.name``; with none it uses ``default``. The
attribute is supplied through the standard ``OTEL_RESOURCE_ATTRIBUTES`` setting,
which the SDK ``Resource.create`` merges, declared once in the ``tracing`` bundle
and mirrored in the shared runtime environment of the infra compose file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.tracing import configure_tracing

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ATTRIBUTE = "openinference.project.name"
_DECLARED = f"{_PROJECT_ATTRIBUTE}=onex-runtime"


def _tracing_bundle_env() -> dict[str, str]:
    bundles = yaml.safe_load((_ROOT / "docker/catalog/bundles.yaml").read_text())
    return dict(bundles["tracing"]["inject_env"])


def _runtime_env() -> dict[str, str]:
    compose = yaml.safe_load((_ROOT / "docker/docker-compose.infra.yml").read_text())
    return dict(compose["x-runtime-env"])


def test_tracing_bundle_declares_phoenix_project() -> None:
    assert _tracing_bundle_env().get("OTEL_RESOURCE_ATTRIBUTES") == _DECLARED


def test_infra_compose_runtime_env_declares_phoenix_project() -> None:
    assert _runtime_env().get("OTEL_RESOURCE_ATTRIBUTES") == _DECLARED


def test_configure_tracing_carries_declared_project_on_the_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider configure_tracing installs carries the declared project name."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix:6006")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "svc")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", _DECLARED)
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    installed: list[object] = []
    monkeypatch.setattr(trace, "set_tracer_provider", installed.append)
    assert configure_tracing() is True
    (provider,) = installed
    assert isinstance(provider, TracerProvider)
    attributes = provider.resource.attributes
    assert attributes[_PROJECT_ATTRIBUTE] == "onex-runtime"
    assert attributes["service.name"] == "svc"
