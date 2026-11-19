# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25587: the OpenTelemetry instrumentation ranges admit the patched exporter.

The six instrumentation requirements were capped at ``<0.66``. That cap excluded
the 0.66 beta family, whose semantic-conventions pin is the one the patched OTLP
HTTP exporter (1.45.2) needs, so the exporter could not install next to the
FastAPI instrumentation. The six ranges now read ``>=0.48b0,<0.67`` and the lock
resolves the exporter at 1.45.2 with the instrumentation family at 0.66b2.

Each test reads the manifest, the lock or the installed environment, so each one
fails on a tree that still holds the ``<0.66`` cap and the 1.40.0 exporter.
"""

from __future__ import annotations

import importlib
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]

INSTRUMENTATIONS = (
    "opentelemetry-instrumentation",
    "opentelemetry-instrumentation-fastapi",
    "opentelemetry-instrumentation-asyncpg",
    "opentelemetry-instrumentation-aiohttp-client",
    "opentelemetry-instrumentation-kafka-python",
    "opentelemetry-instrumentation-redis",
)

EXPORTER = "opentelemetry-exporter-otlp-proto-http"
PATCHED_EXPORTER = Version("1.45.2")

# Every opentelemetry requirement the manifest declares, exactly: the three
# core and exporter ranges are untouched, and only the six instrumentation
# upper bounds moved from 0.66 to 0.67.
EXPECTED_OTEL_REQUIREMENTS = frozenset(
    {
        "opentelemetry-api>=1.27.0,<2.0.0",
        "opentelemetry-sdk>=1.27.0,<2.0.0",
        "opentelemetry-exporter-otlp-proto-http>=1.27.0,<2.0.0",
        *(f"{name}>=0.48b0,<0.67" for name in INSTRUMENTATIONS),
    }
)

INSTRUMENTOR_IMPORTS = (
    ("opentelemetry.instrumentation.fastapi", "FastAPIInstrumentor"),
    ("opentelemetry.instrumentation.asyncpg", "AsyncPGInstrumentor"),
    ("opentelemetry.instrumentation.aiohttp_client", "AioHttpClientInstrumentor"),
    ("opentelemetry.instrumentation.kafka", "KafkaInstrumentor"),
    ("opentelemetry.instrumentation.redis", "RedisInstrumentor"),
)


def _manifest_dependencies() -> list[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as manifest:
        return list(tomllib.load(manifest)["project"]["dependencies"])


def _locked_versions() -> dict[str, Version]:
    with (REPO_ROOT / "uv.lock").open("rb") as lock:
        return {
            package["name"]: Version(package["version"])
            for package in tomllib.load(lock)["package"]
        }


@pytest.mark.unit
def test_manifest_admits_the_066_instrumentation_family() -> None:
    dependencies = set(_manifest_dependencies())
    for name in INSTRUMENTATIONS:
        assert f"{name}>=0.48b0,<0.67" in dependencies, name
    assert not [
        entry
        for entry in dependencies
        if entry.startswith("opentelemetry-instrumentation") and entry.endswith("<0.66")
    ]


@pytest.mark.unit
def test_manifest_changes_only_the_six_instrumentation_ranges() -> None:
    otel_requirements = {
        entry
        for entry in _manifest_dependencies()
        if entry.startswith("opentelemetry-")
    }
    assert otel_requirements == EXPECTED_OTEL_REQUIREMENTS


@pytest.mark.unit
def test_lock_pairs_the_patched_exporter_with_the_066_family() -> None:
    locked = _locked_versions()
    assert locked[EXPORTER] >= PATCHED_EXPORTER, locked[EXPORTER]
    for name in INSTRUMENTATIONS:
        resolved = locked[name]
        assert resolved.release[:2] == (0, 66), (name, resolved)
        assert resolved.pre is not None and resolved.pre[0] == "b", (name, resolved)


@pytest.mark.unit
def test_installed_environment_has_the_patched_exporter_and_importable_instrumentors() -> (
    None
):
    assert Version(version(EXPORTER)) >= PATCHED_EXPORTER
    for module_name, class_name in INSTRUMENTOR_IMPORTS:
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), (module_name, class_name)
