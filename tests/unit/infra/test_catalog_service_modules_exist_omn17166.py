# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every catalog service that runs an omnibase_infra module names a module that exists.

OMN-17166. ``docker/catalog/services/savings-estimation-consumer.yaml`` kept
declaring ``python -m omnibase_infra.services.observability.savings_estimation.consumer``
after OMN-16293 (omnibase_infra#2818, 2026-08-23) deleted that module, and the
``runtime-observability-projections`` bundle (part of ``runtime``) kept listing the
service. Any lane bring-up that rendered the bundle started a container that exits
with ``ModuleNotFoundError`` under ``restart: unless-stopped``.

This is the census the ticket asks for: it reads every catalog service file and
checks each ``python -m omnibase_infra.<module>`` command against the source tree.
It does not import the module, so it needs no runtime dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVICES_DIR = _REPO_ROOT / "docker" / "catalog" / "services"
_BUNDLES_FILE = _REPO_ROOT / "docker" / "catalog" / "bundles.yaml"
_SRC_DIR = _REPO_ROOT / "src"
_PACKAGE = "omnibase_infra"


def _module_from_command(command: object) -> str | None:
    """Return the module after ``-m`` in a ``python -m <module>`` command, or None."""
    if not isinstance(command, list):
        return None
    tokens = [str(token) for token in command]
    if "-m" not in tokens:
        return None
    index = tokens.index("-m")
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def _module_exists(module: str) -> bool:
    path = _SRC_DIR.joinpath(*module.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _catalog_modules() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for service_file in sorted(_SERVICES_DIR.glob("*.yaml")):
        data = yaml.safe_load(service_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        module = _module_from_command(data.get("command"))
        if module is not None and module.split(".")[0] == _PACKAGE:
            pairs.append((service_file.stem, module))
    return pairs


@pytest.mark.unit
def test_census_reads_the_catalog() -> None:
    """Positive control: the census finds omnibase_infra modules, so a zero is real."""
    services = {service for service, _ in _catalog_modules()}
    assert "agent-actions-consumer" in services
    assert len(services) >= 8


@pytest.mark.unit
@pytest.mark.parametrize(
    ("service", "module"),
    _catalog_modules(),
    ids=[service for service, _ in _catalog_modules()],
)
def test_catalog_service_module_exists(service: str, module: str) -> None:
    assert _module_exists(module), (
        f"catalog service {service!r} runs python -m {module}, "
        f"but src/{module.replace('.', '/')}(.py|/__init__.py) does not exist"
    )


@pytest.mark.unit
def test_every_bundle_service_has_a_catalog_file() -> None:
    """A bundle may not list a service that has no catalog service file."""
    bundles = yaml.safe_load(_BUNDLES_FILE.read_text(encoding="utf-8"))
    known: set[str] = set()
    for path in _SERVICES_DIR.glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            known.add(data["name"])
    missing = sorted(
        f"{bundle_name}:{service}"
        for bundle_name, bundle in bundles.items()
        if isinstance(bundle, dict)
        for service in bundle.get("services", [])
        if service not in known
    )
    assert not missing, f"bundle services with no catalog service file: {missing}"
