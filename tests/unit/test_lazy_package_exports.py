# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Guard the public exports and import cycles of OMN-19444's lazy packages.

Export checks run in-process; submodule access and cycle regressions use fresh
interpreters so previously imported modules cannot mask broken import order.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import ModuleType

import pytest

LAZY_PACKAGES: tuple[str, ...] = (
    "omnibase_infra.nodes",
    "omnibase_infra.models",
    "omnibase_infra.models.projection",
    "omnibase_infra.utils",
    "omnibase_infra.runtime",
    "omnibase_infra.handlers",
    "omnibase_infra.services",
)


@pytest.mark.unit
@pytest.mark.parametrize("package_name", LAZY_PACKAGES)
def test_every_public_name_resolves(package_name: str) -> None:
    """Every declared export must be discoverable and resolve on access."""
    package = importlib.import_module(package_name)
    visible_names = set(dir(package))
    for name in package.__all__:
        assert hasattr(package, name), f"{package_name}.{name} does not resolve"
        assert name in visible_names, f"{package_name}.{name} is missing from dir()"


@pytest.mark.unit
def test_top_level_submodule_attributes_resolve() -> None:
    """Top-level attribute access and from-imports must expose module objects."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import omnibase_infra; "
            "print(type(omnibase_infra.nodes).__name__, "
            "type(omnibase_infra.models).__name__, "
            "type(omnibase_infra.enums).__name__, "
            "type(omnibase_infra.utils).__name__); "
            "from omnibase_infra import nodes, models, enums, utils",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert result.stdout.strip() == "module module module module"


@pytest.mark.unit
@pytest.mark.parametrize("package_name", LAZY_PACKAGES)
def test_unknown_attribute_raises_attribute_error(package_name: str) -> None:
    """Lazy lookup must preserve the normal missing-attribute exception."""
    package = importlib.import_module(package_name)
    missing_name = "definitely_not_exported"
    with pytest.raises(AttributeError):
        getattr(package, missing_name)


@pytest.mark.unit
def test_handler_identity_is_function_not_module() -> None:
    """Importing the identity submodule must not replace the exported function."""
    runtime = importlib.import_module("omnibase_infra.runtime")
    assert callable(runtime.handler_identity)
    assert not isinstance(runtime.handler_identity, ModuleType)

    importlib.import_module("omnibase_infra.runtime.handler_identity")

    assert callable(runtime.handler_identity)
    assert not isinstance(runtime.handler_identity, ModuleType)


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name",
    [
        "omnibase_infra.cli.cli_skill",
        "omnibase_infra.cli.cli_auth",
        "omnibase_infra.cli.cli_kafka",
        "omnibase_infra.cli.cli_node",
        "omnibase_infra.services",
        "omnibase_infra.services.contract_publisher",
        "omnibase_infra.nodes.node_contract_registry_reducer",
        "omnibase_infra.topics",
        "omnibase_infra.runtime",
        "omnibase_infra.handlers",
        "omnibase_infra.event_bus.event_bus_kafka",
        "omnibase_infra.observability",
    ],
)
def test_import_has_no_cycle_in_fresh_interpreter(module_name: str) -> None:
    """Each entry point must import without relying on a warmed module cache."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Importing {module_name} failed with exit code {result.returncode}:\n"
        f"{result.stderr[-4000:]}"
    )
