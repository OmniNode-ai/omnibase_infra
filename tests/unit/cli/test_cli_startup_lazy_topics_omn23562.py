# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Guard OMN-23562 AC2: CLI start must not load the generated topic code.

``omnibase_infra.cli.delegate_lane_credentials`` sat on a chain that loaded
``omnibase_infra.topics.platform_topic_suffixes`` (about 1.4 s) before any
command ran. Each test lists the modules a fresh interpreter loads, so a package
``__init__`` that goes back to importing the topic code eagerly fails here.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys

import pytest

_TOPIC_MODULE = "omnibase_infra.topics.platform_topic_suffixes"
_VALIDATION_MODULE = "omnibase_core.validation"


def _imported_modules(statement: str) -> set[str]:
    """Return modules recorded by importtime in a fresh, environment-inheriting process."""
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", statement],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    modules: set[str] = set()
    for line in result.stderr.splitlines():
        match = re.fullmatch(r"import time:\s*\d+\s*\|\s*\d+\s*\|\s*(\S+)\s*", line)
        if match is not None:
            modules.add(match.group(1).strip())
    return modules


@pytest.mark.unit
def test_credential_module_does_not_import_topic_module() -> None:
    """AC2: importing the credential resolver leaves the topic module unloaded."""
    modules = _imported_modules("import omnibase_infra.cli.delegate_lane_credentials")
    assert "omnibase_infra.cli.delegate_lane_credentials" in modules
    assert _TOPIC_MODULE not in modules


@pytest.mark.unit
def test_ledger_command_module_does_not_import_topic_code() -> None:
    """``onex ledger --help`` loads the ledger module; it must not pay for topics."""
    modules = _imported_modules("import omnibase_infra.cli.cli_ledger")
    assert "omnibase_infra.cli.cli_ledger" in modules
    assert _TOPIC_MODULE not in modules
    assert _VALIDATION_MODULE not in modules


@pytest.mark.unit
def test_detector_reports_topic_module_when_imported() -> None:
    """Positive control: the detector sees the topic module when it is imported."""
    assert _TOPIC_MODULE in _imported_modules(f"import {_TOPIC_MODULE}")
    assert _VALIDATION_MODULE in _imported_modules(f"import {_VALIDATION_MODULE}")


_LAZY_PACKAGES = ["omnibase_infra.event_bus", "omnibase_infra.topics"]


@pytest.mark.unit
@pytest.mark.parametrize("package", _LAZY_PACKAGES)
def test_lazy_package_exports_resolve_to_their_defining_module(package: str) -> None:
    """Every name a lazy package lists in ``__all__`` still imports from the package."""
    module = importlib.import_module(package)
    defining = [importlib.import_module(name) for name in module._EXPORT_MODULES]
    for name in module.__all__:
        resolved = getattr(module, name)
        assert any(getattr(m, name, None) is resolved for m in defining), name


@pytest.mark.unit
@pytest.mark.parametrize("package", _LAZY_PACKAGES)
def test_lazy_package_unknown_name_raises_attribute_error(package: str) -> None:
    """A name the package never exported is still an AttributeError."""
    module = importlib.import_module(package)
    missing = "not_an_export_omn23562"
    with pytest.raises(AttributeError):
        getattr(module, missing)


@pytest.mark.unit
def test_topic_resolver_still_rejects_malformed_suffix() -> None:
    """The deferred validation import still runs: a bad suffix is refused."""
    from omnibase_infra.errors import ProtocolConfigurationError
    from omnibase_infra.topics import TopicResolver

    with pytest.raises(ProtocolConfigurationError):
        TopicResolver().resolve("not-an-onex-topic")
