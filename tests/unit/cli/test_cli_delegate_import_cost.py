# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Guard OMN-19444 AC2: importing delegate must avoid heavy handler dependencies.

Fresh interpreters expose the complete import tree, while positive controls
verify that the importtime detector reports each forbidden package.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest


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
def test_delegate_entry_point_imports_no_heavy_modules() -> None:
    """Importing the delegate leaf must not load the heavy handler tree."""
    modules = _imported_modules("import omnibase_infra.cli.cli_delegate")
    assert "omnibase_infra.cli.cli_delegate" in modules
    prefixes = ("omnibase_infra.gateway", "qdrant_client", "fastapi")
    offending_modules = sorted(
        name
        for name in modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )
    assert not offending_modules, f"Heavy modules imported: {offending_modules}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("statement", "module_name"),
    [
        ("import omnibase_infra.gateway", "omnibase_infra.gateway"),
        ("import qdrant_client", "qdrant_client"),
        ("import fastapi", "fastapi"),
    ],
)
def test_importtime_detects_heavy_modules(statement: str, module_name: str) -> None:
    """Each forbidden package must be detectable when explicitly imported."""
    assert module_name in _imported_modules(statement)
