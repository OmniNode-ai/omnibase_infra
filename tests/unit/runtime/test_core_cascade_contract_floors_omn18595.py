# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Core cascade evidence in OMN-18595 states version floors (OMN-18595).

Every core cascade cites OMN-18595 and the evidence gate runs every
dod_evidence item of the contract at each cascade's head. An item that
asserts the exact omnibase-core version of its own cascade therefore fails
at the head of every later cascade, which did nothing wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = REPO_ROOT / "contracts" / "OMN-18595.yaml"
EXACT_CORE_VERSION = re.compile(r"""version\(\s*['"]omnibase-core['"]\s*\)\s*==""")


def _check_values(contract: object) -> list[tuple[str, str]]:
    assert isinstance(contract, dict)
    items = contract.get("dod_evidence")
    assert isinstance(items, list)
    values: list[tuple[str, str]] = []
    for item in items:
        assert isinstance(item, dict)
        for check in item.get("checks") or []:
            assert isinstance(check, dict)
            values.append((str(item.get("id")), str(check.get("check_value", ""))))
    return values


def test_no_cascade_evidence_asserts_an_exact_core_version() -> None:
    """A later cascade's head still satisfies every earlier cascade's evidence."""
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    exact = [
        evidence_id
        for evidence_id, value in _check_values(contract)
        if EXACT_CORE_VERSION.search(value)
    ]
    assert exact == [], (
        f"{CONTRACT.name} items assert an exact omnibase-core version: {exact}; "
        "assert a floor (Version(v) >= Version('X.Y.Z')) instead"
    )


def test_the_exact_version_pattern_is_detected() -> None:
    """Positive control: the refused shape is the one cascade items used."""
    refused = (
        'uv run --frozen python -c "import importlib.metadata as m; '
        "assert m.version('omnibase-core') == '0.47.53', m.version('omnibase-core')\""
    )
    admitted = (
        'uv run --frozen python -c "import importlib.metadata as m; '
        "from packaging.version import Version as V; v = m.version('omnibase-core'); "
        "assert V(v) >= V('0.47.53'), v\""
    )
    assert EXACT_CORE_VERSION.search(refused)
    assert not EXACT_CORE_VERSION.search(admitted)
