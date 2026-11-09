# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23362: the lane-deploy preflight's grant registry exists where it reads it.

``scripts/preflight_lane_deploy_attribution.py`` reads the prod-promotion grant
registry from ``GRANTS_REPO_RELPATH`` in this repository (OMN-17427 moved the
read here). With no file at that path every governed-lane deploy fails closed
as UNREADABLE. These tests pin that the registry is tracked at exactly that
path, that it holds no grant, and that the preflight's own evaluator reads it
as CLEAR, not UNREADABLE and not as carrying a live grant.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from scripts.preflight_lane_deploy_attribution import (
    GRANTS_REPO_RELPATH,
    EnumGrantVerdict,
    evaluate_grant_state,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = REPO_ROOT / GRANTS_REPO_RELPATH


def test_grant_registry_present_at_the_path_the_preflight_reads() -> None:
    assert REGISTRY.is_file(), f"no grant registry at {GRANTS_REPO_RELPATH}"


def test_grant_registry_present_holds_no_grant() -> None:
    document = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert document == {"entries": []}


def test_grant_registry_present_reads_clear_in_the_preflight() -> None:
    block = evaluate_grant_state(REGISTRY.read_bytes(), now=datetime.now(UTC))
    assert block["verdict"] == EnumGrantVerdict.CLEAR.value, block["errors"]
    assert block["live_grants"] == []
    assert block["errors"] == []


def test_grant_registry_present_changes_route_to_code_owners() -> None:
    codeowners = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    owner_lines = [
        line.split()
        for line in codeowners.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert any(
        parts[0] == f"/{GRANTS_REPO_RELPATH}" and len(parts) > 1
        for parts in owner_lines
    )
