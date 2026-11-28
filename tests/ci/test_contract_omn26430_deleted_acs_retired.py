# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26430: ``contracts/OMN-26430.yaml`` no longer claims the deleted AC3, AC4 and AC5.

The ticket carries AC1 and AC2 only. A contract that still declares AC3, AC4 or AC5
as acceptance criteria claims criteria the ticket no longer has, and the items that
bound them keep proving them. The contract retires each of the three bindings with a
``supersedes_ac_binding`` entry on an item that binds only the live criteria. The read is
of the contract as committed at ``HEAD``: the hosted control copies the new contract
over the older commit's tree, so a working-tree read could not tell the two apart,
while the committed read at the older commit still declares the three criteria and
the test fails there.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT = "contracts/OMN-26430.yaml"
_LIVE = {"AC1", "AC2"}
_DELETED = {"AC3", "AC4", "AC5"}


def _committed_contract() -> dict[str, Any]:
    """Return the contract as committed at HEAD, never the working tree."""
    text = subprocess.run(
        ["git", "show", f"HEAD:{_CONTRACT}"],
        cwd=_ROOT,
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    document = yaml.safe_load(text)
    assert isinstance(document, dict)
    return document


def test_contract_declares_only_the_criteria_the_ticket_carries() -> None:
    declared = {
        a["id"] for r in _committed_contract()["requirements"] for a in r["acceptance"]
    }
    assert declared == _LIVE, declared


def test_each_deleted_criterion_binding_is_retired_by_an_item_that_binds_only_live_ones() -> (
    None
):
    retired: dict[str, str] = {}
    for item in _committed_contract()["dod_evidence"]:
        for entry in item.get("supersedes_ac_binding") or []:
            assert set(item.get("binds_ac") or []) <= _LIVE, item["id"]
            retired[entry["label"]] = entry["item"]
    assert set(retired) == _DELETED, retired
