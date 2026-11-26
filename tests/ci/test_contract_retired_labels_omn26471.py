# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26471: the contract declares only the criteria the live ticket carries.

The live ticket carries AC1 to AC6. ``contracts/OMN-26471.yaml`` also declared AC8, a criterion
for the bare-sequence ownership change that the ticket never carried, so the verifier answered
CRITERIA_DRIFT and skipped every item. The contract retires AC8 through typed
``supersedes_ac_binding`` entries and keeps its test running. A retired label that comes back as
a criterion, or a retirement naming an item that does not bind it, fails here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-26471.yaml"
KEPT = {"AC1", "AC2", "AC3", "AC4", "AC5", "AC6"}
RETIRED = {"AC8"}


def retired_labels(contract: dict[str, Any]) -> set[str]:
    return {
        str(entry["label"]).upper()
        for item in contract.get("dod_evidence", [])
        for entry in item.get("supersedes_ac_binding", [])
        if entry.get("reason_kind") == "no_longer_applicable"
    }


def declared_labels(contract: dict[str, Any]) -> set[str]:
    return {
        str(criterion["id"]).upper()
        for requirement in contract.get("requirements", [])
        for criterion in requirement.get("acceptance", [])
    }


def test_the_contract_declares_only_the_ticket_criteria_and_retires_ac8() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert declared_labels(contract) == KEPT
    assert retired_labels(contract) == RETIRED


def test_every_retirement_names_an_item_that_binds_the_label() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    items = {item["id"]: item for item in contract["dod_evidence"]}
    for item in contract["dod_evidence"]:
        for entry in item.get("supersedes_ac_binding", []):
            assert entry["item"] in items, entry["item"]
            assert entry["label"] in (items[entry["item"]].get("binds_ac") or [])


def test_every_kept_criterion_has_an_item() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    bound = {
        label for item in contract["dod_evidence"] for label in item.get("binds_ac", [])
    }
    assert bound >= KEPT


def test_control_a_contract_declaring_the_retired_label_is_caught() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    contract["requirements"][0]["acceptance"].append(
        {"id": "AC8", "statement": "AC8: back again -- falsifier: none"}
    )
    assert declared_labels(contract) & retired_labels(contract) == {"AC8"}
