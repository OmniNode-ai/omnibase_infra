# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26316: no acceptance criterion is declared under a label the contract retired.

The live ticket carries AC1 to AC4. ``contracts/OMN-26316.yaml`` had declared further
labels for other changes, so ``dod_verify`` refused with CRITERIA_DRIFT. The contract
retires each of them through typed ``supersedes_ac_binding`` entries; their checks keep
running under their own items. A retired label that returns as a criterion, or a new
criterion the ticket does not carry, fails here instead of at the drift check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-26316.yaml"
KEPT = {"AC1", "AC2", "AC3", "AC4"}


def load() -> dict[str, Any]:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def declared_labels(contract: dict[str, Any]) -> set[str]:
    return {
        str(criterion["id"]).upper()
        for requirement in contract.get("requirements", [])
        for criterion in requirement.get("acceptance", [])
    }


def retired_labels(contract: dict[str, Any]) -> set[str]:
    return {
        str(entry["label"]).upper()
        for item in contract.get("dod_evidence", [])
        for entry in item.get("supersedes_ac_binding", [])
        if entry.get("reason_kind") == "no_longer_applicable"
    }


def test_the_contract_declares_only_the_ticket_labels_and_retires_every_other_bound_label() -> (
    None
):
    contract = load()
    assert declared_labels(contract) == KEPT
    bound = {
        str(label).upper()
        for item in contract["dod_evidence"]
        for label in item.get("binds_ac") or []
    }
    assert bound - KEPT, "the contract no longer binds labels beyond the ticket's"
    assert retired_labels(contract) == bound - KEPT


def test_every_retirement_names_the_item_that_bound_the_label() -> None:
    contract = load()
    items = {item["id"]: item for item in contract["dod_evidence"]}
    for item in contract["dod_evidence"]:
        for entry in item.get("supersedes_ac_binding", []):
            assert entry["item"] != item["id"], (
                entry
            )  # a target cannot carry its own retirement
            assert entry["label"] in (items[entry["item"]].get("binds_ac") or []), entry


def test_control_a_contract_declaring_a_retired_label_is_caught() -> None:
    contract = load()
    contract["requirements"][0]["acceptance"].append(
        {"id": "AC5", "statement": "AC5: back again -- falsifier: none"}
    )
    assert declared_labels(contract) & retired_labels(contract) == {"AC5"}


def test_the_pin_item_binds_the_ticket_labels_so_the_gate_has_a_bound_check() -> None:
    contract = load()
    pin = next(
        item
        for item in contract["dod_evidence"]
        if item["id"] == "dod-omn26316-retired-labels-pinned"
    )
    assert {str(label).upper() for label in pin["binds_ac"]} == KEPT
