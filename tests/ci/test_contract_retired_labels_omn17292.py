# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: no acceptance criterion is declared under a label the contract retired.

``contracts/OMN-17292.yaml`` declared AC5 and AC6 that the live ticket never
carried, so ``node_dod_criteria_check_effect`` held every pull request citing the
ticket with CRITERIA_DRIFT. The contract retires both labels through typed
``supersedes_ac_binding`` entries and keeps their tests running under one carrier
item. This pins that state: a retired label that comes back as an acceptance
criterion, which is what a refresh that re-renders the whole contract does, fails
here instead of at the drift check on the next pull request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-17292.yaml"
RETIRED = {"AC5", "AC6"}
KEPT = {"AC1", "AC2", "AC3", "AC4"}


def retired_labels(contract: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for item in contract.get("dod_evidence", []):
        for entry in item.get("supersedes_ac_binding", []):
            if entry.get("reason_kind") == "no_longer_applicable":
                labels.add(str(entry["label"]).upper())
    return labels


def declared_labels(contract: dict[str, Any]) -> set[str]:
    return {
        str(criterion["id"]).upper()
        for requirement in contract.get("requirements", [])
        for criterion in requirement.get("acceptance", [])
    }


def test_the_contract_retires_ac5_and_ac6_and_still_declares_ac1_to_ac4() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert retired_labels(contract) >= RETIRED
    declared = declared_labels(contract)
    assert declared >= KEPT
    assert not declared & retired_labels(contract)


def test_control_a_contract_declaring_a_retired_label_is_caught() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    contract["requirements"][0]["acceptance"].append(
        {"id": "AC5", "statement": "AC5: back again -- falsifier: none"}
    )
    assert declared_labels(contract) & retired_labels(contract) == {"AC5"}
