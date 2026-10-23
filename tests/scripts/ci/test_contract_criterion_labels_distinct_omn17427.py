# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: every committed contract's criterion labels bind a receipt.

``lab_pass_receipt.py emit --bind-from-commit`` copies the cited ticket
contract's acceptance ids into the receipt's ``criterion_labels``, and the
receipt refuses labels that repeat. ``contracts/OMN-17427.yaml`` declared AC49,
AC50, AC51, AC53 and AC97 twice each, so the candidate-boot receipt of every
dev commit titled OMN-17427 was refused ("criterion_labels must be distinct
ticket criterion labels", deliver-dev-candidate-to-staging.yml run 39389364212
on e302a800a). This builds a bound receipt from every committed contract.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.scripts.ci._lab_pass_fixtures import SHA_4ACA, receipt

pytestmark = pytest.mark.unit

CONTRACTS = Path(__file__).resolve().parents[3] / "contracts"
TICKET = re.compile(r"OMN-\d+")


def _criterion_labels(contract: dict[str, Any]) -> tuple[str, ...]:
    """The acceptance ids ``bind_commit_checks`` copies into a bound receipt."""
    return tuple(
        acceptance["id"]
        for requirement in contract.get("requirements", [])
        for acceptance in requirement.get("acceptance", [])
    )


def _ticket_contracts() -> list[tuple[str, dict[str, Any]]]:
    found = []
    for path in sorted(CONTRACTS.glob("*.yaml")):
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(contract, dict) and TICKET.fullmatch(
            str(contract.get("ticket_id", ""))
        ):
            found.append((path.name, contract))
    return found


def test_every_contract_yields_a_bindable_receipt() -> None:
    contracts = _ticket_contracts()
    assert contracts, f"no ticket contract under {CONTRACTS}"
    refused = []
    for name, contract in contracts:
        labels = _criterion_labels(contract)
        try:
            replace(
                receipt(SHA_4ACA),
                ticket_id=str(contract["ticket_id"]),
                criterion_labels=labels,
            )
        except ValueError as exc:
            repeated = sorted({label for label in labels if labels.count(label) > 1})
            refused.append((name, str(exc), repeated))
    assert not refused, f"contracts whose labels a receipt refuses: {refused}"


def test_a_repeated_label_is_refused() -> None:
    contract: dict[str, Any] = {
        "ticket_id": "OMN-17427",
        "requirements": [
            {"acceptance": [{"id": "AC1"}]},
            {"acceptance": [{"id": "AC1"}]},
        ],
    }
    labels = _criterion_labels(contract)
    assert labels == ("AC1", "AC1")
    with pytest.raises(ValueError, match="criterion_labels must be distinct"):
        replace(receipt(SHA_4ACA), ticket_id="OMN-17427", criterion_labels=labels)
