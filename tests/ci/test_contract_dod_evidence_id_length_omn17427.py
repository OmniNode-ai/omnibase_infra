# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: every contract's ``dod_evidence`` id fits the core limit.

omnibase_core's ``ModelContractDodItem`` caps each id with ``MaxLen`` metadata.
The contract-validation job loads the branch ticket's contract through
``ModelTicketContract``. ``contracts/OMN-19812.yaml`` once carried a
59-character id that failed every OMN-19812 pull request; this pins every
committed contract to the core limit and checks that contract loads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from annotated_types import MaxLen

from omnibase_core.models.ticket.model_contract_dod_item import ModelContractDodItem
from omnibase_core.models.ticket.model_ticket_contract import ModelTicketContract

pytestmark = pytest.mark.unit

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


def _id_limit() -> int:
    limits = [
        entry.max_length
        for entry in ModelContractDodItem.model_fields["id"].metadata
        if isinstance(entry, MaxLen)
    ]
    assert limits, "ModelContractDodItem.id has no MaxLen metadata"
    return limits[0]


def test_every_contract_dod_evidence_id_fits_core_limit() -> None:
    limit = _id_limit()
    paths = sorted(CONTRACTS.glob("*.yaml"))
    assert paths, f"no contract under {CONTRACTS}"
    offenders: list[tuple[str, str, int]] = []
    for path in paths:
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            continue
        items = contract.get("dod_evidence")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if isinstance(item_id, str) and len(item_id) > limit:
                offenders.append((path.name, item_id, len(item_id)))
    assert not offenders, (
        f"dod_evidence ids longer than core limit {limit}: {offenders}"
    )


def test_omn19812_contract_loads_with_model_ticket_contract() -> None:
    text = (CONTRACTS / "OMN-19812.yaml").read_text(encoding="utf-8")
    ModelTicketContract.from_yaml(text)
