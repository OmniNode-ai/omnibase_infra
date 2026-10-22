# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: ticket contracts moved from onex_change_control into this repository.

The scheduled closer's contract resolver reads ``contracts/<id>.yaml`` from a
product repository's default branch. These 80 contracts were held only by the
change-control repository (onex_change_control); each now lives here in
yamlfmt form.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"

MOVED_IDS = (
    "OMN-14060",
    "OMN-14115",
    "OMN-14953",
    "OMN-14993",
    "OMN-15092",
    "OMN-15310",
    "OMN-15330",
    "OMN-15358",
    "OMN-15384",
    "OMN-15484",
    "OMN-15642",
    "OMN-15741",
    "OMN-15807",
    "OMN-15837",
    "OMN-15838",
    "OMN-16056",
    "OMN-16107",
    "OMN-16171",
    "OMN-16293",
    "OMN-16322",
    "OMN-16343",
    "OMN-16367",
    "OMN-16418",
    "OMN-16420",
    "OMN-16661",
    "OMN-16769",
    "OMN-16788",
    "OMN-16821",
    "OMN-16926",
    "OMN-16947",
    "OMN-16961",
    "OMN-17079",
    "OMN-17106",
    "OMN-17135",
    "OMN-17137",
    "OMN-17152",
    "OMN-17166",
    "OMN-17221",
    "OMN-17277",
    "OMN-17296",
    "OMN-17317",
    "OMN-17379",
    "OMN-17395",
    "OMN-17398",
    "OMN-17432",
    "OMN-17443",
    "OMN-17447",
    "OMN-17485",
    "OMN-17648",
    "OMN-17664",
    "OMN-17678",
    "OMN-17792",
    "OMN-17855",
    "OMN-17862",
    "OMN-17886",
    "OMN-17888",
    "OMN-17893",
    "OMN-17949",
    "OMN-17981",
    "OMN-18124",
    "OMN-18134",
    "OMN-18202",
    "OMN-18266",
    "OMN-18272",
    "OMN-18296",
    "OMN-18353",
    "OMN-18411",
    "OMN-18632",
    "OMN-18896",
    "OMN-19315",
    "OMN-19532",
    "OMN-19741",
    "OMN-20320",
    "OMN-20339",
    "OMN-20359",
    "OMN-20439",
    "OMN-20568",
    "OMN-20600",
    "OMN-20602",
    "OMN-8468",
)


def test_moved_id_count() -> None:
    assert len(MOVED_IDS) == 80
    assert len(set(MOVED_IDS)) == len(MOVED_IDS)


@pytest.mark.parametrize("ticket_id", MOVED_IDS)
def test_moved_contract_exists_and_names_its_ticket(ticket_id: str) -> None:
    path = CONTRACTS / f"{ticket_id}.yaml"
    assert path.is_file(), f"{path} is missing"
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(contract, dict), f"{path} is not a mapping"
    assert contract.get("ticket_id") == ticket_id
