# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074: every ``binds_ac`` entry of the ticket contract is a criterion label.

omnibase_core's ``ModelTicketContract`` (0.47.39 and later) refuses a
``binds_ac`` entry that is not an ``AC<n>`` criterion label, and so did
onex_change_control's ``validate-yaml``. ``contracts/OMN-20074.yaml`` once bound
``AC-S8H1``; this pins every acceptance id to the label shape and every
``binds_ac`` entry to a declared acceptance id.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-20074.yaml"
CRITERION_LABEL = re.compile(r"^AC\d+$")


def _contract() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(contract, dict), f"{CONTRACT} is not a mapping"
    return contract


def _acceptance_ids(contract: dict[str, Any]) -> list[str]:
    return [
        criterion["id"]
        for requirement in contract.get("requirements", [])
        for criterion in requirement.get("acceptance", [])
    ]


def test_acceptance_ids_are_criterion_labels() -> None:
    ids = _acceptance_ids(_contract())
    assert ids, f"{CONTRACT} declares no acceptance criteria"
    bad = [ac_id for ac_id in ids if not CRITERION_LABEL.match(ac_id)]
    assert not bad, f"{CONTRACT}: acceptance ids that are not AC<n> labels: {bad}"


def test_binds_ac_names_declared_criterion_labels() -> None:
    contract = _contract()
    declared = set(_acceptance_ids(contract))
    items = contract.get("dod_evidence", [])
    assert items, f"{CONTRACT} has no dod_evidence"
    for item in items:
        binds = item.get("binds_ac", [])
        assert binds, f"{CONTRACT}: {item['id']} binds no criterion"
        for label in binds:
            assert CRITERION_LABEL.match(label), (
                f"{CONTRACT}: {item['id']} binds {label!r}, not an AC<n> label"
            )
            assert label in declared, (
                f"{CONTRACT}: {item['id']} binds {label!r}, not a declared criterion"
            )
