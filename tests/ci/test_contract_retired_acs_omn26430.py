# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-26430 contract carries only the criteria the ticket still has."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts/OMN-26430.yaml"
RETIRED = {"AC3", "AC4"}
RETIRING = "dod-omn26430-ac3-ac4-retired"


def test_no_requirement_declares_a_retired_criterion() -> None:
    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    declared = {
        ac["id"] for req in data["requirements"] for ac in req.get("acceptance", [])
    }
    assert declared == {"AC1", "AC2"}


def test_only_the_retiring_item_cites_a_retired_criterion() -> None:
    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    retired_items = {
        (s["item"], s["label"])
        for item in data["dod_evidence"]
        if item["id"] == RETIRING
        for s in item["supersedes_ac_binding"]
    }
    assert retired_items == {
        ("dod-omn26430-ac3-serial-dispatch", "AC3"),
        ("dod-omn26430-ac4-injected-db-parallel", "AC4"),
    }
    for item in data["dod_evidence"]:
        if set(item.get("binds_ac", [])) & RETIRED:
            assert item["id"] in {i for i, _ in retired_items}
