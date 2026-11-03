# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19812 contract carries only the criteria the ticket still has."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts/OMN-19812.yaml"
RETIRED = {"AC5", "AC6", "AC7"}


def test_no_requirement_declares_a_retired_criterion() -> None:
    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    declared = {
        ac["id"] for req in data["requirements"] for ac in req.get("acceptance", [])
    }
    assert declared
    assert not declared & RETIRED


def test_only_the_retiring_item_cites_a_retired_criterion() -> None:
    data = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    retiring = "dod-omn19812-ac5-ac7-retired"
    retired_items = {
        s["item"]
        for item in data["dod_evidence"]
        if item["id"] == retiring
        for s in item["supersedes_ac_binding"]
    }
    for item in data["dod_evidence"]:
        if set(item.get("binds_ac", [])) & RETIRED:
            assert item["id"] in retired_items
