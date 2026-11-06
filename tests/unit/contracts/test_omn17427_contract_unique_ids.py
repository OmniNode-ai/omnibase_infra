# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Each entry of the OMN-17427 contract has its own id, and the renamed one keeps its criterion."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[3] / "contracts" / "OMN-17427.yaml"


@pytest.mark.parametrize("section", ["requirements", "dod_evidence"])
def test_every_entry_id_is_unique(section: str) -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    counts = Counter(entry["id"] for entry in contract[section])

    assert [entry_id for entry_id, count in counts.items() if count > 1] == []


def test_both_transport_requirements_keep_their_criteria() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    labels = {
        criterion["id"]
        for requirement in contract["requirements"]
        if requirement["id"].startswith("req-omn17427-prepr-pool-")
        for criterion in requirement["acceptance"]
    }

    assert {"AC109", "AC151"} <= labels
