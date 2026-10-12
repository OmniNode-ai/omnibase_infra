# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: the omnimarket pin advance names the recorded omnimarket commit.

The recording is omnimarket's first-parent history from the previous pin to
the new one. The advance must be forward-only along that history, and the pin
file, the omnimarket entry of ``.github/sibling-pins.yaml`` and every inline
workflow ``ref:`` that reads it (OMN-20331) must name the recorded commit.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_PIN_FILE = _ROOT / ".github" / "omnimarket-contract-pin.yaml"
_SIBLING_PINS = _ROOT / ".github" / "sibling-pins.yaml"
_WORKFLOWS = _ROOT / ".github" / "workflows"
_SIBLING_REF_SUFFIX = "# .github/sibling-pins.yaml"


@pytest.mark.live_contact("tests/ci/fixtures/omnimarket_pin_advance_omn17292.json")
def test_pin_advance_follows_recorded_omnimarket_history(
    recorded_response: dict[str, object],
) -> None:
    pinned = recorded_response["pinned"]
    previous = recorded_response["previous_pin"]
    chain = recorded_response["first_parent_chain"]
    assert isinstance(chain, list) and chain

    # Forward-only: walk first parents from the new pin back to the old one.
    assert chain[0]["sha"] == pinned
    for child, parent in pairwise(chain):
        assert child["parents"][0] == parent["sha"]
    assert chain[-1]["parents"][0] == previous

    pin = yaml.safe_load(_PIN_FILE.read_text(encoding="utf-8"))
    assert pin["repository"] == recorded_response["repository"]
    assert pin["omnimarket_contract_ref"] == pinned

    sibling_pins = yaml.safe_load(_SIBLING_PINS.read_text(encoding="utf-8"))
    assert sibling_pins["pins"]["omnimarket"] == pinned

    stale = [
        f"{path.name}:{number}"
        for path in sorted(_WORKFLOWS.glob("*.yml"))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.strip().endswith(_SIBLING_REF_SUFFIX) and previous in line
    ]
    assert stale == []
