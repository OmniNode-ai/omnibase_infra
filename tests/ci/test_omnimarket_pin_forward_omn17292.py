# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: the omnimarket contract pin has moved past every retired ref.

The pin is forward-only (scripts/ci/check_omnimarket_contract_pin_advance.py).
A ref the pin has already advanced past must never be the pin again, and the
omnimarket sibling pin must follow the contract pin.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_PIN = _REPO_ROOT / ".github" / "omnimarket-contract-pin.yaml"
_SIBLING_PINS = _REPO_ROOT / ".github" / "sibling-pins.yaml"

# Refs the pin has advanced past; a later advance appends the ref it leaves.
_RETIRED_REFS = frozenset(
    {
        "f859c7af5e1d2a4df043bbb5d5e9d873a6c1c495",
        "e286bd296e373e8e17e11c735393613a227f7972",
    }
)


def _contract_ref() -> str:
    pin = yaml.safe_load(_CONTRACT_PIN.read_text())
    ref = pin["omnimarket_contract_ref"]
    assert isinstance(ref, str)
    return ref


def test_contract_pin_has_advanced_past_retired_refs() -> None:
    ref = _contract_ref()
    assert re.fullmatch(r"[0-9a-f]{40}", ref), ref
    assert ref not in _RETIRED_REFS


def test_sibling_pin_follows_contract_pin() -> None:
    sibling = yaml.safe_load(_SIBLING_PINS.read_text())
    assert sibling["pins"]["omnimarket"] == _contract_ref()
