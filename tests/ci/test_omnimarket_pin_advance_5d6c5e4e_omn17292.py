# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: the omnimarket pin advance to 5d6c5e4e moves every reader together.

The refresh moves ``.github/omnimarket-contract-pin.yaml``, the omnimarket entry
of ``.github/sibling-pins.yaml`` and every inline workflow ``ref:`` that checks
omnimarket out. A pin that moved alone would have the grants gate and the
sibling checkouts read two different contract sets.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_ADVANCED_REF = "5d6c5e4e4cb3a0563fde9e24fdc864617e0917bd"
_OMNIMARKET_CHECKOUT = re.compile(
    r"repository:\s*OmniNode-ai/omnimarket\s*\n\s*ref:\s*([0-9a-f]{40})"
)


def test_contract_pin_carries_the_advanced_ref() -> None:
    pin = yaml.safe_load((_REPO / ".github/omnimarket-contract-pin.yaml").read_text())
    assert pin["omnimarket_contract_ref"] == _ADVANCED_REF


def test_sibling_pin_carries_the_advanced_ref() -> None:
    siblings = (_REPO / ".github/sibling-pins.yaml").read_text()
    assert re.search(rf"^\s*omnimarket:\s*{_ADVANCED_REF}\s*$", siblings, re.M)


def test_every_inline_omnimarket_checkout_reads_the_advanced_ref() -> None:
    refs = {
        f"{path.name}:{ref}"
        for path in sorted((_REPO / ".github/workflows").glob("*.yml"))
        for ref in _OMNIMARKET_CHECKOUT.findall(path.read_text())
    }
    assert refs
    assert {r for r in refs if not r.endswith(_ADVANCED_REF)} == set()
