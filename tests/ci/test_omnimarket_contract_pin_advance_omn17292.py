# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The omnimarket contract pin and its inline readers carry the advanced sha.

The pin refresh workflow renders contracts/OMN-17292.yaml naming the sha it
advances to. Every place the repo pins omnimarket at a literal sha must carry
that sha, so a refresh that moves only part of the set fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SHA = re.compile(r"\b[0-9a-f]{40}\b")
INLINE_OMNIMARKET_REF = re.compile(
    r"repository:\s*OmniNode-ai/omnimarket\s*\n\s*ref:\s*([0-9a-f]{40})\b"
)


def _target_sha() -> str:
    contract = yaml.safe_load(
        (REPO_ROOT / "contracts" / "OMN-17292.yaml").read_text(encoding="utf-8")
    )
    found = SHA.findall(contract["title"])
    assert len(found) == 1, contract["title"]
    return found[0]


def test_pin_file_carries_the_advanced_sha() -> None:
    pin = yaml.safe_load(
        (REPO_ROOT / ".github" / "omnimarket-contract-pin.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert pin["omnimarket_contract_ref"] == _target_sha()


def test_sibling_pins_carry_the_advanced_sha() -> None:
    pins = yaml.safe_load(
        (REPO_ROOT / ".github" / "sibling-pins.yaml").read_text(encoding="utf-8")
    )
    assert pins["pins"]["omnimarket"] == _target_sha()


def test_every_inline_omnimarket_ref_carries_the_advanced_sha() -> None:
    target = _target_sha()
    refs = {
        f"{path.name}: {ref}"
        for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
        for ref in INLINE_OMNIMARKET_REF.findall(path.read_text(encoding="utf-8"))
    }
    assert refs
    assert {r for r in refs if not r.endswith(target)} == set()
