# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The omnimarket contract pin and its inline readers carry the advanced sha.

The pin refresh producer writes the expected sha to a fixture under
tests/ci/fixtures, a test-side path the Repo Evidence Gate overlays at the
merge base. The pins there still carry the old sha, so these tests fail.
The rendered contracts/OMN-17292.yaml title and every place the repo pins
omnimarket at a literal sha must carry the fixture's sha.
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
    fixture = yaml.safe_load(
        (
            REPO_ROOT
            / "tests/ci/fixtures/omnimarket_contract_pin_advance_omn17292.yaml"
        ).read_text(encoding="utf-8")
    )
    target = fixture["omnimarket_contract_ref"]
    assert isinstance(target, str) and SHA.fullmatch(target), target
    return target


def test_contract_names_the_advanced_sha() -> None:
    contract = yaml.safe_load(
        (REPO_ROOT / "contracts" / "OMN-17292.yaml").read_text(encoding="utf-8")
    )
    found = SHA.findall(contract["title"])
    assert len(found) == 1, contract["title"]
    assert found[0] == _target_sha()


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
