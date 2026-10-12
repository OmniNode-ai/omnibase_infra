# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The omnimarket contract pin moved off e286bd29, and every reader moved with it."""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_PREVIOUS_PIN = "e286bd296e373e8e17e11c735393613a227f7972"


def _pin() -> str:
    data = yaml.safe_load(
        (_ROOT / ".github/omnimarket-contract-pin.yaml").read_text(encoding="utf-8")
    )
    return str(data["omnimarket_contract_ref"])


def test_pin_advanced_past_previous_ref() -> None:
    pin = _pin()
    assert re.fullmatch(r"[0-9a-f]{40}", pin)
    assert pin != _PREVIOUS_PIN


def test_sibling_pins_carry_the_contract_pin() -> None:
    sibling = yaml.safe_load(
        (_ROOT / ".github/sibling-pins.yaml").read_text(encoding="utf-8")
    )
    assert _PREVIOUS_PIN not in str(sibling)
    assert _pin() in str(sibling)


def test_no_workflow_reads_the_previous_ref() -> None:
    stale = [
        path.name
        for path in sorted((_ROOT / ".github/workflows").glob("*.yml"))
        if _PREVIOUS_PIN in path.read_text(encoding="utf-8")
    ]
    assert stale == []


@pytest.mark.live_contact("tests/ci/fixtures/omnimarket_pin_commit_omn17292.json")
def test_pin_is_the_recorded_omnimarket_commit(
    recorded_response: dict[str, object],
) -> None:
    assert recorded_response["object_type"] == "commit"
    content = base64.b64decode(str(recorded_response["content"]))
    object_id = hashlib.sha1(
        b"commit %d\0" % len(content) + content, usedforsecurity=False
    ).hexdigest()
    assert object_id == _pin()
