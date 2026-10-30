# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20134: the receipt-gate caller pins the omnibase_core dev-merge tag that carries the fix-parent fallback."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tests.ci.receipt_gate_pin import (
    RECEIPT_GATE_PIN_SHA,
    RECEIPT_GATE_PIN_TAG,
    RECEIPT_GATE_REUSABLE,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLER = REPO_ROOT / ".github" / "workflows" / "call-repo-evidence-gate.yml"
REQUIRED_CHECKS = REPO_ROOT / ".github" / "required-checks.yaml"

_TAG_SHAPE = r"dev-\d{14}-[0-9a-f]{12}"


def _caller_line() -> str:
    return next(
        raw
        for raw in CALLER.read_text(encoding="utf-8").splitlines()
        if "uses:" in raw and "receipt-gate.yml@" in raw
    )


def test_caller_pins_the_reusable_at_the_pin_sha_and_names_the_pin_tag() -> None:
    doc = yaml.safe_load(CALLER.read_text(encoding="utf-8"))
    assert doc["jobs"]["repo-evidence"]["uses"] == (
        f"{RECEIPT_GATE_REUSABLE}@{RECEIPT_GATE_PIN_SHA}"
    )
    assert re.findall(_TAG_SHAPE, _caller_line().split("#", 1)[1]) == [
        RECEIPT_GATE_PIN_TAG
    ]


def test_pin_constants_have_the_shape_of_a_dev_merge_tag_and_its_commit() -> None:
    assert re.fullmatch(_TAG_SHAPE, RECEIPT_GATE_PIN_TAG)
    assert re.fullmatch(r"[0-9a-f]{40}", RECEIPT_GATE_PIN_SHA)
    assert RECEIPT_GATE_PIN_SHA.startswith(RECEIPT_GATE_PIN_TAG.rsplit("-", 1)[1])


def test_required_checks_cross_repo_ref_pins_the_same_commit_and_tag() -> None:
    text = REQUIRED_CHECKS.read_text(encoding="utf-8")
    found = re.findall(
        rf'cross_repo_ref: "{re.escape(RECEIPT_GATE_REUSABLE)}@([0-9a-f]{{40}})" # ({_TAG_SHAPE})',
        text,
    )
    assert found == [(RECEIPT_GATE_PIN_SHA, RECEIPT_GATE_PIN_TAG)]
