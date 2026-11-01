# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20032: pin the receipt-gate reusable that refuses a verifier below the floor."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.ci.receipt_gate_pin import RECEIPT_GATE_PIN_SHA

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLER_PATH = REPO_ROOT / ".github" / "workflows" / "call-repo-evidence-gate.yml"
REQUIRED_CHECKS_PATH = REPO_ROOT / ".github" / "required-checks.yaml"

# The pin carries omnibase_core#1932, which adds VERIFIER_FLOOR to the reusable's
# "Install the pinned verifier" step. The sha lives in tests/ci/receipt_gate_pin.py.
_FLOOR_REUSABLE_SHA = RECEIPT_GATE_PIN_SHA

# The reusable refuses a verifier-tag that does not descend from the floor commit
# (omnimarket#3511, the refusal of a contract that leaves an acceptance criterion
# unbound), so the caller passes a dev-merge tag and no version number (OMN-21853).
_VERIFIER_TAG_SHAPE = r"dev-(\d{14})-[0-9a-f]{12}"

# The first omnimarket dev-merge tag whose history holds omnimarket#3511 (commit
# e565ba5ba00c), omnimarket#3563 and omnimarket#3694, read from the omnimarket clone with
# `git tag --contains`. Dev-merge tags embed their UTC creation time, so a tag created
# before this one cannot descend from the floor commit. The reusable resolves the real
# ancestry at run time; this holds the caller to the same floor before a push.
_VERIFIER_TAG_FLOOR = "dev-20261016123308-0a385ce46cdf"


def _tag_time(tag: str) -> int | None:
    match = re.fullmatch(_VERIFIER_TAG_SHAPE, tag)
    return int(match.group(1)) if match else None


def _tag_is_at_or_above_floor(tag: str) -> bool:
    time = _tag_time(tag)
    floor = _tag_time(_VERIFIER_TAG_FLOOR)
    assert floor is not None
    return time is not None and time >= floor


def _job() -> dict[str, Any]:
    caller = yaml.safe_load(CALLER_PATH.read_text(encoding="utf-8"))
    job: dict[str, Any] = caller["jobs"]["repo-evidence"]
    return job


def test_caller_pins_the_reusable_that_carries_the_verifier_floor() -> None:
    uses = _job()["uses"]
    assert uses.endswith(f"@{_FLOOR_REUSABLE_SHA}"), (
        "pin the omnibase_core receipt-gate reusable at or past omnibase_core#1932"
    )


def test_required_checks_manifest_names_the_same_reusable_pin() -> None:
    manifest = yaml.safe_load(REQUIRED_CHECKS_PATH.read_text(encoding="utf-8"))
    refs = [
        gate["cross_repo_ref"]
        for gate in manifest["gates"]
        if gate["name"] == "repo-evidence / dod-verify"
    ]
    assert refs == [
        f"OmniNode-ai/omnibase_core/.github/workflows/receipt-gate.yml@{_FLOOR_REUSABLE_SHA}"
    ]


def test_caller_verifier_version_is_at_or_above_the_floor() -> None:
    """The caller's tag is no older than the first tag holding the floor commit (OMN-21853).

    The caller passes an omnimarket dev-merge tag and no verifier-version, the
    reusable's warned migration fallback; the reusable checks the real ancestry.
    """
    with_block = _job()["with"]
    assert "verifier-version" not in with_block, sorted(with_block)
    assert _tag_is_at_or_above_floor(with_block["verifier-tag"]), with_block


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("dev-20261101000200-949993edf3b7", True),
        (_VERIFIER_TAG_FLOOR, True),
        ("dev-20261016123307-0a385ce46cdf", False),
        ("dev-20261015000000-0a385ce46cdf", False),
        ("0.4.320", False),
        ("dev-20261101000200-949993edf3b", False),
        ("dev-20261101000200-949993EDF3B7", False),
        ("v20261101000200-949993edf3b7", False),
    ],
)
def test_floor_comparison_has_a_refusing_and_an_admitting_case(
    tag: str, expected: bool
) -> None:
    assert _tag_is_at_or_above_floor(tag) is expected
