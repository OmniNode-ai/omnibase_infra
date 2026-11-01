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

# The first omnimarket dev-merge tag whose history holds each verifier capability this
# caller depends on, read from the omnimarket clone with `git tag --contains <merge
# commit>`. Dev-merge tags embed their UTC creation time, so a tag created before a
# capability's first tag cannot descend from its merge commit. The reusable at the
# pinned sha resolves the real ancestry from the floor commit at run time (it refuses
# a moved tag, a tag off omnimarket dev, or one that does not descend from
# e565ba5ba00c); this holds the caller to every capability's first tag before a push,
# which a regex on the tag shape alone does not.
_CAPABILITY_FIRST_TAG = {
    # omnimarket#3511 (e565ba5ba00c): refuses a contract leaving a criterion unbound.
    "unbound-criterion refusal": "dev-20261016123308-0a385ce46cdf",
    # omnimarket#3563: the occ-difference classifier.
    "occ-difference classifier": "dev-20261016123308-0a385ce46cdf",
    # omnimarket#3694: admits tool.uv.override-dependencies as a dependency-pin-only change.
    "override-dependency pin exemption": "dev-20261016123308-0a385ce46cdf",
    # omnimarket#4683 (OMN-21676): the hosted dod-verify delegates the criteria
    # comparison to the trusted pre-merge check. A verifier without it reads live
    # Linear from a runner that holds no credential and ends every contract
    # VERIFICATION_UNRESOLVED ticket_unavailable (omnimarket 0.4.320 and older).
    "hosted criteria delegation": "dev-20261031203936-91354974f1e3",
}
_VERIFIER_TAG_FLOOR = max(_CAPABILITY_FIRST_TAG.values())


def _tag_time(tag: str) -> int | None:
    match = re.fullmatch(_VERIFIER_TAG_SHAPE, tag)
    return int(match.group(1)) if match else None


def _tag_is_at_or_above_floor(tag: str, floor_tag: str = _VERIFIER_TAG_FLOOR) -> bool:
    time = _tag_time(tag)
    floor = _tag_time(floor_tag)
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


def test_caller_tag_is_no_older_than_every_capability_first_tag() -> None:
    """The caller's tag is a well-formed dev-merge tag created at or after the first
    tag of every capability the caller depends on (OMN-21853), and no verifier-version
    (the reusable's warned migration fallback) is passed.
    """
    with_block = _job()["with"]
    assert "verifier-version" not in with_block, sorted(with_block)
    tag = with_block["verifier-tag"]
    for capability, first_tag in _CAPABILITY_FIRST_TAG.items():
        assert _tag_is_at_or_above_floor(tag, first_tag), (capability, tag, first_tag)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("dev-20261101000200-949993edf3b7", True),
        (_VERIFIER_TAG_FLOOR, True),
        # Created one second before the criteria-delegation tag: lacks omnimarket#4683.
        ("dev-20261031203935-91354974f1e3", False),
        ("dev-20261016123308-0a385ce46cdf", False),
        ("dev-20261015000000-0a385ce46cdf", False),
        ("0.4.320", False),
        ("dev-20261101000200-949993edf3b", False),
        ("dev-20261101000200-949993EDF3B7", False),
        ("v20261101000200-949993edf3b7", False),
    ],
)
def test_tag_floor_comparison_refuses_a_tag_older_than_the_newest_capability(
    tag: str, expected: bool
) -> None:
    assert _tag_is_at_or_above_floor(tag) is expected
