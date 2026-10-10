# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20032: pin the receipt-gate reusable that refuses a verifier below the floor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLER_PATH = REPO_ROOT / ".github" / "workflows" / "call-repo-evidence-gate.yml"
REQUIRED_CHECKS_PATH = REPO_ROOT / ".github" / "required-checks.yaml"

# omnibase_core#1932: the squash commit that adds VERIFIER_FLOOR to the reusable's
# "Install the pinned verifier" step.
_FLOOR_REUSABLE_SHA = "4e4f5e0404d364e296c5d37db6aa3ab9e07f8ffa"

# 0.4.303 is the first omnimarket release carrying omnimarket#3511, the refusal of
# a contract that leaves an acceptance criterion unbound.
_VERIFIER_FLOOR = (0, 4, 303)

# The verifier floor each pinned receipt-gate reusable enforces; a pin absent here
# enforces none, so the caller's verifier-version is not held to a floor.
_REUSABLE_VERIFIER_FLOORS = {_FLOOR_REUSABLE_SHA: _VERIFIER_FLOOR}


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


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
    job = _job()
    pinned_sha = job["uses"].rsplit("@", 1)[1]
    floor = _REUSABLE_VERIFIER_FLOORS.get(pinned_sha)
    assert floor is not None, (
        f"the pinned receipt-gate reusable {pinned_sha} enforces no verifier floor"
    )
    assert _version(job["with"]["verifier-version"]) >= floor, (
        "verifier-version must be at or above the pinned reusable's VERIFIER_FLOOR"
    )


@pytest.mark.parametrize(
    ("version", "expected"),
    [("0.4.280", False), ("0.4.302", False), ("0.4.303", True), ("0.4.305", True)],
)
def test_floor_comparison_has_a_refusing_and_an_admitting_case(
    version: str, expected: bool
) -> None:
    assert (_version(version) >= _VERIFIER_FLOOR) is expected
