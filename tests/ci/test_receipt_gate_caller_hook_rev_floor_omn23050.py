# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23050: the repo-evidence caller pins a receipt-gate that counts a pin-train hook rev swap as pin-only.

The sibling-pin train moves the omnibase_core hook revs of ``.pre-commit-config.yaml``
together with the source pin. The receipt-gate reusable only treats that as a
pin-only change from the omnibase_core dev-merge tag that carries the hook rev
swap exemption (omnibase_core#2284). ``pull_request_target`` reads the caller from
the base branch, so until the caller on ``dev`` pins that tag or a later one, every
train pull request falls to the evidence steps and fails repo-evidence / dod-verify.

The floor is read from the caller workflow, from required-checks.yaml and from the
test-side pin constants, so the three copies cannot drift below it separately.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.ci.receipt_gate_pin import RECEIPT_GATE_PIN_SHA, RECEIPT_GATE_PIN_TAG

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLER = REPO_ROOT / ".github" / "workflows" / "call-repo-evidence-gate.yml"
REQUIRED_CHECKS = REPO_ROOT / ".github" / "required-checks.yaml"

# First omnibase_core dev-merge tag that holds the hook rev swap exemption (#2284).
HOOK_REV_EXEMPTION_FLOOR_TAG = "dev-20261108023315-fe0d8082717e"
# The pin this repository held before the exemption existed.
PREVIOUS_PIN_TAG = "dev-20261101005012-ac2f26e3e5bf"

_TAG = re.compile(r"dev-(\d{14})-[0-9a-f]{12}")


def _stamp(tag: str) -> int:
    match = _TAG.fullmatch(tag)
    assert match is not None, f"not a dev-merge tag: {tag}"
    return int(match.group(1))


def _caller_pin_sha(text: str) -> str:
    found = re.findall(r"receipt-gate\.yml@([0-9a-f]{40})", text)
    assert len(found) == 1, found
    return found[0]


def test_pin_constants_are_at_or_after_the_hook_rev_exemption_tag() -> None:
    assert _stamp(RECEIPT_GATE_PIN_TAG) >= _stamp(HOOK_REV_EXEMPTION_FLOOR_TAG)


def test_caller_and_required_checks_pin_the_floor_commit() -> None:
    sha = RECEIPT_GATE_PIN_SHA
    assert HOOK_REV_EXEMPTION_FLOOR_TAG.rsplit("-", 1)[1] == sha[:12]
    assert _caller_pin_sha(CALLER.read_text(encoding="utf-8")) == sha
    assert _caller_pin_sha(REQUIRED_CHECKS.read_text(encoding="utf-8")) == sha


def test_floor_control_rejects_the_previous_pin() -> None:
    # Positive control: the pin this repository held before the exemption existed
    # sits below the floor, so the checks above can fail.
    assert _stamp(PREVIOUS_PIN_TAG) < _stamp(HOOK_REV_EXEMPTION_FLOOR_TAG)
