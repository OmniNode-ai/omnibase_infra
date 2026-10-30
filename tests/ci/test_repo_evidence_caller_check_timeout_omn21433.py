# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21433: the repo-evidence caller gives each dod_verify check a named ceiling.

The verifier's default ceiling is 30 s per check. A hosted runner starts every
``uv run pytest`` cold, so a check that finishes in seconds on a lab host is killed and
recorded unverifiable there (omnibase_infra#5448: five pull-request heads, each
red on a different test-passing check). The reusable takes the ceiling as an input
since omnibase_core#2085; evidence-autoclose-sweep.yml already names 180 s.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CALLER_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "call-repo-evidence-gate.yml"
)

# omnibase_core#2102 descends from #2085, which added the check-timeout-seconds input.
# A pin absent here may not declare the input, and a caller passing an undeclared
# input is refused by GitHub before the job starts.
_REUSABLE_SHAS_DECLARING_THE_INPUT = {"1088a7b0303059c035ecc0e3ea33580f9556fe41"}

_VERIFIER_DEFAULT_SECONDS = 30
_RECEIPT_GATE_MAXIMUM_SECONDS = 900


def _job() -> dict[str, Any]:
    caller = yaml.safe_load(CALLER_PATH.read_text(encoding="utf-8"))
    job: dict[str, Any] = caller["jobs"]["repo-evidence"]
    return job


def test_caller_pins_a_reusable_that_declares_the_check_timeout_input() -> None:
    pinned_sha = _job()["uses"].rsplit("@", 1)[1]
    assert pinned_sha in _REUSABLE_SHAS_DECLARING_THE_INPUT


def test_caller_names_a_ceiling_above_the_verifier_default() -> None:
    value = _job()["with"].get("check-timeout-seconds")
    assert isinstance(value, str), "the reusable's inputs are strings; quote the number"
    assert re.fullmatch(r"[1-9][0-9]{0,2}", value), value
    assert _VERIFIER_DEFAULT_SECONDS < int(value) <= _RECEIPT_GATE_MAXIMUM_SECONDS


_PINNED_DEV_TAG = "dev-20261030004553-1088a7b03030"


def test_caller_sha_pin_is_the_commit_of_the_dev_tag_named_on_the_same_line() -> None:
    """A pull_request_target caller pins by 40-hex sha; the dev tag is named beside it.

    The dev-merge tag name ends in the first twelve hex digits of the commit it
    points at, so the pinned sha must carry that suffix and the line must name the tag.
    """
    line = next(
        raw
        for raw in CALLER_PATH.read_text(encoding="utf-8").splitlines()
        if "uses:" in raw and "receipt-gate.yml@" in raw
    )
    pinned_sha = _job()["uses"].rsplit("@", 1)[1]
    assert re.fullmatch(r"[0-9a-f]{40}", pinned_sha), pinned_sha
    assert _PINNED_DEV_TAG in line.split("#", 1)[1]
    assert pinned_sha.startswith(_PINNED_DEV_TAG.rsplit("-", 1)[1])
