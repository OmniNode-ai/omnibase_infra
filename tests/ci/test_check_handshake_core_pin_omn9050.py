# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The handshake workflow's omnibase_core pin moved to 2ca615af292a (OMN-9050).

The automated pin bump moves the omnibase_core checkout ref in
check-handshake.yml. The ref must be a full commit sha, the auto-bump comment
above it must name the same commit, and it must be the bumped 2ca615af292a pin, so a revert of the bump fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "check-handshake.yml"
SUPERSEDED_REF = "16ee7c558552"
BUMPED_REF = "2ca615af292a52898359d23538299f3bc0880f07"


def _core_checkout_ref() -> str:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    refs = [
        step["with"]["ref"]
        for job in doc["jobs"].values()
        for step in job.get("steps", [])
        if step.get("with", {}).get("repository") == "OmniNode-ai/omnibase_core"
    ]
    assert len(refs) == 1, refs
    return str(refs[0])


def test_core_pin_is_a_full_sha() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", _core_checkout_ref())


def test_core_pin_matches_auto_bump_comment() -> None:
    match = re.search(
        r"Auto-bumped by omnibase_core publish-downstream-pin-bump\.yml to ([0-9a-f]{12})\.",
        WORKFLOW.read_text(encoding="utf-8"),
    )
    assert match is not None
    assert _core_checkout_ref().startswith(match.group(1))


def test_core_pin_moved_off_superseded_ref() -> None:
    assert not _core_checkout_ref().startswith(SUPERSEDED_REF)


def test_core_pin_is_bumped_ref() -> None:
    assert _core_checkout_ref() == BUMPED_REF
