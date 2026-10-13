# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The handshake workflow's omnibase_core pin follows its auto-bump (OMN-9050).

The automated pin bump moves the omnibase_core checkout ref in
check-handshake.yml. The ref must be a full commit sha, the auto-bump comment
above it must name the same commit, and the superseded pin remains forbidden.
Later automated bumps must not be compared against a frozen historical SHA.
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
    assert _core_checkout_ref() == _auto_bumped_core_ref(
        WORKFLOW.read_text(encoding="utf-8")
    )


def _auto_bumped_core_ref(workflow_text: str) -> str:
    matches: list[tuple[str, str]] = re.findall(
        r"^[ \t]+repository: OmniNode-ai/omnibase_core\n"
        r"[ \t]+# Auto-bumped by omnibase_core "
        r"publish-downstream-pin-bump\.yml to ([0-9a-f]{12})\.\n"
        r"[ \t]+ref: ([0-9a-f]{40})[ \t]*$",
        workflow_text,
        re.MULTILINE,
    )
    assert len(matches) == 1, matches
    short_ref, ref = matches[0]
    assert ref.startswith(short_ref)
    assert not ref.startswith(SUPERSEDED_REF)
    return ref


@pytest.mark.parametrize(
    "ref",
    [
        "2ca615af292a52898359d23538299f3bc0880f07",
        "bac865f4f32a3f8ca80cabcfac06c8400d8c80d7",
    ],
)
def test_auto_bumped_core_ref_accepts_successive_bumps(ref: str) -> None:
    workflow_text = (
        "          repository: OmniNode-ai/omnibase_core\n"
        "          # Auto-bumped by omnibase_core "
        f"publish-downstream-pin-bump.yml to {ref[:12]}.\n"
        f"          ref: {ref}\n"
    )
    assert _auto_bumped_core_ref(workflow_text) == ref


@pytest.mark.parametrize(
    ("short_ref", "ref"),
    [
        ("bac865f4f32a", "bac865f4f32a"),
        ("bac865f4f32a", "dev"),
        ("2ca615af292a", "bac865f4f32a3f8ca80cabcfac06c8400d8c80d7"),
        (SUPERSEDED_REF, SUPERSEDED_REF + "0" * 28),
    ],
)
def test_auto_bumped_core_ref_rejects_invalid_pins(short_ref: str, ref: str) -> None:
    workflow_text = (
        "          repository: OmniNode-ai/omnibase_core\n"
        "          # Auto-bumped by omnibase_core "
        f"publish-downstream-pin-bump.yml to {short_ref}.\n"
        f"          ref: {ref}\n"
    )
    with pytest.raises(AssertionError):
        _auto_bumped_core_ref(workflow_text)


@pytest.mark.parametrize("copies", [0, 2])
def test_auto_bumped_core_ref_requires_one_checkout(copies: int) -> None:
    checkout = (
        "          repository: OmniNode-ai/omnibase_core\n"
        "          # Auto-bumped by omnibase_core "
        "publish-downstream-pin-bump.yml to bac865f4f32a.\n"
        "          ref: bac865f4f32a3f8ca80cabcfac06c8400d8c80d7\n"
    )
    with pytest.raises(AssertionError):
        _auto_bumped_core_ref(checkout * copies)
