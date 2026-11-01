# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: repo evidence verifier admits the omnibase-core override pin."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLER_PATH = REPO_ROOT / ".github" / "workflows" / "call-repo-evidence-gate.yml"

_VERIFIER_TAG_SHAPE = r"dev-\d{14}-[0-9a-f]{12}"

# The first omnimarket dev-merge tag holding omnimarket#3694 (`git tag --contains`).
# Tags embed their UTC creation time, so an older tag cannot admit override pins.
_OVERRIDE_PIN_FIRST_TAG = "dev-20261016123308-0a385ce46cdf"


def _admits_override_pins(tag: str) -> bool:
    match = re.fullmatch(r"dev-(\d{14})-[0-9a-f]{12}", tag)
    floor = re.fullmatch(r"dev-(\d{14})-[0-9a-f]{12}", _OVERRIDE_PIN_FIRST_TAG)
    assert floor is not None
    return match is not None and int(match.group(1)) >= int(floor.group(1))


def test_core_pin_is_the_dev_tag_source_alone() -> None:
    uv = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"
    ]["uv"]
    source = uv["sources"]["omnibase-core"]
    assert set(source) == {"git", "tag"}, f"core source must be git+tag only: {source}"
    assert re.fullmatch(r"dev-\d{14}-[0-9a-f]{12}", source["tag"]), source["tag"]
    assert not [
        requirement
        for requirement in uv["override-dependencies"]
        if re.split(r"[=<>!~;\[ ]", requirement, maxsplit=1)[0]
        .lower()
        .replace("_", "-")
        == "omnibase-core"
    ], "omnibase-core is pinned by its dev-merge tag alone (OMN-17427)"


def test_caller_verifier_admits_override_dependency_pins() -> None:
    """The verifier is an omnimarket dev-merge tag, which descends from omnimarket#3694.

    omnimarket#3694 admits tool.uv.override-dependencies in classify_dependency_pin_only.
    The reusable refuses a verifier-tag off omnimarket dev or before its floor
    commit (OMN-21853), and a version number is no longer compared.
    """
    data = yaml.safe_load(CALLER_PATH.read_text(encoding="utf-8"))
    jobs = data["jobs"]
    assert "repo-evidence" in jobs, f"caller jobs are {sorted(jobs)}"
    with_block = jobs["repo-evidence"].get("with")
    assert isinstance(with_block, dict), "repo-evidence job must carry a with: block"
    assert "verifier-version" not in with_block, f"with keys are {sorted(with_block)}"
    tag = str(with_block.get("verifier-tag"))
    assert re.fullmatch(_VERIFIER_TAG_SHAPE, tag), f"with keys are {sorted(with_block)}"
    assert _admits_override_pins(tag), tag


def test_floor_refuses_the_tags_that_refused_the_core_bumps() -> None:
    assert not _admits_override_pins("dev-20261016123307-0a385ce46cdf")
    assert not _admits_override_pins("0.4.309")
    assert not _admits_override_pins("not-a-tag")
    assert _admits_override_pins("dev-20261016123308-0a385ce46cdf")
    assert _admits_override_pins("dev-20261101000200-949993edf3b7")
