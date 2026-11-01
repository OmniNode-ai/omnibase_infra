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
    assert re.fullmatch(_VERIFIER_TAG_SHAPE, str(with_block.get("verifier-tag"))), (
        f"with keys are {sorted(with_block)}"
    )
