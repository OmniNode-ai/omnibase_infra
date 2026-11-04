# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20919 S8: the bare-feature-flag hook is consumed from omnibase_core.

``no-bare-feature-flags`` moved from onex_change_control into omnibase_core
(omnibase_core#1936, merge commit 9218fe703b08d329f76eecd37fabbbc7b0c2baad)
under the same hook id. This repository's ``.pre-commit-config.yaml`` must take
it from an omnibase_core repo block pinned to a dev-merge tag, keep
``require_serial``, and name no onex_change_control repository at all.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

CONFIG = Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml"
CORE_REPO = "https://github.com/OmniNode-ai/omnibase_core"
OCC_REPO = "https://github.com/OmniNode-ai/onex_change_control"
DEV_MERGE_TAG = re.compile(r"^dev-\d{14}-[0-9a-f]{12}$")


def _owners(hook_id: str) -> list[tuple[str, str, dict[str, Any]]]:
    """Return ``(repo url, rev, hook entry)`` for every block that lists *hook_id*."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return [
        (block["repo"], str(block.get("rev", "")), hook)
        for block in config["repos"]
        for hook in block.get("hooks", [])
        if hook.get("id") == hook_id
    ]


def test_no_bare_feature_flags_comes_from_omnibase_core() -> None:
    owners = _owners("no-bare-feature-flags")
    assert [url for url, _, _ in owners] == [CORE_REPO]
    _, rev, hook = owners[0]
    assert DEV_MERGE_TAG.match(rev), (
        f"omnibase_core rev must be a dev-merge tag, got {rev!r}"
    )
    assert hook.get("require_serial") is True


def test_config_names_no_onex_change_control_repository() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert OCC_REPO not in {block["repo"] for block in config["repos"]}


def test_no_hardcoded_topics_uses_a_core_dev_merge_tag() -> None:
    """The topic hook is an omnibase_core hook and takes a dev-merge tag too."""
    owners = _owners("no-hardcoded-topics")
    assert [url for url, _, _ in owners] == [CORE_REPO]
    assert DEV_MERGE_TAG.match(owners[0][1])
