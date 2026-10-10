# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20074 S8: the hardcoded-topic hook is consumed from omnibase_core.

``no-hardcoded-topics`` moved from onex_change_control into omnibase_core under
the same hook id. This repository's ``.pre-commit-config.yaml`` must take it
from the omnibase_core repo block, pinned to a full commit sha, and carry the
keys the OCC block carried (``require_serial``).
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
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _owners(hook_id: str) -> list[tuple[str, str, dict[str, Any]]]:
    """Return ``(repo url, rev, hook entry)`` for every block that lists *hook_id*."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return [
        (block["repo"], str(block.get("rev", "")), hook)
        for block in config["repos"]
        for hook in block.get("hooks", [])
        if hook.get("id") == hook_id
    ]


def test_no_hardcoded_topics_comes_from_omnibase_core() -> None:
    owners = _owners("no-hardcoded-topics")
    assert [url for url, _, _ in owners] == [CORE_REPO]
    _, rev, hook = owners[0]
    assert FULL_SHA.match(rev), f"omnibase_core rev must be a full sha, got {rev!r}"
    assert hook.get("require_serial") is True
    assert hook.get("exclude")


def test_no_hardcoded_topics_is_not_consumed_from_onex_change_control() -> None:
    assert OCC_REPO not in {url for url, _, _ in _owners("no-hardcoded-topics")}
