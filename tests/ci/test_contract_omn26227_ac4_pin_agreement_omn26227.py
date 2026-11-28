# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26227: the AC4 evidence names no literal core tag, so a pin move cannot break it.

``dod-omn26227-ac4`` grepped ``pyproject.toml`` for one literal core dev tag.
The Done gate reads this contract at the commit of the ticket's last merged
pull request, and runs its checks against the tree of ``dev`` today. Every pin
move after that merge made the literal tag stale, so the gate read the item as
failed while the pin and the lock still agreed. Its replacement,
``dod-omn26227-ac4-pin-agreement``, holds the claim that does not move: the
core pin is a dev tag, and the lock records the core release that tag names.
This test holds both facts: no check bound to AC4 carries a literal dev tag,
and the replacement item exists with the lock check, the tag-shape check and
the pin and lock agreement test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "contracts" / "OMN-26227.yaml"
REPLACEMENT_ITEM = "dod-omn26227-ac4-pin-agreement"
RETIRED_ITEM = "dod-omn26227-ac4"
LITERAL_DEV_TAG = re.compile(r"dev-\d{14}-[0-9a-f]{12}")


def _items() -> list[dict[str, Any]]:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    return list(contract["dod_evidence"])


def _ac4_check_values() -> list[str]:
    return [
        check["check_value"]
        for item in _items()
        if "AC4" in item.get("binds_ac", [])
        for check in item["checks"]
    ]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("grep -q 'dev-20261127051535-94ed7507315d' pyproject.toml", True),
        ("grep -Eq 'tag = \"dev-[0-9]{14}-[0-9a-f]{12}\"' pyproject.toml", False),
        ("uv lock --check", False),
    ],
)
def test_literal_tag_detector_has_a_positive_and_a_negative_control(
    command: str, expected: bool
) -> None:
    assert bool(LITERAL_DEV_TAG.search(command)) is expected


def test_no_ac4_check_names_a_literal_dev_tag() -> None:
    offenders = [v for v in _ac4_check_values() if LITERAL_DEV_TAG.search(v)]
    assert not offenders, f"AC4 checks that go stale at the next pin move: {offenders}"


def test_the_literal_tag_item_is_retired() -> None:
    assert RETIRED_ITEM not in {item["id"] for item in _items()}


def test_replacement_item_binds_ac4_and_checks_lock_tag_shape_and_agreement() -> None:
    matches = [item for item in _items() if item["id"] == REPLACEMENT_ITEM]
    assert len(matches) == 1, f"{REPLACEMENT_ITEM} must be declared exactly once"
    assert matches[0]["binds_ac"] == ["AC4"]
    joined = "\n".join(check["check_value"] for check in matches[0]["checks"])
    assert "uv lock --check" in joined
    assert r"omnibase-core = \{ git" in joined
    assert (
        "tests/ci/test_core_dev_tag_pin_omn17427.py" in joined
        and "pyproject_core_tag_and_lock_core_release_name_the_same_release" in joined
    )
