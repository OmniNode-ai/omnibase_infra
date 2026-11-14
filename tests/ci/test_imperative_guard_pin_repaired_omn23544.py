# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23544: the imperative guard caller names the repaired shared check tag."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CALLER = ROOT / ".github/workflows/imperative-contract-guard.yml"
REPAIRED_TAG = "dev-20261110125852-e6bc5e13e9e7"
OLD_TAG = "dev-20261105190003-87a6f5ddcbcc"


def test_caller_names_repaired_tag() -> None:
    text = CALLER.read_text(encoding="utf-8")
    assert f"imperative-contract-guard.yml@{REPAIRED_TAG}" in text


def test_old_tag_is_gone_from_pins() -> None:
    for path in (
        CALLER,
        ROOT / ".github/workflows/ci.yml",
        ROOT / ".pre-commit-config.yaml",
    ):
        assert OLD_TAG not in path.read_text(encoding="utf-8"), path.name
