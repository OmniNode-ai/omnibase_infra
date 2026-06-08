# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for the main-target guard workflow."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_TARGET_GUARD = REPO_ROOT / ".github" / "workflows" / "main-target-guard.yml"


def test_main_target_guard_allows_promotion_branches_with_receipts() -> None:
    workflow = MAIN_TARGET_GUARD.read_text(encoding="utf-8")

    assert '"${HEAD_REF}" == "dev" || "${HEAD_REF}" == promotion/*' in workflow
    assert "promotion-receipt:[[:space:]]*OCC-[0-9]+" in workflow
    assert "Promotion PR from '${HEAD_REF}' with valid promotion-receipt" in workflow


def test_main_target_guard_keeps_hotfix_evidence_requirements() -> None:
    workflow = MAIN_TARGET_GUARD.read_text(encoding="utf-8")

    assert '"${HEAD_REF}" == hotfix/*' in workflow
    assert "hotfix-evidence:[[:space:]]*OCC-[0-9]+" in workflow
    assert "backmerge:[[:space:]]*#[0-9]+" in workflow
