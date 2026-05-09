# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for the unified env fallback validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_no_env_fallbacks import run

pytestmark = pytest.mark.integration


def test_unified_env_fallback_validator_is_clean_for_runtime_roots() -> None:
    repo_root = Path(__file__).parents[2]
    violations = run([repo_root / "src", repo_root / "scripts"], repo_root)
    assert violations == []
