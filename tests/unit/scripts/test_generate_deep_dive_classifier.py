# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for is_exempt_pr in generate_deep_dive.py [OMN-6921]."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# generate_deep_dive.py is a standalone script, not a package module.
# Import it via importlib to avoid sys.path manipulation.
_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "generate_deep_dive.py"
_spec = importlib.util.spec_from_file_location("generate_deep_dive", _SCRIPT_PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["generate_deep_dive"] = _mod
_spec.loader.exec_module(_mod)
classify_pr = _mod.classify_pr
is_exempt_pr = _mod.is_exempt_pr


class TestIsExemptPr:
    """Test the is_exempt_pr function."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "title",
        [
            "chore(deps): bump requests from 2.31.0 to 2.32.0",
            "chore(deps-dev): bump pytest from 8.0 to 8.1",
            "build(deps): bump actions/checkout from 4 to 6",
            "Bump hashicorp/aws from 6.36.0 to 6.37.0",
            "chore(deps)(deps): bump hashicorp/aws from 6.36.0 to 6.37.0",
            "chore: release omnibase_core v0.34.0",
            "chore(release): v0.12.0",
            "chore(release): omniintelligence v0.18.0",
            "release: omnibase_infra v0.29.0",
        ],
    )
    def test_exempt_titles(self, title: str) -> None:
        assert is_exempt_pr(title) is True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "title",
        [
            "feat: add session registry [OMN-6853]",
            "fix(ci): resolve flaky test [OMN-6878]",
            "feat: multi-model adversarial review system",
            "refactor: migrate ONEX state paths",
        ],
    )
    def test_non_exempt_titles(self, title: str) -> None:
        assert is_exempt_pr(title) is False
