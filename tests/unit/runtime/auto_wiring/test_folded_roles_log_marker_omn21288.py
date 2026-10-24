# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the folded-roles log marker shared by the refresh probe and kernel."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_folded_roles_log_marker_matches_kernel() -> None:
    root = Path(__file__).resolve().parents[4]
    script = (root / "scripts/runtime_build/refresh_stability_lane.sh").read_text(
        encoding="utf-8"
    )
    kernel = (root / "src/omnibase_infra/runtime/service_kernel.py").read_text(
        encoding="utf-8"
    )

    match = re.search(r'readonly FOLDED_ROLES_LOG_MARKER="([^"]+)"', script)
    assert match is not None
    marker = match.group(1)
    assert 'grep -F "${FOLDED_ROLES_LOG_MARKER}"' in script
    assert f'"{marker}: profile=%s' in kernel
