# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The committed uv.lock carries the format revision this refresh moved it to (OMN-13902)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_uv_lock_is_at_format_revision_5() -> None:
    header = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8").splitlines()[:3]
    assert "revision = 5" in header
