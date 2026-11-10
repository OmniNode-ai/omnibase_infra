# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The committed uv.lock carries the refreshed omnibase-spi pin (OMN-13902)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_uv_lock_pins_omnibase_spi_0_23_15() -> None:
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    match = re.search(r'name = "omnibase-spi"\nversion = "([^"]+)"', lock)
    assert match is not None
    assert match.group(1) == "0.23.15"
