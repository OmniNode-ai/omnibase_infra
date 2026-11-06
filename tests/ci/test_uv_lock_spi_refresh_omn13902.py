# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The committed uv.lock carries the refreshed omnibase-spi release (OMN-13902)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_uv_lock_pins_refreshed_omnibase_spi() -> None:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {p["name"]: p["version"] for p in lock["package"]}
    patch = int(versions["omnibase-spi"].split(".")[2])
    assert patch >= 14, versions["omnibase-spi"]
