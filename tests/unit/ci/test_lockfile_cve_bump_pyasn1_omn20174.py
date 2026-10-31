# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20174: uv.lock resolves pyasn1 past its HIGH advisories.

The Lockfile CVE Scan blocks on pyasn1 0.6.4 (GHSA-2qm8-x737-gmw5,
GHSA-9m75-vc3g-4x39, GHSA-fm9c-5rcf-2q5p, GHSA-mm57-4v3p-rchx, fixed in
0.6.5). This test reads the real uv.lock and fails while pyasn1 is locked
below that floor.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
PYASN1_FIXED_FLOOR = (0, 6, 5)


def test_pyasn1_fixed_version_locked() -> None:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    matches = [p for p in lock["package"] if p["name"] == "pyasn1"]
    assert len(matches) == 1, f"expected exactly one pyasn1 entry, got {len(matches)}"
    version = str(matches[0]["version"])
    try:
        parsed = tuple(int(part) for part in version.split("."))
    except ValueError:
        pytest.fail(f"version has a non-integer component: {version!r}")
    assert parsed >= PYASN1_FIXED_FLOOR, (
        f"pyasn1 {version} is below fixed floor {PYASN1_FIXED_FLOOR}"
    )
