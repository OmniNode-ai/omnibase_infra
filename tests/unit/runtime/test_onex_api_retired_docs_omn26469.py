# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26469: every docs line naming onex-api must say it is retired."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

DOCS_ROOT = Path(__file__).resolve().parents[3] / "docs"


@pytest.mark.unit
def test_docs_name_onex_api_only_as_retired() -> None:
    offenders = [
        f"{path.relative_to(DOCS_ROOT.parent)}:{number}: {line.strip()}"
        for path in sorted(DOCS_ROOT.rglob("*"))
        if path.is_file()
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
        )
        if "onex-api" in line.lower() and "retired" not in line.lower()
    ]
    assert not offenders, "\n".join(offenders)
