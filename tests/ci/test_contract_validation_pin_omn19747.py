# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""contract-validation pins a change-control validator that accepts binds_ac (OMN-19747)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

_WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github/workflows/contract-validation.yml"
)


@pytest.mark.live_contact("tests/ci/fixtures/occ_validate_contract_lock_omn19747.json")
def test_validate_contract_pin_matches_recorded_lock(
    recorded_response: dict[str, Any],
) -> None:
    uses = re.findall(r"validate-contract@(\S+)", _WORKFLOW.read_text(encoding="utf-8"))
    assert uses == [recorded_response["ref"]]
    locked = tuple(
        int(p) for p in recorded_response["locked_omnibase_core_version"].split(".")
    )
    stale = tuple(
        int(p)
        for p in recorded_response["stale_main_locked_omnibase_core_version"].split(".")
    )
    assert locked > stale
    assert locked >= (0, 47, 0)
