# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""contract-validation checks out change-control validators at the pin (OMN-19747).

The validate-contract composite action cannot be called by sha (its inner
checkout takes ``github.action_ref``, which resolves to ``v4``), and ``@main``
installs omnibase-core 0.46.7, which rejects ``dod_evidence[].binds_ac``. The
workflow runs the action's steps inline with the onex_change_control checkout
at the ``.github/sibling-pins.yaml`` commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _ROOT / ".github/workflows/contract-validation.yml"
_PINS = yaml.safe_load((_ROOT / ".github/sibling-pins.yaml").read_text())["pins"]


def _steps() -> list[dict[str, object]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps: list[dict[str, object]] = workflow["jobs"]["contract-validation"]["steps"]
    return steps


def test_validate_contract_composite_action_is_not_called() -> None:
    uses = [str(s.get("uses", "")) for s in _steps()]
    assert not [u for u in uses if "validate-contract" in u]


def test_validators_check_out_at_the_declared_pin() -> None:
    occ = [
        s["with"]
        for s in _steps()
        if isinstance(s.get("with"), dict)
        and s["with"].get("repository") == "OmniNode-ai/onex_change_control"
    ]
    assert len(occ) == 1
    assert occ[0]["ref"] == _PINS["onex_change_control"]
    assert occ[0]["path"] == ".onex_change_control_validators"


@pytest.mark.live_contact("tests/ci/fixtures/occ_validate_contract_lock_omn19747.json")
def test_pin_locks_a_core_that_accepts_binds_ac(
    recorded_response: dict[str, Any],
) -> None:
    assert recorded_response["ref"] == _PINS["onex_change_control"]
    locked = tuple(
        int(p) for p in recorded_response["locked_omnibase_core_version"].split(".")
    )
    stale = tuple(
        int(p)
        for p in recorded_response["stale_main_locked_omnibase_core_version"].split(".")
    )
    assert locked > stale
    assert locked >= (0, 47, 0)
