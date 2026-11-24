# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev main runtime's declared join limit shortens its measured boot attach (OMN-26316).

The recorded start of the dev runtime on h201 (2026-11-24T20:01:04Z) answered its
first /health 200 at 210 s. Of those, 127 s were the boot attach: 70 contracts
joined their consumer groups with the declared limit of 4 in flight, each slot
holding one contract for the broker's group join. The limit is the only knob
that scales that phase, so the test derives the per-contract slot cost from the
recording and applies the declared limit to it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
RECORDING_PATH = (
    ROOT
    / "tests"
    / "fixtures"
    / "boot_start_omn26316"
    / "h201_dev_start_20261124T200104Z.json"
)

pytestmark = pytest.mark.unit


def _recording() -> dict[str, object]:
    loaded = json.loads(RECORDING_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _declared_main_join_limit() -> int:
    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    boot = contract.profiles["dev"].processes["main"].boot
    assert boot is not None
    return boot.max_concurrent_contract_attach


def _modeled_attach_seconds(limit: int) -> float:
    """Boot attach seconds at ``limit`` in flight, from the recorded slot cost."""
    attach = _recording()["boot_attach"]
    assert isinstance(attach, dict)
    contracts = int(attach["contracts_attached_at_boot"])
    measured_limit = int(attach["join_limit_when_measured"])
    slot_seconds = int(attach["seconds"]) * measured_limit / contracts
    return math.ceil(contracts / limit) * slot_seconds


def test_the_recording_reproduces_its_own_attach_seconds() -> None:
    # Positive control: the model returns the measured 127 s at the measured limit
    # (to within one slot of rounding), so a different answer below is the limit's
    # doing and not the model's.
    attach = _recording()["boot_attach"]
    assert isinstance(attach, dict)
    measured = _modeled_attach_seconds(int(attach["join_limit_when_measured"]))

    assert abs(measured - int(attach["seconds"])) <= 8


def test_the_declared_main_join_limit_halves_the_measured_boot_attach() -> None:
    attach = _recording()["boot_attach"]
    assert isinstance(attach, dict)
    measured_seconds = int(attach["seconds"])

    modeled = _modeled_attach_seconds(_declared_main_join_limit())

    assert modeled <= measured_seconds / 2, (
        f"declared join limit {_declared_main_join_limit()} models {modeled:.0f} s "
        f"of boot attach against {measured_seconds} s measured at limit 4"
    )
