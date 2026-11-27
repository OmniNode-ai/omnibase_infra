# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev main runtime's memory limit clears its measured boot peak (OMN-26316).

On h201 the dev runtime was OOM-killed by its own 3 GiB cgroup limit at
2026-11-25T02:38:51Z and 02:42:21Z, at 3.10 and 3.12 GB of anonymous memory, and
the restart policy looped it 15 times. The limit of 3072M was sized for the
join limit of 4 (2,367 MiB plus 30%); the join limit of 16 joins four times as
many consumer groups at once. A kill happens at the limit, so the recorded
anonymous memory at a kill is a floor of the boot peak, not the peak.
"""

from __future__ import annotations

import json
import re
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
    / "dev_runtime_oom_omn26316"
    / "h201_dev_oom_20261125T0238Z.json"
)

pytestmark = pytest.mark.unit

_HEADROOM_OVER_KILL_FLOOR = 2.0


def _mib(size: str) -> float:
    match = re.fullmatch(r"([1-9][0-9]*)([MG])", size)
    assert match is not None
    return int(match.group(1)) * (1024 if match.group(2) == "G" else 1)


def _declared_main_memory_mib() -> float:
    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    boot = contract.profiles["dev"].processes["main"].boot
    assert boot is not None
    return _mib(boot.memory_limit)


def test_dev_main_memory_limit_clears_the_recorded_oom_kill_floor() -> None:
    recording = json.loads(RECORDING_PATH.read_text(encoding="utf-8"))
    floor_mib = max(kill["anon_rss_kb"] for kill in recording["oom_kills"]) / 1024
    assert _declared_main_memory_mib() >= floor_mib * _HEADROOM_OVER_KILL_FLOOR


def test_the_recorded_limit_is_the_one_that_killed_the_runtime() -> None:
    """Control: the limit on the recording sits under the floor, so the check bites."""
    recording = json.loads(RECORDING_PATH.read_text(encoding="utf-8"))
    floor_mib = max(kill["anon_rss_kb"] for kill in recording["oom_kills"]) / 1024
    recorded_limit_mib = recording["memory_limit_bytes"] / (1024 * 1024)
    assert recorded_limit_mib < floor_mib * _HEADROOM_OVER_KILL_FLOOR
