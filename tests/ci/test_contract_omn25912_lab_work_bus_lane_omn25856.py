# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The OMN-25912 integration-proof item names the lab bus lane it publishes on.

``onex lab-work run`` without ``--bus-lane`` publishes nowhere the lab serve
process listens, so the closer refuses the ticket on that alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-25912.yaml"


def test_lab_work_run_names_the_dev_bus_lane() -> None:
    doc = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    commands = [
        " ".join(str(check["check_value"]).split())
        for item in doc["dod_evidence"]
        for check in item.get("checks", [])
        if "lab-work run" in str(check.get("check_value", ""))
    ]
    assert commands, "OMN-25912 binds no lab-work run command"
    for command in commands:
        assert "lab-work run --bus-lane dev " in command, command
