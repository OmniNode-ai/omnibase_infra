# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay of the dev bus lane's declared transport through the event publisher (OMN-19566).

The ``lab-proof-receipt`` workflow publishes the pr-head receipt event through
``publish_lab_fact_event.py``, which reads the lane's protocol from the
omnimarket overlay and refuses a SASL lane without credentials. The recording
is that lane's declaration as the overlay carries it. Replayed through the real
publisher, it must name the credentials the lane needs, and the workflow's
publish step must supply exactly those, from secrets, in its environment.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import lab_pass_receipt as receipts

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PUBLISHER = ROOT / "scripts" / "ci" / "publish_lab_fact_event.py"
WORKFLOW = ROOT / ".github" / "workflows" / "lab-proof-receipt.yml"
CREDENTIAL_ENV = re.compile(r"KAFKA_SASL_[A-Z]+")


def _overlay(tmp_path: Path, recorded: dict[str, object]) -> Path:
    """The overlay with the recorded lane declaration and a placeholder broker."""
    lane = {
        "broker": "broker.invalid:9092",
        "security_protocol": recorded["security_protocol"],
        "sasl_mechanism": recorded["sasl_mechanism"],
    }
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(
        yaml.safe_dump(
            {"default": recorded["default"], "lanes": {recorded["lane"]: lane}}
        ),
        encoding="utf-8",
    )
    return path


def _event(tmp_path: Path) -> Path:
    path = tmp_path / "event.json"
    path.write_text(
        json.dumps(
            {"topic": receipts.LAB_PROOF_RECEIPT_EVENT_TOPIC, "lane": "pr-head"}
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.live_contact("tests/ci/fixtures/omn19566_bus_lane_dev_transport.json")
def test_the_recorded_lane_needs_the_credentials_the_workflow_supplies(
    tmp_path: Path, recorded_response: dict[str, object]
) -> None:
    # positive control: the recorded lane really is a SASL lane
    assert str(recorded_response["security_protocol"]).startswith("SASL")
    done = subprocess.run(
        [
            sys.executable,
            str(PUBLISHER),
            "--event",
            str(_event(tmp_path)),
            "--bus-lane",
            str(recorded_response["lane"]),
            "--bus-overlay",
            str(_overlay(tmp_path, recorded_response)),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
        cwd=ROOT,
    )
    assert done.returncode == 0
    needed = set(CREDENTIAL_ENV.findall(done.stderr))
    assert needed == {"KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"}, done.stderr

    steps: list[dict[str, Any]] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))[
        "jobs"
    ]["post"]["steps"]
    (publish,) = [s for s in steps if "Publish the receipt event" in s.get("name", "")]
    assert set(publish["env"]) >= needed
    for name in needed:
        assert publish["env"][name].startswith("${{ secrets.")
