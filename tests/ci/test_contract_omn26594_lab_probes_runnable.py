# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy probes of contracts/OMN-26594.yaml run on a lab host and can fail.

The three local Done gate probes began ``ssh omni-201 ...``. ``omni-201`` does not resolve from any
lab host, so each exited 255 and the Done write was refused. They now read the runtime container
where they run, and fail when ``omninode-runtime`` is absent so a host without it proves nothing.
The contract also retires the bindings of AC4 to AC7, which the live ticket no longer carries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-26594.yaml"
PROBES = (
    "dod-omn26594-ac3-deploy-probe",
    "dod-omn26594-ac5-deploy-probe",
    "dod-omn26594-ac7-deploy-probe",
)
RETIRED = {
    ("dod-omn26594-ac4", "AC4"),
    ("dod-omn26594-ac5", "AC5"),
    ("dod-omn26594-ac6", "AC6"),
    ("dod-omn26594-ac6", "AC7"),
}


def _contract() -> dict[str, Any]:
    loaded = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _command(item_id: str) -> str:
    items = [i for i in _contract()["dod_evidence"] if i["id"] == item_id]
    assert len(items) == 1, item_id
    return str(items[0]["checks"][0]["check_value"])


def _problems(command: str) -> list[str]:
    found = []
    if "omni-201" in command or command.lstrip().startswith("ssh"):
        found.append("names a host a lab machine cannot resolve")
    if not command.lstrip().startswith("docker inspect omninode-runtime"):
        found.append("does not require the runtime container first")
    return found


@pytest.mark.parametrize("item_id", PROBES)
def test_probe_runs_on_the_host_holding_the_runtime(item_id: str) -> None:
    assert _problems(_command(item_id)) == []


def test_check_rejects_the_ssh_form() -> None:
    bad = "ssh omni-201 '! docker logs --since 15m omninode-runtime 2>&1 | grep -q x'"
    assert len(_problems(bad)) == 2


def test_contract_retires_the_bindings_the_ticket_no_longer_has() -> None:
    retired = {
        (entry["item"], entry["label"])
        for item in _contract()["dod_evidence"]
        for entry in item.get("supersedes_ac_binding", [])
    }
    assert retired == RETIRED
    declared = {a["id"] for r in _contract()["requirements"] for a in r["acceptance"]}
    assert declared == {"AC1", "AC2", "AC3"}
