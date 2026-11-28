# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""No hosted check in contracts/OMN-26121.yaml reads GitHub.

A lane's gh shim refuses every read, and a merged branch is gone, so a check
that reads a PR branch through ``gh api`` can never run again. The check's own
predicate (a grep of the checked-out tree) stays; only the GitHub read goes.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-26121.yaml"


def _command_values(contract: dict) -> list[tuple[str, str]]:
    return [
        (item["id"], check["check_value"])
        for item in contract["dod_evidence"]
        for check in item.get("checks", [])
        if check.get("check_type") == "command"
    ]


def _github_readers(contract: dict) -> list[str]:
    return [i for i, v in _command_values(contract) if "gh api" in v]


def test_no_check_of_the_contract_reads_github() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    assert _command_values(contract), "contract holds no command check"
    assert _github_readers(contract) == []


def test_every_tree_probe_still_greps_the_tree() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    probes = [v for i, v in _command_values(contract) if i.endswith("-deploy")]
    assert len(probes) == 5
    for value in probes:
        assert value.startswith(("grep -F", "grep ")), value


def test_a_check_that_reads_github_is_refused() -> None:
    planted = {
        "dod_evidence": [
            {
                "id": "x",
                "checks": [
                    {"check_type": "command", "check_value": "gh api repos/o/r"}
                ],
            }
        ]
    }
    assert _github_readers(planted) == ["x"]
