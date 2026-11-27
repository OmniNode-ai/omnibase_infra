# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26227: the core pin contract's evidence reads the repository, not GitHub.

``dod-omn26227-ac2-ledger`` once called ``gh`` to read a compose file from the
pull request branch. A lane cannot read GitHub, and the call hit the verifier's
30 second ceiling, so the item could not be verified from a checkout. Its
replacement, ``dod-omn26227-ac2-ledger-local``, reads the checked-out tree only.
This test holds both facts: no check of the contract runs ``gh``, and the local
item exists with its checks reading the two compose files and running the
ledger-source binding tests.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "contracts" / "OMN-26227.yaml"
LOCAL_ITEM = "dod-omn26227-ac2-ledger-local"
RETIRED_ITEM = "dod-omn26227-ac2-ledger"
COMPOSE_FILES = ("docker-compose.dev-lane.yml", "docker-compose.dev-tenant-fold.yml")


def _items() -> list[dict[str, Any]]:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    return list(contract["dod_evidence"])


def _check_values(item: dict[str, Any]) -> list[str]:
    return [check["check_value"] for check in item["checks"]]


def _runs_gh(check_value: str) -> bool:
    """True when any word of the command line, split by shell rules, is ``gh``."""
    return any(
        token == "gh" or token.endswith("/gh")
        for token in shlex.shlex(check_value, posix=True, punctuation_chars=True)
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("gh api repos/o/r/contents/f --jq .content", True),
        ("grep x f > /dev/null && gh api 'repos/o/r'", True),
        ("grep -q 'ghost' docker/docker-compose.dev-lane.yml", False),
        ("uv run --frozen pytest tests/ci -q", False),
    ],
)
def test_gh_detector_has_a_positive_and_a_negative_control(
    command: str, expected: bool
) -> None:
    assert _runs_gh(command) is expected


def test_no_check_in_the_contract_runs_gh() -> None:
    offenders = [
        (item["id"], value)
        for item in _items()
        for value in _check_values(item)
        if _runs_gh(value)
    ]
    assert not offenders, f"checks that read GitHub from the contract: {offenders}"


def test_the_retired_github_reading_item_is_gone() -> None:
    assert RETIRED_ITEM not in {item["id"] for item in _items()}


def test_local_item_reads_both_compose_files_and_runs_the_binding_tests() -> None:
    matches = [item for item in _items() if item["id"] == LOCAL_ITEM]
    assert len(matches) == 1, f"{LOCAL_ITEM} must be declared exactly once"
    joined = "\n".join(_check_values(matches[0]))
    for compose in COMPOSE_FILES:
        assert f"docker/{compose}" in joined
    assert "tests/ci/test_dev_lane_ledger_source_binding.py" in joined
