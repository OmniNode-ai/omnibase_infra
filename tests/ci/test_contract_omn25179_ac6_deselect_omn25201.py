# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25201: the AC6 check of ``contracts/OMN-25179.yaml`` names its ten base failures.

The deploy agent's whole ``tests/unit`` tree reads ``10 failed, 2265 passed`` on the
merge base ``1f5f40f80b2d`` (lab host h051, read 2026-11-17, 16 workers): the failures are in
files this pull request does not touch. The AC6 check therefore deselects those ten by name
and nothing else. These checks pin that list: exactly the ten, each naming a real test
in a real file, and none of them in the dev signing key test or the launcher test, so a
failure of the work itself can never hide in the deselect list.
"""

from __future__ import annotations

import copy
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_CONTRACT = _REPO / "contracts" / "OMN-25179.yaml"
_UNIT_TESTS = "tests/unit"
#: pytest's rootdir for the deploy agent is its own project directory, so a node id
#: reads ``tests/unit/...``; an id spelled from the repository root deselects nothing.
_PROJECT = "scripts/deploy-agent"

#: The ten tests that fail on the merge base 1f5f40f80b2d, as pytest reported them on
#: lab host h051 (read 2026-11-17: ``10 failed, 2265 passed``).
_BASE_FAILURES = frozenset(
    {
        f"{_UNIT_TESTS}/test_companion_restore_omn22845.py::{name}"
        for name in (
            "test_the_default_dev_lane_needs_main_and_effects",
            "test_read_lane_presence[states1-companion_down]",
            "test_read_lane_presence[states2-companion_down]",
            "test_read_lane_presence[states3-companion_down]",
            "test_the_dev_lane_reads_its_own_containers",
            "test_the_down_container_is_named_in_the_detail",
            "test_an_unreadable_effects_container_never_restores",
        )
    }
    | {
        f"{_UNIT_TESTS}/test_dev_refresh_llm_endpoint_env_omn24081.py::"
        "test_seam_the_scripts_own_python_renders_the_overlay_address",
        f"{_UNIT_TESTS}/test_stability_refresh_llm_endpoint_env_omn24119.py::"
        "test_error_the_scripts_own_program_prints_nothing_for_an_overlay_without_the_slot",
        f"{_UNIT_TESTS}/test_stability_refresh_llm_endpoint_env_omn24119.py::"
        "test_golden_the_scripts_own_program_renders_the_overlay_address",
    }
)
_OWN_TEST_FILES = (
    "test_dev_agent_own_signing_key_omn24565.py",
    "test_deploy_agent_launcher_omn18073.py",
)


def _deselected_by(document: dict[str, Any]) -> list[str]:
    commands = [
        str(check["check_value"])
        for item in document["dod_evidence"]
        if "AC6" in item["binds_ac"]
        for check in item["checks"]
    ]
    assert len(commands) == 1, commands
    words = shlex.split(commands[0])
    assert f"{_PROJECT}/{_UNIT_TESTS}" in words, (
        "the AC6 check no longer runs tests/unit"
    )
    return [
        words[index + 1]
        for index, word in enumerate(words[:-1])
        if word == "--deselect"
    ]


def _contract() -> dict[str, Any]:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{_CONTRACT} is not a mapping"
    return document


def _ac6_deselected() -> list[str]:
    return _deselected_by(_contract())


def test_ac6_deselects_exactly_the_ten_base_failures() -> None:
    deselected = _ac6_deselected()
    assert len(deselected) == len(set(deselected)) == 10, deselected
    assert set(deselected) == _BASE_FAILURES, sorted(set(deselected) ^ _BASE_FAILURES)


def test_none_of_the_ten_is_in_the_key_test_or_the_launcher_test() -> None:
    for node_id in _ac6_deselected():
        assert not node_id.split("::")[0].endswith(_OWN_TEST_FILES), node_id


def test_every_deselected_name_is_a_test_in_its_file() -> None:
    """A deselect that names no test deselects nothing and hides nothing -- but is stale."""
    for node_id in _ac6_deselected():
        path, *scope = node_id.split("::")
        text = (_REPO / _PROJECT / path).read_text(encoding="utf-8")
        assert f"def {scope[-1].split('[')[0]}(" in text, node_id
        if len(scope) == 2:
            assert f"class {scope[0]}" in text, node_id


def test_every_id_is_spelled_from_the_project_root() -> None:
    """The failing control for the id shape: from the repository root nothing matches."""
    for node_id in _ac6_deselected():
        assert node_id.startswith(f"{_UNIT_TESTS}/"), node_id
        assert not node_id.startswith(_PROJECT), node_id


@pytest.mark.parametrize("dropped", sorted(_BASE_FAILURES))
def test_control_the_contract_without_one_of_the_ten_fails_the_pin(
    dropped: str,
) -> None:
    """The failing control, run on the contract's own text: drop one id from the AC6 check.

    The command is edited in a copy of the parsed contract and read back through the
    same extractor the pin uses; the pin (``_BASE_FAILURES``, spelled in this file)
    must then no longer hold, for each of the ten.
    """
    document = copy.deepcopy(_contract())
    for item in document["dod_evidence"]:
        if "AC6" in item["binds_ac"]:
            for check in item["checks"]:
                words = shlex.split(str(check["check_value"]))
                kept: list[str] = []
                skip = False
                for index, word in enumerate(words):
                    if skip:
                        skip = False
                    elif word == "--deselect" and words[index + 1] == dropped:
                        skip = True
                    else:
                        kept.append(word)
                check["check_value"] = shlex.join(kept)
    remaining = _deselected_by(document)
    assert len(remaining) == 9, remaining
    assert dropped not in remaining
    assert set(remaining) != _BASE_FAILURES
