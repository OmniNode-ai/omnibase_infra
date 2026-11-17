# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25179: the evidence checks of ``contracts/OMN-24611.yaml`` can fail for what they bind.

AC2, AC3 and AC4 of the dev signing key contract are readings of other systems: the
secret store's own node (an omnimarket change), the key check values the dev agent and
the store serve, and one signed no-op on the dev host with the live unit digest before
and after. A check that greps a comment this pull request adds, or runs a test on a
synthetic value, passes whether or not any of those readings exists. These checks read
the reading itself, so each is red while the reading is absent and red when it
disagrees:

* AC2 reads the omnimarket sibling checkout at ``origin/dev``;
* AC3 reads ``context.evidence_readings`` -- the two check values must be present and
  equal;
* AC4 reads ``context.evidence_readings`` -- the no-op must be ``accepted`` and the two
  unit digests present and equal.

Every value below is a synthetic throwaway.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "OMN-24611.yaml"
_READINGS_KEY = "evidence_readings"

_CHECK = "a1b2c3d4e5f60718"
_DIGEST = "0123456789abcdef" * 4


def _contract() -> dict[str, Any]:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{_CONTRACT} is not a mapping"
    return document


def _readings_check(document: dict[str, Any], criterion: str) -> dict[str, Any]:
    """The one check of ``criterion`` that reads ``context.evidence_readings``."""
    found = [
        check
        for item in document["dod_evidence"]
        if criterion in item["binds_ac"]
        for check in item["checks"]
        if "'context'" in str(check["check_value"])
    ]
    assert len(found) == 1, (criterion, len(found))
    return dict(found[0])


def _run(command: str, cwd: Path) -> int:
    """Run a contract check command the way the verifier does: a shell in ``cwd``."""
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    return result.returncode


def _with_contract(tmp_path: Path, document: dict[str, Any]) -> Path:
    (tmp_path / "contracts").mkdir(exist_ok=True)
    (tmp_path / "contracts" / "OMN-24611.yaml").write_text(
        yaml.safe_dump(document), encoding="utf-8"
    )
    return tmp_path


def _set(document: dict[str, Any], **readings: str) -> dict[str, Any]:
    changed = copy.deepcopy(document)
    changed["context"][_READINGS_KEY].update(readings)
    return changed


_AC3_AGREE = {"ac3_agent_check_value": _CHECK, "ac3_store_check_value": _CHECK}
_AC4_GOOD = {
    "ac4_noop_result": "accepted",
    "ac4_live_unit_digest_before": _DIGEST,
    "ac4_live_unit_digest_after": _DIGEST,
}


# ---- the contract declares the slots each check reads ------------------------


def test_the_contract_declares_every_reading_slot_the_checks_read() -> None:
    document = _contract()
    slots = document["context"][_READINGS_KEY]
    assert set(slots) == {*_AC3_AGREE, *_AC4_GOOD}, sorted(slots)
    for criterion in ("AC3", "AC4"):
        command = str(_readings_check(document, criterion)["check_value"])
        named = [slot for slot in slots if slot in command]
        assert named, f"{criterion}: its check reads none of the declared slots"
        for slot in named:
            assert slot in slots, slot


def test_every_slot_is_read_by_a_check() -> None:
    """A recorded reading nothing reads is evidence nothing can fail on."""
    document = _contract()
    commands = " ".join(
        str(check["check_value"])
        for item in document["dod_evidence"]
        for check in item["checks"]
    )
    for slot in document["context"][_READINGS_KEY]:
        assert slot in commands, slot


def test_the_readings_are_open_until_a_lane_records_them() -> None:
    """No reading is invented here: the slots are empty and the criteria stay open."""
    slots = _contract()["context"][_READINGS_KEY]
    assert all(value == "" for value in slots.values()), slots


# ---- AC3: the two check values are present and equal --------------------------


def test_ac3_is_red_while_the_readings_are_absent(tmp_path: Path) -> None:
    command = str(_readings_check(_contract(), "AC3")["check_value"])
    assert _run(command, _with_contract(tmp_path, _contract())) == 1


def test_ac3_is_green_when_both_check_values_agree(tmp_path: Path) -> None:
    command = str(_readings_check(_contract(), "AC3")["check_value"])
    root = _with_contract(tmp_path, _set(_contract(), **_AC3_AGREE))
    assert _run(command, root) == 0


def test_ac3_is_red_when_the_check_values_differ(tmp_path: Path) -> None:
    command = str(_readings_check(_contract(), "AC3")["check_value"])
    other = {**_AC3_AGREE, "ac3_store_check_value": "ffffffffffffffff"}
    assert _run(command, _with_contract(tmp_path, _set(_contract(), **other))) == 1


def test_ac3_is_red_when_the_reading_is_removed(tmp_path: Path) -> None:
    command = str(_readings_check(_contract(), "AC3")["check_value"])
    document = _set(_contract(), **_AC3_AGREE)
    del document["context"][_READINGS_KEY]["ac3_store_check_value"]
    assert _run(command, _with_contract(tmp_path, document)) == 1


def test_ac3_is_red_when_the_values_are_not_check_values(tmp_path: Path) -> None:
    command = str(_readings_check(_contract(), "AC3")["check_value"])
    junk = {"ac3_agent_check_value": "not-hex", "ac3_store_check_value": "not-hex"}
    assert _run(command, _with_contract(tmp_path, _set(_contract(), **junk))) == 1


# ---- AC4: the no-op is accepted and the live unit did not change ---------------


def test_ac4_is_red_while_the_readings_are_absent(tmp_path: Path) -> None:
    command = str(_readings_check(_contract(), "AC4")["check_value"])
    assert _run(command, _with_contract(tmp_path, _contract())) == 1


def test_ac4_is_green_for_an_accepted_no_op_and_an_unchanged_unit(
    tmp_path: Path,
) -> None:
    command = str(_readings_check(_contract(), "AC4")["check_value"])
    root = _with_contract(tmp_path, _set(_contract(), **_AC4_GOOD))
    assert _run(command, root) == 0


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"ac4_noop_result": "rejected"}, id="rejected-no-op"),
        pytest.param({"ac4_noop_result": ""}, id="no-no-op-result"),
        pytest.param(
            {"ac4_live_unit_digest_after": "f" * 64}, id="live-unit-digest-differs"
        ),
        pytest.param({"ac4_live_unit_digest_before": ""}, id="digest-before-removed"),
        pytest.param({"ac4_live_unit_digest_after": "short"}, id="digest-not-a-digest"),
    ],
)
def test_ac4_is_red_when_what_it_binds_fails(
    tmp_path: Path, change: dict[str, str]
) -> None:
    command = str(_readings_check(_contract(), "AC4")["check_value"])
    root = _with_contract(tmp_path, _set(_contract(), **{**_AC4_GOOD, **change}))
    assert _run(command, root) == 1


# ---- AC2: the store entry is on omnimarket's dev branch -------------------------


def _omnimarket_clone(tmp_path: Path, *, with_entry: bool, with_test: bool) -> Path:
    """A throwaway repository whose ``origin/dev`` ref holds the files the check reads."""
    repo = tmp_path / "omnimarket"
    node = repo / "src/omnimarket/nodes/node_secret_store_effect"
    tests = repo / "tests/unit/nodes/node_secret_store_effect"
    node.mkdir(parents=True)
    tests.mkdir(parents=True)
    entry = "key: DEPLOY_AGENT_DEV_HMAC_SECRET" if with_entry else "key: OTHER_SECRET"
    (node / "contract.yaml").write_text(f"entries:\n  - {entry}\n")
    if with_test:
        (tests / "test_generate_deploy_agent_dev_key.py").write_text("")
    (repo / "README").write_text("x")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    for argv in (
        [*git, "init", "-q"],
        [*git, "add", "-A"],
        [*git, "commit", "-q", "-m", "x"],
        [*git, "update-ref", "refs/remotes/origin/dev", "HEAD"],
    ):
        subprocess.run(argv, cwd=repo, check=True, capture_output=True)
    return repo


def _ac2_store_check() -> dict[str, Any]:
    document = _contract()
    found = [
        check
        for item in document["dod_evidence"]
        if "AC2" in item["binds_ac"]
        for check in item["checks"]
        if str(check["cwd"]).endswith("/omnimarket")
    ]
    assert len(found) == 1, len(found)
    return dict(found[0])


def test_ac2_reads_the_omnimarket_sibling_checkout() -> None:
    check = _ac2_store_check()
    assert "DEPLOY_AGENT_DEV_HMAC_SECRET" in str(check["check_value"])
    assert "origin/dev" in str(check["check_value"])


def test_ac2_is_green_when_the_entry_and_its_generate_test_are_on_dev(
    tmp_path: Path,
) -> None:
    repo = _omnimarket_clone(tmp_path, with_entry=True, with_test=True)
    assert _run(str(_ac2_store_check()["check_value"]), repo) == 0


@pytest.mark.parametrize(
    ("with_entry", "with_test"),
    [
        pytest.param(False, True, id="entry-missing"),
        pytest.param(True, False, id="generate-test-missing"),
    ],
)
def test_ac2_is_red_when_either_is_missing(
    tmp_path: Path, with_entry: bool, with_test: bool
) -> None:
    repo = _omnimarket_clone(tmp_path, with_entry=with_entry, with_test=with_test)
    assert _run(str(_ac2_store_check()["check_value"]), repo) != 0
