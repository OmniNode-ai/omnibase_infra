# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19812 contract binds AC3, the board registration of the C28 workflow."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CONTRACT = Path(__file__).resolve().parents[2] / "contracts/OMN-19812.yaml"
FALSIFIER = "tests/scripts/test_c28_consumer_flow_probe_registration.py"


def _load() -> dict[str, object]:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def test_ac3_is_declared_once_with_the_tickets_statement() -> None:
    data = _load()
    declared = [
        ac
        for req in data["requirements"]
        for ac in req.get("acceptance", [])
        if ac["id"] == "AC3"
    ]
    assert len(declared) == 1
    assert declared[0]["statement"].startswith(
        "AC3: beta_board_data.py PROBES carries exactly one C28 entry naming "
        "that workflow"
    )


def test_exactly_one_item_binds_ac3_with_a_hosted_runnable_check() -> None:
    data = _load()
    binders = [i for i in data["dod_evidence"] if "AC3" in i.get("binds_ac", [])]
    assert len(binders) == 1
    checks = binders[0]["checks"]
    assert checks
    assert any(c["check_type"] == "test_passes" for c in checks)
    statement = next(
        a["statement"]
        for r in data["requirements"]
        for a in r["acceptance"]
        if a["id"] == "AC3"
    )
    assert FALSIFIER in statement


def test_no_ac3_check_needs_the_sibling_checkout() -> None:
    data = _load()
    binder = next(i for i in data["dod_evidence"] if "AC3" in i.get("binds_ac", []))
    for check in binder["checks"]:
        assert "BOARD_REPO_DIR" not in check["check_value"]
