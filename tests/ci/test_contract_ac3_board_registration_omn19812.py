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


def test_exactly_one_item_binds_ac3_and_runs_the_tickets_falsifier() -> None:
    data = _load()
    binders = [i for i in data["dod_evidence"] if "AC3" in i.get("binds_ac", [])]
    assert len(binders) == 1
    checks = binders[0]["checks"]
    assert checks
    assert any(
        c["check_type"] == "command" and FALSIFIER in c["check_value"] for c in checks
    )


def test_the_ac3_check_resolves_the_sibling_checkout_from_an_env_var() -> None:
    data = _load()
    binder = next(i for i in data["dod_evidence"] if "AC3" in i.get("binds_ac", []))
    commands = [c for c in binder["checks"] if c["check_type"] == "command"]
    assert commands
    for check in commands:
        assert 'cd "${BOARD_REPO_DIR:?' in check["check_value"]
