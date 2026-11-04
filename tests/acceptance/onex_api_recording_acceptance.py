# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Live corpus checks, explicitly invoked with private input paths.

This file is intentionally an explicit acceptance probe rather than a default
unit-test input: a public checkout cannot contain the private recorded bodies.
Missing inputs fail; this probe never skips or substitutes synthetic recordings.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


def load_input(variable: str) -> dict:
    path = os.environ.get(variable, "")
    assert path, f"required private input: {variable}"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_ac1_recorded_identities_equal_inventory() -> None:
    fixture = load_input("ONEX_RECORD_ROUTE_FIXTURE")
    recorded = load_input("ONEX_RECORD_RECORDED_SET")
    expected = Counter((r["method"], r["path"]) for r in fixture["routes"])
    actual = Counter((p["method"], p["path"]) for p in recorded["pairs"])
    assert len(expected) == 63, "expected the complete route inventory"
    assert all(n == 1 for n in expected.values()), "duplicate fixture identity"
    assert actual == expected, "recorded identity multiplicities differ from inventory"
    for pair in recorded["pairs"]:
        assert set(pair["request"]) == {"method", "path", "query", "headers", "body"}
        assert pair["request"]["method"] == pair["method"]
        assert pair["request"]["path"] == pair["path"]
        assert isinstance(pair["response"]["status"], int)
        assert 100 <= pair["response"]["status"] <= 599
        assert "body" in pair["response"]


def masked_values(value: object, pointer: str = "") -> dict[str, str]:
    if isinstance(value, str) and "<MASK:" in value:
        assert re.fullmatch(r"<MASK:[a-z_]+>", value), "malformed masking marker"
        return {pointer: value[6:-1]}
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            result.update(masked_values(child, pointer + "/" + escaped))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(masked_values(child, f"{pointer}/{index}"))
    return result


def test_ac2_named_masks_and_mutation_cleanup() -> None:
    recorded = load_input("ONEX_RECORD_RECORDED_SET")
    assert recorded["pairs"], "empty recorded set cannot establish masking or cleanup"
    contract = Path(__file__).resolve().parents[2] / (
        "src/omnibase_infra/nodes/node_api_fixture_record_effect/contract.yaml"
    )
    rules = yaml.safe_load(contract.read_text())["config"]["recording_policy"][
        "masking_rules"
    ]
    cleanup = set(recorded["cleanup_names"])
    for pair in recorded["pairs"]:
        found = masked_values(
            {"request": pair["request"], "response": pair["response"]}
        )
        assert pair["masks"] == found, "mask annotations must match actual fields"
        assert set(found.values()) <= set(rules), "unknown masking rule"
        if pair["method"] not in {"GET", "HEAD", "OPTIONS"}:
            assert pair["cleanup"], "mutating pair requires cleanup"
        assert set(pair["cleanup"]) <= cleanup, "undeclared cleanup"


def test_ac3_count_receipt_has_no_resource_delta() -> None:
    recorded = load_input("ONEX_RECORD_RECORDED_SET")
    receipt = recorded["lifecycle"]
    assert receipt["cleanup_completed"] is True
    assert receipt["before"] == receipt["after"]
    assert set(receipt["before"]) == {"tenants", "accounts", "keycloak_users"}
    assert all(type(n) is int and n >= 0 for n in receipt["before"].values())
