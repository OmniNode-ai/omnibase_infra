# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keep PR-level pre-commit exemptions limited to the OMN-19854 gates."""

from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
def test_required_checks_no_precommit_twin() -> None:
    manifest_path = Path(__file__).resolve().parents[2] / ".github/required-checks.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    gates = {gate["name"]: gate for gate in manifest["gates"]}
    exempt_gates = {"Hostile Review Thread Gate", "CI Summary"}

    for name in exempt_gates:
        assert gates[name].get("pre_commit") == "not-applicable", name

    for gate in manifest["gates"]:
        if gate["name"] not in exempt_gates:
            assert "pre_commit" not in gate, gate["name"]
