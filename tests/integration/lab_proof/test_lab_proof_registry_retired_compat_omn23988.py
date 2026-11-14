# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lab proof registry and plan node contract name no retired omnibase_compat."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    load_lab_proof_profile_registry,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = REPO_ROOT / "config/lab_proof_profiles.yaml"
PLAN_CONTRACT = (
    REPO_ROOT / "src/omnibase_infra/nodes/node_lab_proof_plan_compute/contract.yaml"
)


def test_registry_loads_without_retired_compat_repository() -> None:
    registry = load_lab_proof_profile_registry(REGISTRY)
    assert registry is not None
    text = REGISTRY.read_text(encoding="utf-8")
    assert "omnibase_compat" not in text


def test_plan_node_contract_names_no_retired_compat_package() -> None:
    assert "omnibase_compat" not in PLAN_CONTRACT.read_text(encoding="utf-8")
