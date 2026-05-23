# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration guard for OMN-6659 type-suppression contract touches."""

from __future__ import annotations

from pathlib import Path

import yaml

_NODE_CONTRACTS = (
    "node_artifact_change_detector_effect",
    "node_impact_analyzer_compute",
    "node_llm_inference_effect",
    "node_runtime_error_triage_effect",
    "node_session_state_effect",
)


def test_omn_6659_handler_contracts_are_refreshed() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    for node_name in _NODE_CONTRACTS:
        contract_path = (
            repo_root / "src" / "omnibase_infra" / "nodes" / node_name / "contract.yaml"
        )
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

        assert contract["metadata"]["updated"] == "2026-05-22"
