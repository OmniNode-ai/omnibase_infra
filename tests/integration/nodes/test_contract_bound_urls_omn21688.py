# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two node contracts, read from disk, bind the URLs their code uses (OMN-21688)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_board_probe_effect import contract_topics
from omnibase_infra.nodes.node_board_probe_effect.models import (
    ModelConsumerFlowRequest,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect import contract_api
from omnibase_infra.nodes.node_merge_provenance_observe_effect.handlers.handler_merge_group_run_read_github import (
    HandlerMergeGroupRunReadGithub,
)

pytestmark = pytest.mark.integration


def _load(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_board_probe_request_default_equals_the_shipped_contract_value() -> None:
    contract = Path(contract_topics.__file__).resolve().parent / "contract.yaml"
    block = _load(contract)["consumer_flow_endpoints"]
    assert isinstance(block, dict)
    declared = block["projection_api_base_url"]
    assert declared
    assert contract_topics.consumer_flow_projection_api_base_url() == declared
    assert (
        ModelConsumerFlowRequest(
            subject_lane="dev", compose_project="omnibase-infra"
        ).base_url
        == declared
    )


def test_merge_group_reader_binds_the_shipped_contract_value() -> None:
    contract = Path(contract_api.__file__).resolve().parent / "contract.yaml"
    block = _load(contract)["github_api"]
    assert isinstance(block, dict)
    declared = block["base_url"]
    assert declared
    assert contract_api.github_api_url() == declared
    reader = HandlerMergeGroupRunReadGithub(token="t")
    assert reader._api_url == str(declared).rstrip("/")
