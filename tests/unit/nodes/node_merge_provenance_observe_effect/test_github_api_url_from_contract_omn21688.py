# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The merge-group reader's GitHub API base comes from the node contract (OMN-21688)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect import contract_api
from omnibase_infra.nodes.node_merge_provenance_observe_effect.handlers import (
    handler_merge_group_run_read_github as reader_module,
)
from omnibase_infra.nodes.node_merge_provenance_observe_effect.handlers.handler_merge_group_run_read_github import (
    HandlerMergeGroupRunReadGithub,
)

pytestmark = pytest.mark.unit

_HANDLER_FILE = Path(reader_module.__file__).resolve()
_READER_FILE = Path(contract_api.__file__).resolve()
_CONTRACT_FILE = _READER_FILE.parent / "contract.yaml"


def _declared() -> str:
    raw = yaml.safe_load(_CONTRACT_FILE.read_text(encoding="utf-8"))
    value = raw["github_api"]["base_url"]
    assert isinstance(value, str)
    return value


def test_default_reads_the_contract_declared_base() -> None:
    reader = HandlerMergeGroupRunReadGithub(token="t")
    assert reader._api_url == _declared().rstrip("/")


def test_an_injected_base_overrides_the_contract() -> None:
    reader = HandlerMergeGroupRunReadGithub(
        token="t", api_url="https://ghe.example.invalid/api/v3/"
    )
    assert reader._api_url == "https://ghe.example.invalid/api/v3"


def test_reader_refuses_a_contract_without_the_base(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text("github_api: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"github_api\.base_url"):
        contract_api.github_api_url(contract)


def test_slice_files_hold_no_url_literal(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text('API = "https://api.github.com"\n', encoding="utf-8")
    result = ValidatorContractsAndDi().validate([planted, _HANDLER_FILE, _READER_FILE])
    flagged = {str(i.file_path) for i in result.issues if i.rule_name == "literal_url"}
    assert flagged == {str(planted)}
