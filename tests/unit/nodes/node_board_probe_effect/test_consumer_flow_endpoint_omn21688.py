# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The consumer-flow projection API base comes from the node contract (OMN-21688)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.validation.validator_contracts_and_di import (
    ValidatorContractsAndDi,
)
from omnibase_infra.nodes.node_board_probe_effect import contract_topics
from omnibase_infra.nodes.node_board_probe_effect.contract_topics import (
    consumer_flow_projection_api_base_url,
)
from omnibase_infra.nodes.node_board_probe_effect.models import (
    ModelConsumerFlowRequest,
)

pytestmark = pytest.mark.unit

_NODE_DIR = Path(contract_topics.__file__).resolve().parent
_SLICE_FILES = (
    _NODE_DIR / "contract_topics.py",
    _NODE_DIR / "handlers" / "_consumer_flow_constants.py",
    _NODE_DIR / "models" / "model_consumer_flow_request.py",
)


def _declared() -> str:
    raw = yaml.safe_load((_NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    value = raw["consumer_flow_endpoints"]["projection_api_base_url"]
    assert isinstance(value, str)
    return value


def test_request_default_is_the_contract_declared_base_url() -> None:
    assert ModelConsumerFlowRequest(subject_lane="dev").base_url == _declared()


def test_request_keeps_an_injected_base_url() -> None:
    request = ModelConsumerFlowRequest(
        subject_lane="dev", base_url="http://projection.example.invalid:1"
    )
    assert request.base_url == "http://projection.example.invalid:1"


def test_reader_follows_the_contract_it_is_given(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "consumer_flow_endpoints:\n"
        "  projection_api_base_url: http://other.example.invalid:9\n",
        encoding="utf-8",
    )
    assert (
        consumer_flow_projection_api_base_url(contract)
        == "http://other.example.invalid:9"
    )


@pytest.mark.parametrize(
    "body",
    ["other: {}\n", "consumer_flow_endpoints:\n  projection_api_base_url: ''\n"],
)
def test_reader_refuses_a_contract_without_the_base_url(
    tmp_path: Path, body: str
) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="projection_api_base_url"):
        consumer_flow_projection_api_base_url(contract)


def _url_findings(paths: list[Path]) -> list[str]:
    result = ValidatorContractsAndDi().validate(paths)
    return [str(i.file_path) for i in result.issues if i.rule_name == "literal_url"]


def test_the_check_reports_a_planted_url(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text('BASE = "http://host.docker.internal:3002"\n', encoding="utf-8")
    assert _url_findings([planted]) == [str(planted)]


@pytest.mark.parametrize("path", _SLICE_FILES, ids=lambda p: p.name)
def test_slice_file_holds_no_url_literal(path: Path) -> None:
    assert path.is_file()
    assert _url_findings([path]) == []
