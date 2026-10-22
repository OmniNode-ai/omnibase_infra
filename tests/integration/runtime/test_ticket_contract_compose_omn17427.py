# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Explicit composition preserves a valid ticket and persists runtime readback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from omnibase_core.models.ticket.model_ticket_contract import ModelTicketContract
from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.models.model_ticket_contract_compose_request import (
    ModelTicketContractComposeRequest,
)
from omnibase_infra.models.model_ticket_contract_compose_result import (
    ModelTicketContractComposeResult,
)
from tests.ci.test_compose_ticket_contract_omn17427 import contribution

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO / "src/omnibase_infra/nodes/node_ticket_contract_compose_effect/contract.yaml"
)


@pytest.mark.parametrize("invalid", [False, True])
def test_explicit_runtime_composes_real_ticket_and_persists_terminal(
    tmp_path: Path, invalid: bool
) -> None:
    contracts = tmp_path / "contracts"
    contracts.mkdir()
    root = contracts / "OMN-17427.yaml"
    original = (REPO / "contracts/OMN-17427.yaml").read_bytes()
    root.write_bytes(original)
    fragments = contracts / "OMN-17427.d"
    fragments.mkdir()
    for scope in ("beta", "alpha"):
        fragment = contribution(scope)
        if invalid and scope == "beta":
            fragment["dod_evidence"][0]["binds_ac"] = []
        (fragments / f"{scope}.yaml").write_text(
            yaml.safe_dump(fragment), encoding="utf-8"
        )
    request = ModelTicketContractComposeRequest(
        contracts_dir=str(contracts), ticket_ref="OMN-17427"
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    state = tmp_path / "runtime-state"
    runtime = RuntimeLocal(
        workflow_path=CONTRACT,
        state_root=state,
        input_path=request_path,
        backend_overrides={"event_bus": "inmemory"},
        timeout=10,
    )
    runtime.run()
    readback = json.loads((state / "workflow_result.json").read_text())
    result = ModelTicketContractComposeResult.model_validate(
        readback["terminal_payload"]
    )
    assert runtime.exit_code == readback["exit_code"] == (1 if invalid else 0)
    assert result.success is not invalid
    assert root.read_bytes() == original
    if invalid:
        assert not result.contract_yaml
        assert "unbound" in result.error_message
    else:
        validated = ModelTicketContract.from_yaml(result.contract_yaml)
        assert validated.ticket_id == "OMN-17427"
        composed = yaml.safe_load(result.contract_yaml)
        assert [item["id"] for item in composed["dod_evidence"][-2:]] == [
            "dod-alpha-1",
            "dod-beta-1",
        ]
        assert composed["requirements"][:-2] == yaml.safe_load(original)["requirements"]
