# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The probe hygiene sweep reads per-repository contract roots end to end (OMN-17427).

Loads the node's real contract and the real skill mapping from disk, resolves
the handler and input model the contract declares, and drives the handler over
real ``contracts/`` directories: this repository's own checkout and a second
root built on disk. Only Linear is replaced, by a recording fake.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.enum_probe_hygiene_decision import (
    EnumProbeHygieneDecision,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_NODE_DIR = _ROOT / "src/omnibase_infra/nodes/node_in_progress_probe_hygiene_effect"
_SKILL_MAPPING = _ROOT / "src/omnibase_infra/cli/skill_mapping.yaml"
_NODE_NAME = "node_in_progress_probe_hygiene_effect"
_TICKET_WITH_CHECKS = "OMN-17427"
_TICKET_IN_OTHER_ROOT = "OMN-990001"
_TICKET_WITHOUT_CONTRACT = "OMN-990002"


class _RecordingLinear:
    last_error = ""

    def __init__(self, issues: list[dict[str, object]]) -> None:
        self._issues = issues
        self.comments: list[tuple[str, str]] = []

    async def fetch_in_progress(
        self, project: str, max_tickets: int
    ) -> list[dict[str, object]] | None:
        return self._issues

    async def has_marker(self, issue_id: str, marker: str) -> bool | None:
        return False

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True


def _issue(identifier: str) -> dict[str, object]:
    return {
        "id": f"uuid-{identifier}",
        "identifier": identifier,
        "title": f"{identifier} title",
        "description": "",
        "state": {"name": "In Progress", "type": "started"},
    }


def _load_declared(contract: dict[str, object]) -> tuple[type, type]:
    declared = contract["handler"]
    assert isinstance(declared, dict)
    handler_cls = getattr(
        importlib.import_module(declared["module"]), declared["class"]
    )
    model_module, _, model_name = declared["input_model"].rpartition(".")
    return handler_cls, getattr(importlib.import_module(model_module), model_name)


def test_contract_and_skill_mapping_declare_contract_roots() -> None:
    contract = yaml.safe_load((_NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    _, request_model = _load_declared(contract)
    operations = {op["operation"]: op for op in contract["io_operations"]}
    declared_inputs = operations["sweep_in_progress_probe_hygiene"]["input_fields"]
    assert "contract_roots" in declared_inputs
    assert "occ_repo_dir" not in declared_inputs
    assert "contract_roots" in request_model.model_fields
    assert "occ_repo_dir" not in request_model.model_fields

    skills = yaml.safe_load(_SKILL_MAPPING.read_text(encoding="utf-8"))["skills"]
    skill = next(s for s in skills if s["node_name"] == _NODE_NAME)
    args = {a["name"]: a for a in skill["args"]}
    assert args["contract-root"]["payload_field"] == "contract_roots"
    assert args["contract-root"]["arg_type"] == "string_list"
    assert "occ-repo-dir" not in args


def test_sweep_reads_this_checkout_and_a_second_root(tmp_path: Path) -> None:
    contract = yaml.safe_load((_NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    handler_cls, request_model = _load_declared(contract)

    other_root = tmp_path / "other_repo"
    (other_root / "contracts").mkdir(parents=True)
    (other_root / "contracts" / f"{_TICKET_IN_OTHER_ROOT}.yaml").write_text(
        f"ticket_id: {_TICKET_IN_OTHER_ROOT}\n"
        "dod_evidence:\n"
        "  - id: dod-1\n"
        "    checks:\n"
        "      - check_type: command\n"
        "        check_value: 'true'\n",
        encoding="utf-8",
    )
    assert (_ROOT / "contracts" / f"{_TICKET_WITH_CHECKS}.yaml").is_file()

    linear = _RecordingLinear(
        [
            _issue(_TICKET_WITH_CHECKS),
            _issue(_TICKET_IN_OTHER_ROOT),
            _issue(_TICKET_WITHOUT_CONTRACT),
        ]
    )
    request = request_model(
        correlation_id=uuid4(),
        contract_roots=(str(_ROOT), str(other_root)),
        apply=True,
        linear_retry_base_delay_seconds=0.0,
    )
    result = asyncio.run(handler_cls(linear).handle(request))

    decisions = {o.ticket: o.decision for o in result.outcomes}
    assert decisions == {
        _TICKET_WITH_CHECKS: EnumProbeHygieneDecision.HAS_PROBE,
        _TICKET_IN_OTHER_ROOT: EnumProbeHygieneDecision.HAS_PROBE,
        _TICKET_WITHOUT_CONTRACT: EnumProbeHygieneDecision.COMMENTED,
    }
    assert result.tickets_with_probe == 2
    assert result.tickets_without_probe == 1
    assert [issue_id for issue_id, _ in linear.comments] == [
        f"uuid-{_TICKET_WITHOUT_CONTRACT}"
    ]
    assert "OCC" not in linear.comments[0][1]


def test_sweep_with_no_roots_reports_unreadable_not_a_finding() -> None:
    contract = yaml.safe_load((_NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    handler_cls, request_model = _load_declared(contract)
    linear = _RecordingLinear([_issue(_TICKET_WITHOUT_CONTRACT)])
    request = request_model(
        correlation_id=uuid4(), apply=True, linear_retry_base_delay_seconds=0.0
    )
    result = asyncio.run(handler_cls(linear).handle(request))

    assert [o.decision for o in result.outcomes] == [
        EnumProbeHygieneDecision.ERROR_CONTRACT_UNREADABLE
    ]
    assert result.tickets_errored == 1
    assert result.tickets_without_probe == 0
    assert linear.comments == []
