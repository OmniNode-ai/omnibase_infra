# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Done-write binding checks the packaged contract before calling its gate."""

from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import yaml

from omnibase_core.models.ticket.model_done_write_receipt_request import (
    ModelDoneWriteReceiptRequest,
)
from omnibase_core.nodes.node_done_write_receipt_gate_compute import (
    handler_done_write_receipt_gate as core,
)
from omnibase_infra.handlers.done_write_receipt_guard import (
    DONE_WRITE_RECEIPT_GATE_NODE,
    resolve_done_write_receipt_gate,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
    handler_evidence_autoclose_sweep as sweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models import (
    model_check_result_row as fields,
)
from omnibase_infra.protocols.protocol_done_write_receipt_gate import (
    ProtocolDoneWriteReceiptGate,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("allowed", [True, False])
def test_pinned_gate_matches_core_decisions(allowed: bool) -> None:
    gate = resolve_done_write_receipt_gate()
    description = "## Acceptance Criteria\n- **AC1**: the test exercises the handler\n"
    verdict: dict[str, object] = {
        "status": "verified",
        "total_checks": 1,
        "verified_count": 1,
        "failed_count": 0,
        "non_probative_count": 0,
        "checks": [
            {
                "evidence_id": "binding-test",
                "status": "verified",
                "proof_class": "behavior",
                "binds_ac": ["AC1"] if allowed else [],
            }
        ],
    }
    request = ModelDoneWriteReceiptRequest(
        ticket_id="OMN-17427", description=description, verdict=verdict
    )
    decision = gate.decide(request)
    assert decision == core.evaluate_done_write_receipt(
        ticket_id=request.ticket_id, description=description, verdict=verdict
    )
    assert decision.allowed is allowed
    assert isinstance(gate, ProtocolDoneWriteReceiptGate)
    assert resolve_done_write_receipt_gate() is gate


@pytest.mark.parametrize(
    "name",
    [
        "CHECK_STATUS_VERIFIED",
        "MAX_UNCOVERED_LISTED",
        "dod_verify_verdict_key",
        "dod_verify_checks_key",
        "check_id_key",
        "check_status_key",
        "check_proof_class_key",
        "check_binds_ac_key",
    ],
)
def test_sweep_verdict_format_matches_the_pin(name: str) -> None:
    assert getattr(fields, name) == getattr(core, name)


@pytest.mark.parametrize(
    "name",
    [
        name
        for name, member in inspect.getmembers(
            ProtocolDoneWriteReceiptGate, inspect.isfunction
        )
        if not name.startswith("_") and name != "decide"
    ],
)
def test_protocol_function_signatures_match_the_pin(name: str) -> None:
    expected = inspect.signature(getattr(core, name))
    actual = inspect.signature(getattr(ProtocolDoneWriteReceiptGate, name))
    assert actual.replace(parameters=list(actual.parameters.values())[1:]) == expected


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("valid", "allowed"),
        ("input_model", "input_model"),
        ("output_model", "output_model"),
        ("name", "name"),
        ("missing_class", "class"),
        ("missing_export", "as_int"),
        ("absolute_handler", "node-relative"),
        ("missing_handle", "handle"),
    ],
)
def test_planted_contract_is_checked_before_binding(
    tmp_path: Path, damage: str, message: str
) -> None:
    """A fresh interpreter finds the planted package first, without cached modules."""
    package = tmp_path / "omnibase_core"
    package.mkdir()
    # Extend the package to the installed public models, shadowing just this node.
    (package / "__init__.py").write_text(
        "from pkgutil import extend_path\n"
        "from importlib.metadata import version\n"
        "__path__ = extend_path(__path__, __name__)\n"
        "__version__ = version('omnibase-core')\n"
    )
    nodes = package / "nodes"
    nodes.mkdir()
    (nodes / "__init__.py").write_text(
        "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
    )
    node = nodes / DONE_WRITE_RECEIPT_GATE_NODE
    shutil.copytree(
        Path(core.__file__).parent, node, ignore=shutil.ignore_patterns("__pycache__")
    )
    contract_path = node / "contract.yaml"
    contract = yaml.safe_load(contract_path.read_text())
    module = node / "handler_done_write_receipt_gate.py"
    if damage in {"input_model", "output_model", "name"}:
        contract[damage] = "wrong"
    elif damage == "missing_class":
        contract["handler_routing"]["default_handler"] = (
            "handler_done_write_receipt_gate:MissingClass"
        )
    elif damage == "missing_export":
        module.write_text(module.read_text() + "\n__all__.remove('as_int')\n")
    elif damage == "absolute_handler":
        contract["handler_routing"]["default_handler"] = (
            ".handler_done_write_receipt_gate:HandlerDoneWriteReceiptGate"
        )
    elif damage == "missing_handle":
        module.write_text(
            module.read_text() + "\ndel HandlerDoneWriteReceiptGate.handle\n"
        )
    contract_path.write_text(yaml.safe_dump(contract))
    probe = """
import json, sys
sys.path.insert(0, sys.argv[1])
from omnibase_infra.handlers.done_write_receipt_guard import (
    DoneWriteGateResolutionError, resolve_done_write_receipt_gate,
)
from omnibase_core.models.ticket.model_done_write_receipt_request import ModelDoneWriteReceiptRequest
try:
    gate = resolve_done_write_receipt_gate()
except DoneWriteGateResolutionError as exc:
    print(json.dumps({"error": str(exc)}))
else:
    print(json.dumps({"allowed": gate.decide(ModelDoneWriteReceiptRequest(ticket_id="OMN-17427", description="")).allowed}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    if damage == "valid":
        assert observed == {"allowed": False}
    else:
        assert message in observed["error"]


def test_sweep_helpers_consult_the_injected_gate() -> None:
    gate = Mock(spec=ProtocolDoneWriteReceiptGate)
    gate.live_acceptance_criteria_items.return_value = ["AC1"]
    gate.canonical_ac_label.side_effect = lambda item: item
    gate.check_records.return_value = ({"status": "verified"},)
    gate.check_status.return_value = "verified"
    binding = cast("ProtocolDoneWriteReceiptGate", gate)
    handler = sweep.HandlerEvidenceAutocloseSweep(gate=binding)
    assert handler._gate is binding
    assert sweep._coverage_corpus_counts(
        {"checks": [{"binds_ac": ["AC1"], "status": "verified"}]},
        "a body",
        gate=binding,
    ) == (1, 0)
    assert sweep._withheld_check_ids({}, gate=binding) == ()
    gate.live_acceptance_criteria_items.assert_called_once_with("a body")
    gate.check_records.assert_called_once_with({})
