# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The shipped LLM inference contract resolves to one request class (OMN-17104).

Loads the real ``contract.yaml`` from disk and imports every module it names,
the way contract-driven wiring does, then builds a request from the resolved
``input_model`` at the contract's declared timeout ceiling. No services needed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.integration]

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_llm_inference_effect/contract.yaml"
)


def _load_contract() -> dict[str, object]:
    with _CONTRACT.open() as f:
        return yaml.safe_load(f)


def test_every_module_the_contract_names_imports() -> None:
    contract = _load_contract()
    declared = [contract["input_model"], contract["output_model"]]
    for entry in contract["handler_routing"]["handlers"]:
        declared.append(entry["handler"])
        if "event_model" in entry:
            declared.append(entry["event_model"])

    for ref in declared:
        module = importlib.import_module(ref["module"])
        assert hasattr(module, ref["name"]), f"{ref['module']} lacks {ref['name']}"


def test_contract_input_model_accepts_the_declared_timeout_ceiling() -> None:
    contract = _load_contract()
    declared = contract["input_model"]
    request_cls = getattr(importlib.import_module(declared["module"]), declared["name"])
    ceiling = float(contract["handler_routing"]["max_timeout_seconds"])

    request = request_cls(
        base_url="http://llm.invalid:8000",
        model="contract-resolution-probe",
        prompt="ping",
        operation_type="completion",
        timeout_seconds=ceiling,
    )

    assert request.timeout_seconds == ceiling
    with pytest.raises(ValueError):
        request_cls(
            base_url="http://llm.invalid:8000",
            model="contract-resolution-probe",
            prompt="ping",
            operation_type="completion",
            timeout_seconds=ceiling + 1.0,
        )
