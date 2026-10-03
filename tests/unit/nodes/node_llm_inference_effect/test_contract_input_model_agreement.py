# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contract's input_model must be the class the handler imports (OMN-17104).

Before OMN-17104 the contract named ``omnibase_infra.models.llm`` (timeout ceiling
600s) while ``HandlerLlmOpenaiCompatible`` imported the node-local class (ceiling
1800s). Two same-named classes with different shapes disagreed on the ceiling.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_llm_inference_effect.handlers import (
    handler_llm_openai_compatible,
)

pytestmark = [pytest.mark.unit]

_NODE_DIR = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/nodes/node_llm_inference_effect"
)


def _contract() -> dict[str, object]:
    with (_NODE_DIR / "contract.yaml").open() as f:
        return yaml.safe_load(f)


def _timeout_ceiling(model: type) -> float:
    (constraint,) = [
        m for m in model.model_fields["timeout_seconds"].metadata if hasattr(m, "le")
    ]
    return float(constraint.le)


def test_contract_input_model_is_the_class_the_handler_imports() -> None:
    declared = _contract()["input_model"]
    resolved = getattr(importlib.import_module(declared["module"]), declared["name"])

    assert resolved is handler_llm_openai_compatible.ModelLlmInferenceRequest
    assert (
        inspect.getmodule(
            handler_llm_openai_compatible.ModelLlmInferenceRequest
        ).__name__
        == declared["module"]
    )


def test_contract_max_timeout_matches_request_timeout_ceiling() -> None:
    declared = _contract()["input_model"]
    resolved = getattr(importlib.import_module(declared["module"]), declared["name"])

    max_timeout = _contract()["handler_routing"]["max_timeout_seconds"]
    assert _timeout_ceiling(resolved) == float(max_timeout)


def test_no_second_request_class_in_shared_llm_package() -> None:
    shared = importlib.import_module("omnibase_infra.models.llm")

    assert not hasattr(shared, "ModelLlmInferenceRequest")
