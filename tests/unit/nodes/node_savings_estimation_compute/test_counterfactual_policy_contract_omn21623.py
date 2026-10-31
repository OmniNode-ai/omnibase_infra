# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""The counterfactual model policy is declared in the node contract (OMN-21623).

HandlerSavingsCorrelation held a model-name map and a fallback model in code.
They are now the ``config.counterfactual_policy`` block of the node contract;
every field is required, so a contract missing one fails to load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.nodes.node_savings_estimation_compute.handlers import (
    handler_savings_correlation,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models import (
    ModelEffectivenessEntry,
    ModelSavingsCounterfactualPolicy,
    ModelSavingsEstimationInput,
)

pytestmark = pytest.mark.unit

_CONTRACT = (
    Path(handler_savings_correlation.__file__).resolve().parents[1] / "contract.yaml"
)


def _policy_block() -> dict[str, Any]:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    block = document["config"]["counterfactual_policy"]
    assert isinstance(block, dict)
    return block


def _write_contract(tmp_path: Path, block: dict[str, object] | None) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(
        yaml.safe_dump({"config": {"counterfactual_policy": block}}), encoding="utf-8"
    )
    return path


def test_policy_loads_every_rule_from_the_node_contract() -> None:
    policy = ModelSavingsCounterfactualPolicy.from_node_contract()
    block = _policy_block()
    assert policy.fallback_model == block["fallback_model"]
    assert [(rule.match, rule.counterfactual_model) for rule in policy.rules] == [
        (r["match"], r["counterfactual_model"]) for r in block["rules"]
    ]


def test_resolve_uses_the_first_matching_rule_and_falls_back_to_itself(
    tmp_path: Path,
) -> None:
    path = _write_contract(
        tmp_path,
        {
            "fallback_model": "fallback-model",
            "rules": [
                {"match": "alpha", "counterfactual_model": "first"},
                {"match": "alpha-beta", "counterfactual_model": "second"},
            ],
        },
    )
    policy = ModelSavingsCounterfactualPolicy.from_node_contract(path)
    assert policy.resolve("Alpha-Beta-1") == "first"
    assert policy.resolve("unmatched") == "unmatched"


@pytest.mark.parametrize("missing", ["fallback_model", "rules"])
def test_a_contract_missing_a_field_fails_to_load(tmp_path: Path, missing: str) -> None:
    block = dict(_policy_block())
    del block[missing]
    with pytest.raises(ValidationError, match=missing):
        ModelSavingsCounterfactualPolicy.from_node_contract(
            _write_contract(tmp_path, block)
        )


def test_a_contract_without_the_block_fails_to_load(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ModelSavingsCounterfactualPolicy.from_node_contract(
            _write_contract(tmp_path, None)
        )


def test_estimation_input_requires_the_actual_model() -> None:
    with pytest.raises(ValidationError, match="actual_model_id"):
        ModelSavingsEstimationInput.model_validate(
            {
                "session_id": "s",
                "effectiveness_entries": (
                    ModelEffectivenessEntry(
                        utilization_score=0.5, patterns_count=1, tokens_saved=10
                    ),
                ),
            }
        )
