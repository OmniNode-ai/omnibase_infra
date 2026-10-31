# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The premium-model markers are declared in the node contract (OMN-21623).

The model_overkill analyzer held its markers in code. They are now the
``config.premium_model_policy`` block of the node contract and the analyzer
takes them as an argument.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.nodes.node_waste_detection_compute.analyzers.analyzer_model_overkill import (
    analyze_model_overkill,
)
from omnibase_infra.nodes.node_waste_detection_compute.handlers import (
    handler_waste_detection,
)
from omnibase_infra.nodes.node_waste_detection_compute.handlers.handler_waste_detection import (
    HandlerWasteDetection,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteDetectionInput,
    ModelWastePremiumModelPolicy,
)

pytestmark = pytest.mark.unit

_CONTRACT = (
    Path(handler_waste_detection.__file__).resolve().parents[1] / "contract.yaml"
)
_DETECTED_AT = datetime(2026, 10, 31, tzinfo=UTC)


def _call(model_id: str) -> ModelWasteCall:
    return ModelWasteCall(
        session_id="session",
        correlation_id="corr",
        emitted_at=_DETECTED_AT,
        model_id=model_id,
        request_type="classification",
        input_tokens=300,
        output_tokens=100,
        total_tokens=400,
        cost_usd=0.03,
    )


def _write_contract(tmp_path: Path, block: dict[str, object] | None) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(
        yaml.safe_dump({"config": {"premium_model_policy": block}}), encoding="utf-8"
    )
    return path


def test_markers_come_from_the_node_contract() -> None:
    document = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    declared = tuple(document["config"]["premium_model_policy"]["markers"])
    assert ModelWastePremiumModelPolicy.from_node_contract().markers == declared


def test_the_analyzer_uses_the_markers_it_is_given() -> None:
    calls = (_call("vendor-big-1"),)
    assert analyze_model_overkill(calls, _DETECTED_AT, ("vendor-big",))
    assert not analyze_model_overkill(calls, _DETECTED_AT, ("other",))


def test_the_handler_reads_its_markers_from_an_injected_policy() -> None:
    detection_input = ModelWasteDetectionInput(
        session_id="session", calls=(_call("vendor-big-1"),), detected_at=_DETECTED_AT
    )
    default = HandlerWasteDetection().detect(detection_input)
    injected = HandlerWasteDetection(
        ModelWastePremiumModelPolicy(markers=("vendor-big",))
    ).detect(detection_input)
    assert [f.rule_id for f in default] == []
    assert [f.rule_id for f in injected] == ["model_overkill"]


@pytest.mark.parametrize("block", [None, {}, {"markers": []}])
def test_a_contract_without_markers_fails_to_load(
    tmp_path: Path, block: dict[str, object] | None
) -> None:
    with pytest.raises(ValidationError):
        ModelWastePremiumModelPolicy.from_node_contract(
            _write_contract(tmp_path, block)
        )
