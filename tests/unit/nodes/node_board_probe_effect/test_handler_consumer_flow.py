# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""C28 recorded observations grade identically through the node boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_consumer_flow import (
    HandlerConsumerFlow,
    grade_consumer_flow,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_observation import (
    ModelConsumerFlowObservation,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)
from scripts.ci import c28_consumer_flow_probe as script

pytestmark = pytest.mark.unit
FIXTURE = Path(__file__).parent / "fixtures/consumer_flow_recorded.json"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("recorded", "PASS"),
        ("stalled", "FAIL"),
        ("unreachable_group", "FAIL"),
        ("negative", "FAIL"),
        ("boot", "FAIL"),
        ("unreadable", "INDETERMINATE"),
        ("empty", "INDETERMINATE"),
    ],
)
def test_handler_table(case: str, expected: str) -> None:
    obs = json.loads(FIXTURE.read_text())
    if case == "stalled":
        for sample in obs["kinds"]["samples"]:
            sample[1].update(messages_out=0, flow_state="STALLED")
    elif case == "unreachable_group":
        obs["cursor"]["walked_groups"] = []
    elif case == "negative":
        obs["negative"]["mutations"]["event_bus"]["restored"] = False
    elif case == "boot":
        obs["boot"]["injection"].update(dlq_copies=0, boundary_lines=0)
    elif case == "empty":
        obs = {}
    observation = ModelConsumerFlowObservation(
        read_ok=case != "unreadable",
        read_error="connection refused" if case == "unreadable" else "",
        **obs,
    )
    request = ModelConsumerFlowRequest(subject_lane="dev")

    class Target:
        async def observe(
            self, received: ModelConsumerFlowRequest
        ) -> ModelConsumerFlowObservation:
            assert received == request
            return observation

    result = asyncio.run(HandlerConsumerFlow(target=Target()).handle(request))
    assert result == grade_consumer_flow(request, observation)
    assert result.outcome == expected
    assert result.subject == "dev"
    assert result.check_id == "consumer_flow"
    assert result.surface_class == "lab_hardware"
    assert result.status == ("success" if expected == "PASS" else "failure")
    assert result.as_lab_proof_check_result().passed == (expected == "PASS")
    if expected == "FAIL":
        assert result.reasons == tuple(script.grade(obs).failures)
    if case == "unreadable":
        assert "connection refused" in result.reasons[0]


def test_recorded_fixture_is_the_scripts_own_green_shape() -> None:
    import runpy

    source = (
        Path(__file__).resolve().parents[4] / "tests/ci/test_c28_consumer_flow_probe.py"
    )
    assert json.loads(FIXTURE.read_text()) == runpy.run_path(str(source))["_green"]()


@pytest.mark.parametrize(
    "model", [ModelConsumerFlowRequest, ModelConsumerFlowObservation]
)
def test_models_forbid_unknown_fields_and_are_frozen(model: type) -> None:
    kwargs = (
        {"subject_lane": "dev"}
        if model is ModelConsumerFlowRequest
        else {"read_ok": True}
    )
    with pytest.raises(ValidationError):
        model(**kwargs, typo=True)
    instance = model(**kwargs)
    with pytest.raises(ValidationError):
        setattr(instance, next(iter(kwargs)), None)


@pytest.mark.parametrize("read_ok", [True, False])
def test_handler_writes_self_contained_record_even_when_unreadable(
    tmp_path: Path, read_ok: bool
) -> None:
    record = tmp_path / "c28.json"
    request = ModelConsumerFlowRequest(subject_lane="dev", record=record)
    observation = ModelConsumerFlowObservation(
        read_ok=read_ok,
        read_error="" if read_ok else "unreachable",
        **json.loads(FIXTURE.read_text()),
    )

    class Target:
        async def observe(
            self, request: ModelConsumerFlowRequest
        ) -> ModelConsumerFlowObservation:
            return observation

    result = asyncio.run(HandlerConsumerFlow(target=Target()).handle(request))
    body = json.loads(record.read_text())
    assert body["result"] == result.model_dump(mode="json")
    assert body["observation"] == observation.model_dump(mode="json")
    assert body["criterion"] == "C28"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("kinds", "samples"), []),
        (("cursor", "terminated"), False),
        (("cursor", "second_page_differs"), False),
        (("cursor", "live_groups"), []),
        (
            ("cursor", "pages"),
            [{"row_count": 500, "row_limit": 500, "next_cursor": None}],
        ),
        (("negative", "clean", "returncode"), 1),
        (("negative", "clean", "outcomes"), {}),
        (("negative", "mutations", "raw_event_projection", "applied"), False),
        (("negative", "mutations", "event_bus", "outcomes"), {}),
        (("boot", "applied_hwm_after"), 0),
        (("boot", "natural"), {}),
        (("boot", "natural", "omninode-runtime", "natural"), 1),
        (("boot", "injection", "offset"), None),
        (("boot", "injection", "validation_errors_after"), 0),
        (("boot", "injection", "boundary_lines"), 0),
        (("boot", "injection", "dlq_copies"), 0),
    ],
)
def test_each_clause_preserves_script_failure_reasons(
    path: tuple[str, ...], value: object
) -> None:
    obs = json.loads(FIXTURE.read_text())
    target = obs
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    result = grade_consumer_flow(
        ModelConsumerFlowRequest(subject_lane="dev"),
        ModelConsumerFlowObservation(read_ok=True, **obs),
    )
    assert result.outcome == "FAIL"
    assert result.reasons == tuple(script.grade(obs).failures)
