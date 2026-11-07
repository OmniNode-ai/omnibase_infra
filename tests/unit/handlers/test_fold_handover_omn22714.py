# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22714 (T8 part 2): one handover declaration per folded profile.

Golden chain: a runtime policy contract declares the effects fold's handover
once; the renderer writes ``ONEX_RUNTIME_FOLD_HANDOVERS``; the runtime gives it
to a contract the effects profile owns, and the successor group resumes from the
retired effects container's commits. Error chain: a fold declaration the runtime
cannot read, or that names a profile it does not fold, is refused, never read as
absent (which would start the folded groups at the bus default, latest).
"""

from __future__ import annotations

import json

import pytest
from aiokafka.structs import TopicPartition
from pydantic import ValidationError

from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    FOLD_HANDOVERS_ENV_VAR,
    HandlerContractSubscribePreflight,
    resolve_fold_handover,
)
from omnibase_infra.models.event_bus.model_contract_subscribe_preflight_request import (
    ModelContractSubscribePreflightRequest,
)
from omnibase_infra.models.model_fold_handover import ModelFoldHandover
from omnibase_infra.models.model_runtime_fold_policy import ModelRuntimeFoldPolicy
from omnibase_infra.runtime.auto_wiring.models.model_consume_concurrency import (
    ModelConsumeConcurrency,
)
from tests.unit.handlers.test_consumer_group_handover_from_instance_omn22714 import (
    BASE_GROUP,
    EFFECTS_INSTANCE,
    TOPIC,
    _Broker,
    _bus,
    _patched,
    _retired_group,
)

pytestmark = pytest.mark.unit

EFFECTS = ModelFoldHandover(
    from_instance_id=EFFECTS_INSTANCE,
    from_auto_offset_reset="latest",
    handover_rule="last_ran",
)


def _policy(**overrides: object) -> ModelRuntimeFoldPolicy:
    return ModelRuntimeFoldPolicy(
        folded_profiles=("effects",),
        fold_handovers={"effects": EFFECTS},
        **overrides,
    )


def _env_of(policy: ModelRuntimeFoldPolicy) -> dict[str, str]:
    rendered = policy.render_fold_env("DEV_RUNTIME_MAIN")
    return {
        "ONEX_RUNTIME_FOLDED_PROFILES": rendered["DEV_RUNTIME_MAIN_FOLDED_PROFILES"],
        FOLD_HANDOVERS_ENV_VAR: rendered["DEV_RUNTIME_MAIN_FOLD_HANDOVERS"],
    }


def test_the_policy_renders_the_declaration_the_runtime_reads() -> None:
    env = _env_of(_policy())

    assert json.loads(env[FOLD_HANDOVERS_ENV_VAR]) == {
        "effects": {
            "from_instance_id": EFFECTS_INSTANCE,
            "from_auto_offset_reset": "latest",
            "handover_rule": "last_ran",
            "fence": {"wait_seconds": 60.0, "poll_seconds": 2.0},
        }
    }


def test_a_contract_of_the_folded_profile_gets_the_fold_handover() -> None:
    env = _env_of(_policy())

    handover = resolve_fold_handover(("effects",), env)

    assert handover is not None
    assert handover.from_instance_id == EFFECTS_INSTANCE
    assert handover.handover_rule == "last_ran"


def test_a_contract_outside_the_folded_profile_gets_none() -> None:
    env = _env_of(_policy())

    assert resolve_fold_handover(("main",), env) is None
    assert resolve_fold_handover((), env) is None
    assert resolve_fold_handover(("effects",), {}) is None


def test_error_a_policy_cannot_declare_a_handover_for_a_profile_it_does_not_fold() -> (
    None
):
    with pytest.raises(ValidationError, match="does not fold"):
        ModelRuntimeFoldPolicy(
            folded_profiles=("effects",), fold_handovers={"worker": EFFECTS}
        )


def test_error_a_declaration_without_a_rule_is_refused_never_defaulted() -> None:
    with pytest.raises(ValidationError, match="handover_rule"):
        ModelFoldHandover.model_validate(
            {"from_instance_id": "runtime-effects", "from_auto_offset_reset": "latest"}
        )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{not json", "not valid JSON"),
        ('["effects"]', "must be a JSON object"),
        ('{"effects": {"from_instance_id": "x"}}', "handover_rule"),
    ],
)
def test_error_an_unreadable_fold_declaration_is_refused_never_read_as_absent(
    raw: str, message: str
) -> None:
    env = {"ONEX_RUNTIME_FOLDED_PROFILES": "effects", FOLD_HANDOVERS_ENV_VAR: raw}

    with pytest.raises(ValueError, match=message):
        resolve_fold_handover(("effects",), env)


def test_error_a_declaration_for_an_unfolded_profile_is_refused() -> None:
    env = _env_of(_policy())
    env["ONEX_RUNTIME_FOLDED_PROFILES"] = "tenant-projection"

    with pytest.raises(ValueError, match="does not fold"):
        resolve_fold_handover(("effects",), env)


@pytest.mark.asyncio
async def test_golden_the_folded_contract_resumes_where_the_effects_container_stopped() -> (
    None
):
    """Policy to env to resolver to preflight to the runtime group's commit."""
    handover = resolve_fold_handover(("effects",), _env_of(_policy()))
    assert handover is not None
    broker = _Broker({0, 1, 2})
    bus = _bus()
    physical = bus.resolve_bus_binding(TOPIC).physical_topic
    retired = _retired_group(bus)
    broker.committed[retired] = {
        TopicPartition(physical, 0): 12,
        TopicPartition(physical, 1): 99,
        TopicPartition(physical, 2): 5,
    }
    preflight = HandlerContractSubscribePreflight(
        contract_name="node_effect_demo",
        concurrency=ModelConsumeConcurrency(),
        handover=handover,
        event_bus=bus,
        environment="dev",
        consumer_group=BASE_GROUP,
    )

    with _patched(broker):
        result = await preflight.handle(
            ModelContractSubscribePreflightRequest(topic=TOPIC)
        )

    assert result.handover is not None
    assert result.handover.committed == {0: 12, 1: 99, 2: 5}
    assert result.handover.from_group == retired
