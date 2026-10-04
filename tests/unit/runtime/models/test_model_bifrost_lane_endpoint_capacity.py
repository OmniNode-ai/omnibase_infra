# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shape and arithmetic of the lane overlay's per-endpoint capacity (OMN-20490)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.delegation.model_bifrost_lane_endpoint_capacity import (
    ModelBifrostLaneEndpointCapacity,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

pytestmark = pytest.mark.unit

_ENDPOINT = "10.0.0.9:8000"
_URL = "http://10.0.0.9:8000/v1/chat/completions"
_CAPACITY: dict[str, object] = {
    "endpoint": _ENDPOINT,
    "pool_tokens": 1_200,
    "parallel_slots": 3,
    "prompt_headroom_tokens": 100,
}
_BACKEND: dict[str, object] = {
    "backend_id": "control-chat",
    "endpoint_url": _URL,
    "served_model_id": "control-model",
    "parameter_count": "1B",
    "context_window": 1_200,
    "max_tokens": 300,
    "timeout_ms": 1_000,
}


def _overlay(**overrides: object) -> dict[str, object]:
    overlay: dict[str, object] = {
        "schema_version": "bifrost_lane_overlay.v3",
        "lane": "control",
        "locale": "lab",
        "backends": [_BACKEND],
    }
    overlay.update(overrides)
    return overlay


def test_share_and_output_ceiling_are_derived_from_the_declared_fields() -> None:
    capacity = ModelBifrostLaneEndpointCapacity.model_validate(_CAPACITY)

    assert capacity.per_slot_tokens == 400
    assert capacity.max_output_tokens == 300


def test_serves_matches_host_and_port_of_an_endpoint_url() -> None:
    capacity = ModelBifrostLaneEndpointCapacity.model_validate(_CAPACITY)

    assert capacity.serves(_URL)
    assert not capacity.serves("http://10.0.0.9:8001/v1/chat/completions")
    assert not capacity.serves("http://10.0.0.8:8000/v1/chat/completions")


def test_the_scheme_default_port_is_the_endpoint_port() -> None:
    capacity = ModelBifrostLaneEndpointCapacity.model_validate(
        {**_CAPACITY, "endpoint": "models.example.invalid:443"}
    )

    assert capacity.serves("https://models.example.invalid/v1/chat/completions")


@pytest.mark.parametrize(
    "override",
    [
        {"extra_field": 1},
        {"pool_tokens": 0},
        {"parallel_slots": 0},
        {"prompt_headroom_tokens": 0},
        {"endpoint": "10.0.0.9"},
        {"endpoint": "10.0.0.9:port"},
        {"endpoint": ":8000"},
        {"pool_tokens": 2, "parallel_slots": 3},
        {"prompt_headroom_tokens": 400},
    ],
)
def test_a_malformed_capacity_is_refused(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelBifrostLaneEndpointCapacity.model_validate({**_CAPACITY, **override})


def test_overlay_accepts_a_capacity_block_and_defaults_to_none() -> None:
    declared = ModelBifrostLaneOverlay.model_validate(
        _overlay(endpoint_capacity=[_CAPACITY])
    )
    undeclared = ModelBifrostLaneOverlay.model_validate(_overlay())

    assert [capacity.endpoint for capacity in declared.endpoint_capacity] == [_ENDPOINT]
    assert undeclared.endpoint_capacity == ()


def test_overlay_still_rejects_an_unknown_top_level_key() -> None:
    with pytest.raises(ValidationError, match="endpoint_capcity"):
        ModelBifrostLaneOverlay.model_validate(_overlay(endpoint_capcity=[_CAPACITY]))


def test_overlay_refuses_one_endpoint_declared_twice() -> None:
    with pytest.raises(ValidationError, match="must not declare an endpoint twice"):
        ModelBifrostLaneOverlay.model_validate(
            _overlay(endpoint_capacity=[_CAPACITY, _CAPACITY])
        )
