# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hostile-case coverage for the overlay-only Bifrost lane contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

_ENDPOINT = "http://192.168.86.201:8000/v1/chat/completions"


def _binding(backend_id: str = "local-coder", **overrides: object) -> dict[str, object]:
    binding: dict[str, object] = {
        "backend_id": backend_id,
        "endpoint_url": _ENDPOINT,
        "served_model_id": "qwen3.8",
        "parameter_count": "27B",
        "context_window": 122_880,
        "max_tokens": 65_536,
        "timeout_ms": 300_000,
    }
    binding.update(overrides)
    return binding


def _overlay(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": "bifrost_lane_overlay.v2",
        "lane": "dev",
        "backends": [_binding(), _binding("local-heavy-reasoning")],
    }
    data.update(overrides)
    return data


@pytest.mark.unit
def test_valid_overlay_has_exact_qwen_lab_bindings() -> None:
    overlay = ModelBifrostLaneOverlay.model_validate(_overlay())

    assert [binding.backend_key for binding in overlay.backends] == [
        "local-coder",
        "local-heavy-reasoning",
    ]
    assert {binding.advertised_model for binding in overlay.backends} == {"qwen3.8"}
    assert {binding.context_window for binding in overlay.backends} == {122_880}
    assert overlay.model_dump(by_alias=True)["backends"][0] == _binding()


@pytest.mark.unit
@pytest.mark.parametrize(
    "endpoint_url",
    [
        "http://192.168.86.201:8001/v1/chat/completions",
        "http://192.168.86.201:8000/v1",
        "http://192.168.86.201:8000/v1/chat/completions?model=qwen3.8",
        "http://user@192.168.86.201:8000/v1/chat/completions",
        "http://localhost:8000/v1/chat/completions",
        "http://192.168.86.200:8000/v1/chat/completions",
        "http://192.168.86.202:8000/v1/chat/completions",
        "https://192.168.86.201:8000/v1/chat/completions",
    ],
)
def test_incomplete_or_retired_endpoint_is_rejected(endpoint_url: str) -> None:
    with pytest.raises(ValidationError, match="endpoint_url"):
        ModelBifrostLaneOverlay.model_validate(
            _overlay(
                backends=[
                    _binding(endpoint_url=endpoint_url),
                    _binding("local-heavy-reasoning"),
                ]
            )
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("served_model_id", "qwen3.8-27b"),
        ("parameter_count", "27b"),
        ("context_window", 32_768),
        ("max_tokens", 122_881),
        ("endpoint_url_env", "BIFROST_LOCAL_CODER_ENDPOINT_URL"),
        ("secret_ref", "infisical://local-coder"),
    ],
)
def test_model_metadata_and_env_transport_are_rejected(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        ModelBifrostLaneOverlay.model_validate(
            _overlay(
                backends=[_binding(**{field: value}), _binding("local-heavy-reasoning")]
            )
        )


@pytest.mark.unit
def test_unknown_duplicate_or_missing_backend_is_rejected() -> None:
    for backends in (
        [_binding(), _binding()],
        [_binding(), _binding("local-reasoner")],
        [_binding()],
    ):
        with pytest.raises(ValidationError):
            ModelBifrostLaneOverlay.model_validate(_overlay(backends=backends))
