# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hostile-case coverage for the overlay-only Bifrost lane contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    ACTIVE_BACKEND_KEYS,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

_QWEN_ENDPOINT = "http://192.168.86.201:8000/v1/chat/completions"
_DS_V4_ENDPOINT = "http://192.168.86.200:8101/v1/chat/completions"

# Per-backend authorized shape, mirroring the live 2026-08-28 readback recorded in
# OMN-16833: .201:8000 serves ``qwen3.8`` (max_model_len 122880) and .200:8101 serves
# ``deepseek-v4-flash`` (context_length 131072).
_SHAPES: dict[str, dict[str, object]] = {
    "local-coder": {
        "endpoint_url": _QWEN_ENDPOINT,
        "served_model_id": "qwen3.8",
        "parameter_count": "27B",
        "context_window": 122_880,
        "max_tokens": 65_536,
        "timeout_ms": 300_000,
    },
    "local-heavy-reasoning": {
        "endpoint_url": _QWEN_ENDPOINT,
        "served_model_id": "qwen3.8",
        "parameter_count": "27B",
        "context_window": 122_880,
        "max_tokens": 65_536,
        "timeout_ms": 300_000,
    },
    "local-ds-v4-flash": {
        "endpoint_url": _DS_V4_ENDPOINT,
        "served_model_id": "deepseek-v4-flash",
        "parameter_count": "284B",
        "context_window": 131_072,
        "max_tokens": 65_536,
        "timeout_ms": 300_000,
    },
}


def _binding(backend_id: str = "local-coder", **overrides: object) -> dict[str, object]:
    binding: dict[str, object] = {"backend_id": backend_id, **_SHAPES[backend_id]}
    binding.update(overrides)
    return binding


def _overlay(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": "bifrost_lane_overlay.v2",
        "lane": "dev",
        "backends": [_binding(backend_id) for backend_id in _SHAPES],
    }
    data.update(overrides)
    return data


@pytest.mark.unit
def test_valid_overlay_has_exact_authorized_lab_bindings() -> None:
    overlay = ModelBifrostLaneOverlay.model_validate(_overlay())

    assert [binding.backend_key for binding in overlay.backends] == [
        "local-coder",
        "local-heavy-reasoning",
        "local-ds-v4-flash",
    ]
    assert {binding.advertised_model for binding in overlay.backends} == {
        "qwen3.8",
        "deepseek-v4-flash",
    }
    assert overlay.model_dump(by_alias=True)["backends"][0] == _binding()


@pytest.mark.unit
def test_active_backend_keys_cover_every_bindable_local_backend() -> None:
    """OMN-16833: the DS-V4-Flash rung is a required member, not optional.

    ``escalation``'s large-window (65536) local rung is ``local-ds-v4-flash``.  If it
    is absent from the required set an overlay can omit it and the lane silently
    degrades to the metered ceiling, which is exactly the OMN-16833 defect.
    """
    assert (
        frozenset({"local-coder", "local-heavy-reasoning", "local-ds-v4-flash"})
        == ACTIVE_BACKEND_KEYS
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend_id", "endpoint_url"),
    [
        # local-coder / local-heavy-reasoning are pinned to .201:8000.
        ("local-coder", "http://192.168.86.201:8001/v1/chat/completions"),
        ("local-coder", "http://192.168.86.201:8000/v1"),
        ("local-coder", "http://192.168.86.201:8000/v1/chat/completions?model=qwen3.8"),
        ("local-coder", "http://user@192.168.86.201:8000/v1/chat/completions"),
        ("local-coder", "http://localhost:8000/v1/chat/completions"),
        ("local-coder", "http://192.168.86.200:8000/v1/chat/completions"),
        ("local-coder", "http://192.168.86.202:8000/v1/chat/completions"),
        ("local-coder", "https://192.168.86.201:8000/v1/chat/completions"),
        # local-ds-v4-flash is pinned to .200:8101 — the qwen host/port is NOT
        # interchangeable, and a bare base still fails closed (OMN-12815).
        ("local-ds-v4-flash", "http://192.168.86.201:8000/v1/chat/completions"),
        ("local-ds-v4-flash", "http://192.168.86.200:8101/v1"),
        ("local-ds-v4-flash", "http://192.168.86.200:8102/v1/chat/completions"),
        ("local-ds-v4-flash", "https://192.168.86.200:8101/v1/chat/completions"),
    ],
)
def test_incomplete_or_unauthorized_endpoint_is_rejected(
    backend_id: str, endpoint_url: str
) -> None:
    with pytest.raises(ValidationError, match="endpoint_url"):
        ModelBifrostLaneOverlay.model_validate(
            _overlay(
                backends=[
                    _binding(
                        bid,
                        **({"endpoint_url": endpoint_url} if bid == backend_id else {}),
                    )
                    for bid in _SHAPES
                ]
            )
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend_id", "field", "value"),
    [
        ("local-coder", "served_model_id", "qwen3.8-27b"),
        ("local-coder", "parameter_count", "27b"),
        ("local-coder", "context_window", 32_768),
        ("local-coder", "max_tokens", 122_881),
        ("local-coder", "endpoint_url_env", "BIFROST_LOCAL_CODER_ENDPOINT_URL"),
        ("local-coder", "secret_ref", "infisical://local-coder"),
        # OMN-16833: the DS-V4-Flash served id is ``deepseek-v4-flash``, NOT the
        # ``ds-v4-flash`` label the repo contract carried before this ticket.  A
        # mismatch here is the OMN-16419 silent-misattribution class.
        ("local-ds-v4-flash", "served_model_id", "ds-v4-flash"),
        ("local-ds-v4-flash", "context_window", 65_536),
        ("local-ds-v4-flash", "max_tokens", 131_073),
        (
            "local-ds-v4-flash",
            "endpoint_url_env",
            "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL",
        ),
    ],
)
def test_model_metadata_and_env_transport_are_rejected(
    backend_id: str, field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        ModelBifrostLaneOverlay.model_validate(
            _overlay(
                backends=[
                    _binding(bid, **({field: value} if bid == backend_id else {}))
                    for bid in _SHAPES
                ]
            )
        )


@pytest.mark.unit
def test_unknown_duplicate_or_missing_backend_is_rejected() -> None:
    complete = [_binding(bid) for bid in _SHAPES]
    for backends in (
        [*complete, _binding()],  # duplicate local-coder
        [*complete, {**_binding(), "backend_id": "local-reasoner"}],  # unknown
        complete[:-1],  # missing local-ds-v4-flash — the OMN-16833 defect
        [_binding()],
    ):
        with pytest.raises(ValidationError):
            ModelBifrostLaneOverlay.model_validate(_overlay(backends=backends))
