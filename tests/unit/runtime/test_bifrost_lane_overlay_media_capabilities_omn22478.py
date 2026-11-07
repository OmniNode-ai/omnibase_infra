# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22756: a lane overlay declares typed media capabilities; a node asks for one.

A node asks the overlay for ``tts``, ``stt``, ``embed``, ``decide`` or ``video``
and gets the serving binding and its URL back, never naming a host or port. The
binding checks that a capability and its endpoint path agree, and the overlay
keeps one serving backend per capability.
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_bifrost_lane_media_capability import (
    EnumBifrostLaneMediaCapability,
)
from omnibase_infra.models.delegation import (
    model_bifrost_lane_media_capability_resolver as resolver_module,
)
from omnibase_infra.models.delegation.model_bifrost_lane_media_capability_error import (
    BifrostLaneMediaCapabilityAmbiguousError,
    BifrostLaneMediaCapabilityNotServingError,
    BifrostLaneMediaCapabilityShapeError,
    BifrostLaneMediaCapabilityUnknownError,
)
from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    ModelBifrostLaneBackendBinding,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

pytestmark = pytest.mark.unit

# RFC 5737 documentation-range address, never a lab host.
_HOST = "http://192.0.2.30"
_PORTS = {"tts": 9001, "stt": 9002, "embed": 9003, "decide": 9004, "video": 9005}
_PATHS = {
    "tts": "/v1/audio/speech",
    "stt": "/v1/audio/transcriptions",
    "embed": "/v1/embeddings",
    "decide": "/v1/chat/completions",
    "video": "/v1/video/generations",
}


def _binding(capability: str, **overrides: object) -> dict[str, object]:
    binding: dict[str, object] = {
        "backend_id": f"media-{capability}",
        "endpoint_url": f"{_HOST}:{_PORTS[capability]}{_PATHS[capability]}",
        "served_model_id": f"{capability}-model",
        "parameter_count": "1B",
        "context_window": 8192,
        "max_tokens": 1024,
        "timeout_ms": 60_000,
        "provider": "local",
        "tier": "local",
        "credential": {"kind": "none"},
        "media_capability": capability,
    }
    binding.update(overrides)
    return binding


def _overlay(*backends: dict[str, object]) -> ModelBifrostLaneOverlay:
    return ModelBifrostLaneOverlay.model_validate(
        {
            "schema_version": "bifrost_lane_overlay.v3",
            "lane": "media-test",
            "locale": "lab",
            "backends": list(backends),
        }
    )


def _full_overlay() -> ModelBifrostLaneOverlay:
    return _overlay(*(_binding(c.value) for c in EnumBifrostLaneMediaCapability))


@pytest.mark.parametrize(
    "capability", [c.value for c in EnumBifrostLaneMediaCapability]
)
def test_golden_resolves_binding_and_url_by_capability_alone(capability: str) -> None:
    overlay = _full_overlay()

    resolved = overlay.resolve_media_capability(capability)

    assert resolved.backend_key == f"media-{capability}"
    assert resolved.endpoint_url == (
        f"{_HOST}:{_PORTS[capability]}{_PATHS[capability]}"
    )
    assert (
        overlay.resolve_media_capability(EnumBifrostLaneMediaCapability(capability))
        is resolved
    )


def test_golden_each_capability_has_its_own_endpoint_path() -> None:
    paths = {c.value: c.endpoint_path for c in EnumBifrostLaneMediaCapability}

    assert paths == _PATHS


def test_golden_dark_peer_does_not_shadow_the_serving_backend() -> None:
    overlay = _overlay(
        _binding("tts", backend_id="tts-old", serving=False),
        _binding("tts", backend_id="tts-new"),
    )

    assert overlay.resolve_media_capability("tts").backend_key == "tts-new"


def _assert_shape_disagreement_raises() -> None:
    with pytest.raises(BifrostLaneMediaCapabilityShapeError) as raised:
        ModelBifrostLaneBackendBinding.model_validate(
            _binding("stt", endpoint_url=f"{_HOST}:9002{_PATHS['tts']}")
        )
    assert "media-stt" in str(raised.value)


def test_error_capability_and_endpoint_shape_disagree_names_backend() -> None:
    _assert_shape_disagreement_raises()


def test_error_two_serving_backends_for_one_capability_names_both() -> None:
    with pytest.raises(BifrostLaneMediaCapabilityAmbiguousError) as raised:
        _overlay(
            _binding("tts", backend_id="tts-a"),
            _binding("tts", backend_id="tts-b"),
        )

    assert "tts-a" in str(raised.value)
    assert "tts-b" in str(raised.value)


def test_error_unknown_capability_names_the_capability() -> None:
    with pytest.raises(BifrostLaneMediaCapabilityUnknownError) as raised:
        _full_overlay().resolve_media_capability("hologram")

    assert "hologram" in str(raised.value)


def test_error_capability_asked_with_none_serving_names_dark_backend() -> None:
    overlay = _overlay(_binding("tts", backend_id="tts-dark", serving=False))

    with pytest.raises(BifrostLaneMediaCapabilityNotServingError) as raised:
        overlay.resolve_media_capability("tts")

    assert "tts-dark" in str(raised.value)


def test_error_capability_asked_with_no_declaring_backend() -> None:
    overlay = _overlay(_binding("tts"))

    with pytest.raises(BifrostLaneMediaCapabilityNotServingError) as raised:
        overlay.resolve_media_capability("video")

    assert "video" in str(raised.value)


def test_shape_check_break_turns_the_disagreement_test_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: with the shape check removed the disagreement test fails."""
    _assert_shape_disagreement_raises()

    monkeypatch.setattr(
        resolver_module,
        "check_media_capability_endpoint_shape",
        lambda backend_key, media_capability, endpoint_path: None,
    )

    with pytest.raises(pytest.fail.Exception):
        _assert_shape_disagreement_raises()
