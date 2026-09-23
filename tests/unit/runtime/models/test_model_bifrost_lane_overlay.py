# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hostile-case coverage for the overlay-only Bifrost lane contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

_QWEN_ENDPOINT = "http://192.168.86.201:8000/v1/chat/completions"
_DS_V4_ENDPOINT = "http://192.168.86.200:8101/v1/chat/completions"

# Per-backend shape, mirroring the committed dev lane overlay. OMN-17099: these
# are fixture values, no longer an authorization table the model checks against
# — the model validates SHAPE and completeness; which endpoint a lab lane binds
# is the lane overlay's own declaration.
_SHAPES: dict[str, dict[str, object]] = {
    "local-coder": {
        "endpoint_url": _QWEN_ENDPOINT,
        "served_model_id": "Qwen3.8-27B",
        "parameter_count": "27B",
        "context_window": 131_072,
        "max_tokens": 65_536,
        "timeout_ms": 300_000,
        "serving": True,
    },
    "local-heavy-reasoning": {
        "endpoint_url": _QWEN_ENDPOINT,
        "served_model_id": "Qwen3.8-27B",
        "parameter_count": "27B",
        "context_window": 131_072,
        "max_tokens": 65_536,
        "timeout_ms": 300_000,
        "serving": True,
    },
    "local-ds-v4-flash": {
        "endpoint_url": _DS_V4_ENDPOINT,
        "served_model_id": "deepseek-v4-flash",
        "parameter_count": "284B",
        "context_window": 131_072,
        "max_tokens": 65_536,
        "timeout_ms": 300_000,
        # OMN-16999: declared but not serving — .200:8101 answered http=000 on
        # the 2026-09-05 probe.
        "serving": False,
    },
}


def _binding(backend_id: str = "local-coder", **overrides: object) -> dict[str, object]:
    binding: dict[str, object] = {"backend_id": backend_id, **_SHAPES[backend_id]}
    binding.update(overrides)
    return binding


def _overlay(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": "bifrost_lane_overlay.v3",
        "lane": "dev",
        "locale": "lab",
        "backends": [_binding(backend_id) for backend_id in _SHAPES],
    }
    data.update(overrides)
    return data


@pytest.mark.unit
def test_valid_lab_overlay_round_trips_its_bindings() -> None:
    overlay = ModelBifrostLaneOverlay.model_validate(_overlay())

    assert [binding.backend_key for binding in overlay.backends] == [
        "local-coder",
        "local-heavy-reasoning",
        "local-ds-v4-flash",
    ]
    assert {binding.advertised_model for binding in overlay.backends} == {
        "Qwen3.8-27B",
        "deepseek-v4-flash",
    }
    dumped = overlay.model_dump(by_alias=True)["backends"][0]
    assert {key: dumped[key] for key in _binding()} == _binding()
    # A binding of a base-declared backend carries no added-backend declaration.
    assert (dumped["provider"], dumped["tier"], dumped["credential"]) == (
        None,
        None,
        None,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "endpoint_url",
    [
        # A bare base is not a complete endpoint (OMN-12815).
        "http://192.168.86.201:8000/v1",
        "http://192.168.86.201:8000/v1/chat/completions?model=Qwen3.8-27B",
        "http://user@192.168.86.201:8000/v1/chat/completions",
        "http://192.168.86.201:8000/v1/chat/completions#fragment",
        "ftp://192.168.86.201:8000/v1/chat/completions",
    ],
)
def test_incomplete_or_malformed_endpoint_is_rejected(endpoint_url: str) -> None:
    with pytest.raises(ValidationError, match="endpoint_url"):
        ModelBifrostLaneOverlay.model_validate(
            _overlay(
                backends=[
                    _binding(
                        bid,
                        **(
                            {"endpoint_url": endpoint_url}
                            if bid == "local-coder"
                            else {}
                        ),
                    )
                    for bid in _SHAPES
                ]
            )
        )


@pytest.mark.unit
def test_the_endpoint_host_is_the_lane_overlays_declaration() -> None:
    """OMN-17099: the model no longer pins a binding to a shipped lab host."""
    overlay = ModelBifrostLaneOverlay.model_validate(
        _overlay(
            backends=[
                _binding(
                    bid,
                    **(
                        {"endpoint_url": "https://192.0.2.9:8443/v1/chat/completions"}
                        if bid == "local-coder"
                        else {}
                    ),
                )
                for bid in _SHAPES
            ]
        )
    )

    assert (
        overlay.backends[0].endpoint_url == "https://192.0.2.9:8443/v1/chat/completions"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("backend_id", "field", "value"),
    [
        # A served id that disagrees with the base contract is refused by the
        # renderer, not here (OMN-17099) — see test_base_model_mismatch_fails_*.
        ("local-coder", "max_tokens", 131_073),
        ("local-coder", "context_window", 0),
        ("local-coder", "endpoint_url_env", "BIFROST_LOCAL_CODER_ENDPOINT_URL"),
        # A credential is a declared ``credential``, never a bare secret_ref.
        ("local-coder", "secret_ref", "infisical://local-coder"),
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
def test_duplicate_backend_is_rejected() -> None:
    complete = [_binding(bid) for bid in _SHAPES]
    with pytest.raises(ValidationError, match="duplicate"):
        ModelBifrostLaneOverlay.model_validate(
            _overlay(backends=[*complete, _binding()])
        )


@pytest.mark.unit
def test_a_backend_the_model_cannot_place_is_left_to_the_renderer() -> None:
    """OMN-17099: set equality with a shipped list is gone from the model.

    Whether a binding names a base backend, adds a new one, or omits a routed
    one is only decidable against the base contract, so the renderer refuses
    those cases (``test_bifrost_lane_overlay_adds_backend_omn17099.py``). The
    model accepts an overlay binding fewer backends than before.
    """
    overlay = ModelBifrostLaneOverlay.model_validate(
        _overlay(backends=[_binding("local-coder")])
    )

    assert [binding.backend_key for binding in overlay.backends] == ["local-coder"]


# ---------------------------------------------------------------------------
# OMN-17502: execution locale. A lane that runs where the lab backends do not
# exist (the onex-dev cloud lane — beta axiom 9, cloud execution locale, BYOK)
# must be able to state that as a FACT. Before this ticket the only schema-valid
# overlay was the exact lab set, so a cloud lane could either mount three
# unreachable lab endpoints or not render at all (OMN-17502 CrashLoopBackOff).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cloud_locale_declares_zero_local_backends() -> None:
    """A cloud lane states the absence explicitly: locale + an empty mapping."""
    overlay = ModelBifrostLaneOverlay.model_validate(
        {
            "schema_version": "bifrost_lane_overlay.v3",
            "lane": "onex-dev",
            "locale": "cloud",
            "backends": [],
        }
    )

    assert overlay.locale is EnumBifrostLaneLocale.CLOUD
    assert overlay.backends == ()


@pytest.mark.unit
def test_cloud_locale_listing_a_backend_is_rejected_naming_lane_and_rule() -> None:
    """A cloud lane may not smuggle a lab endpoint in through the overlay."""
    with pytest.raises(ValidationError) as excinfo:
        ModelBifrostLaneOverlay.model_validate(
            _overlay(lane="onex-dev", locale="cloud", backends=[_binding()])
        )

    message = str(excinfo.value)
    assert "'onex-dev'" in message
    assert "locale 'cloud'" in message
    assert "local-coder" in message


@pytest.mark.unit
def test_lab_locale_with_zero_backends_is_rejected() -> None:
    """An empty lab overlay is the OMN-16833 silent-degradation shape."""
    with pytest.raises(ValidationError, match="locale 'lab'"):
        ModelBifrostLaneOverlay.model_validate(_overlay(locale="lab", backends=[]))


@pytest.mark.unit
def test_locale_is_required_and_has_no_default() -> None:
    """No default: a lane's execution locale is a stated fact, never inherited.

    A defaulted locale would make ``lab`` the silent answer for any overlay that
    forgot to declare one — the same fallthrough class OMN-17150 removed from
    the overlay PATH, one level down in the overlay CONTENT.
    """
    data = _overlay()
    del data["locale"]

    with pytest.raises(ValidationError, match="locale"):
        ModelBifrostLaneOverlay.model_validate(data)

    assert ModelBifrostLaneOverlay.model_fields["locale"].is_required()


@pytest.mark.unit
def test_schema_version_names_the_locale_shape() -> None:
    """v2 files predate the locale field and no longer validate as v3."""
    with pytest.raises(ValidationError, match=r"bifrost_lane_overlay\.v3"):
        ModelBifrostLaneOverlay.model_validate(
            _overlay(schema_version="bifrost_lane_overlay.v2")
        )
