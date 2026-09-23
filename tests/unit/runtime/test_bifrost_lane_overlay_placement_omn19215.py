# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19215: a lane-added backend may carry a tier placement to the routing authority.

OMN-17099 lets a lane overlay ADD a backend, but routing only offers a backend
the packaged routing ladder names, so an added backend was reachable only by a
per-run pin. The routing authority (omnimarket) now reads an optional
``placement`` on a rendered backend entry and mirrors the backend into the named
tier after the rungs it backs. This renderer's part is small: validate what it
alone can check, and pass the placement through unchanged.

What it alone can check is the context window, because only the lane overlay
declares one: a placement may not offer the backend a larger prompt than the
backend says it holds. The tier and rung names are checked by the routing
authority against the ladder it loads, which this renderer does not read.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.models.model_bifrost_lane_backend_binding import (
    ModelBifrostLaneBackendBinding,
)
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

# RFC 5737 documentation-range addresses, never lab hosts.
_BASE_ENDPOINT = "http://192.0.2.10:8000/v1/chat/completions"
_ADDED_ENDPOINT = "http://192.0.2.20:8000/v1/chat/completions"

_PLACEMENT = {
    "tier": "local",
    "fallback_for": ["local-coder"],
    "max_context_tokens": 32_768,
}


def _base_contract(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-coder",
                        "provider": "local",
                        "endpoint_url": None,
                        "model_name": "served-model",
                        "tier": "local",
                        "timeout_ms": 300_000,
                    },
                    {
                        "backend_id": "cloud-frontier",
                        "provider": "gemini",
                        "endpoint_url": "https://api.example.test/v1/chat/completions",
                        "model_name": "frontier-model",
                        "secret_ref": "llm.frontier.api_key",
                        "tier": "frontier_api",
                        "timeout_ms": 60_000,
                    },
                ],
                "routing_rules": [
                    {"task_class": "code_generation", "backend_ids": ["local-coder"]}
                ],
                "default_backends": ["cloud-frontier"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _base_binding(**overrides: object) -> dict[str, object]:
    binding: dict[str, object] = {
        "backend_id": "local-coder",
        "endpoint_url": _BASE_ENDPOINT,
        "served_model_id": "served-model",
        "parameter_count": "27B",
        "context_window": 131_072,
        "max_tokens": 65_536,
        "timeout_ms": 300_000,
    }
    binding.update(overrides)
    return binding


def _added_binding(**overrides: object) -> dict[str, object]:
    binding: dict[str, object] = {
        "backend_id": "local-second-host",
        "endpoint_url": _ADDED_ENDPOINT,
        "served_model_id": "served-model",
        "parameter_count": "27B",
        "context_window": 32_768,
        "max_tokens": 16_384,
        "timeout_ms": 300_000,
        "provider": "local",
        "tier": "local",
        "credential": {"kind": "none"},
        "placement": dict(_PLACEMENT),
    }
    binding.update(overrides)
    return binding


def _render(tmp_path: Path, backends: list[dict[str, object]]) -> dict[str, object]:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "schema_version": "bifrost_lane_overlay.v3",
                "lane": "dev",
                "locale": "lab",
                "backends": backends,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target = tmp_path / "rendered.yaml"
    render_bifrost_delegation_contract(
        source_path=_base_contract(tmp_path / "base.yaml"),
        overlay_path=overlay,
        target_path=target,
        environ={},
        verify_endpoints=False,
    )
    rendered = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert isinstance(rendered, dict)
    return rendered


def _added(rendered: dict[str, object]) -> dict[str, object]:
    backends = rendered["backends"]
    assert isinstance(backends, list)
    return next(b for b in backends if b["backend_id"] == "local-second-host")


@pytest.mark.unit
def test_a_placement_is_passed_through_to_the_rendered_backend(tmp_path: Path) -> None:
    rendered = _render(tmp_path, [_base_binding(), _added_binding()])
    assert _added(rendered)["placement"] == _PLACEMENT


@pytest.mark.unit
def test_an_added_backend_without_a_placement_renders_no_placement_key(
    tmp_path: Path,
) -> None:
    binding = _added_binding()
    del binding["placement"]
    rendered = _render(tmp_path, [_base_binding(), binding])
    assert "placement" not in _added(rendered)


@pytest.mark.unit
def test_a_placement_larger_than_the_context_window_fails_naming_the_backend() -> None:
    with pytest.raises(ValidationError, match="local-second-host"):
        ModelBifrostLaneBackendBinding.model_validate(
            _added_binding(placement={**_PLACEMENT, "max_context_tokens": 65_536})
        )


@pytest.mark.unit
def test_a_base_backend_cannot_be_given_a_placement_by_a_lane(tmp_path: Path) -> None:
    # A base-declared rung is already in the ladder; a lane placing it again
    # would duplicate it. Only a backend the lane adds carries a placement.
    with pytest.raises(
        (ValidationError, ProtocolConfigurationError), match="local-coder"
    ):
        _render(tmp_path, [_base_binding(placement=dict(_PLACEMENT))])


@pytest.mark.unit
@pytest.mark.parametrize(
    "placement",
    [
        {"tier": "local", "fallback_for": [], "max_context_tokens": 1024},
        {"tier": "", "fallback_for": ["local-coder"], "max_context_tokens": 1024},
        {"tier": "local", "fallback_for": ["local-coder"], "max_context_tokens": 0},
        {
            "tier": "local",
            "fallback_for": ["local-coder"],
            "weight": 2,
            "max_context_tokens": 1,
        },
    ],
    ids=["no-rungs", "no-tier", "no-context", "unknown-field"],
)
def test_a_malformed_placement_is_refused(placement: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelBifrostLaneBackendBinding.model_validate(
            _added_binding(placement=placement)
        )


@pytest.mark.unit
def test_the_render_refuses_an_oversized_placement_with_a_typed_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtocolConfigurationError, match="local-second-host"):
        _render(
            tmp_path,
            [
                _base_binding(),
                _added_binding(placement={**_PLACEMENT, "max_context_tokens": 65_536}),
            ],
        )
