# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20490: the dev overlay's .202 ceiling reaches the rendered contract within its share.

The second lab host serves one unified token pool to several slots, so a
request must keep prompt plus output within ``pool / slots``. This drives the
COMMITTED dev overlay through the real renderer and checks the rendered
``max_tokens`` of the rung bound to that host against the capacity the same
overlay declares.

The negative control renders a copy of the real overlay whose ceiling is the
whole share. The renderer carries it through unchanged, so the same check
refuses it; without that, a renderer that rewrote the ceiling would pass the
positive case and prove nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.models.delegation.model_bifrost_lane_endpoint_capacity import (
    ModelBifrostLaneEndpointCapacity,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_DEV_OVERLAY = _ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
_POOLED = "local-omnipc2-chat"


def _overlay_data() -> dict[str, Any]:
    loaded = yaml.safe_load(_DEV_OVERLAY.read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _render(overlay_data: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Render ``overlay_data`` against a base declaring the rungs it rebinds."""
    overlay_path = tmp_path / "overlay.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay_data, sort_keys=False), "utf-8")
    overlay = ModelBifrostLaneOverlay.model_validate(overlay_data)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": binding.backend_key,
                        "model_name": binding.advertised_model,
                        "tier": "local",
                    }
                    for binding in overlay.backends
                    if not binding.declares_new_backend
                ]
            }
        ),
        "utf-8",
    )
    target = tmp_path / "rendered.yaml"
    render_bifrost_delegation_contract(
        source_path=base_path,
        overlay_path=overlay_path,
        target_path=target,
        environ={},
        verify_endpoints=False,
    )
    rendered = yaml.safe_load(target.read_text("utf-8"))
    return {backend["backend_id"]: backend for backend in rendered["backends"]}


def _capacity_of(
    overlay_data: dict[str, Any], endpoint_url: str
) -> ModelBifrostLaneEndpointCapacity:
    overlay = ModelBifrostLaneOverlay.model_validate(overlay_data)
    capacity = next(
        (item for item in overlay.endpoint_capacity if item.serves(endpoint_url)),
        None,
    )
    assert capacity is not None, f"{endpoint_url} has no declared endpoint_capacity"
    return capacity


def _fits_share(max_tokens: int, capacity: ModelBifrostLaneEndpointCapacity) -> bool:
    return max_tokens + capacity.prompt_headroom_tokens <= capacity.per_slot_tokens


def test_rendered_pooled_rung_fits_its_declared_share(tmp_path: Path) -> None:
    data = _overlay_data()
    rendered = _render(data, tmp_path)[_POOLED]

    capacity = _capacity_of(data, rendered["endpoint_url"])
    assert _fits_share(rendered["max_tokens"], capacity), (
        f"{_POOLED} renders max_tokens={rendered['max_tokens']}; its share allows "
        f"{capacity.max_output_tokens}"
    )


def test_rendered_ceiling_is_the_overlays_own_not_a_default(tmp_path: Path) -> None:
    data = _overlay_data()
    capacity = _capacity_of(
        data,
        next(
            backend["endpoint_url"]
            for backend in data["backends"]
            if backend["backend_id"] == _POOLED
        ),
    )
    for backend in data["backends"]:
        if backend["backend_id"] == _POOLED:
            backend["max_tokens"] = capacity.per_slot_tokens

    rendered = _render(data, tmp_path)[_POOLED]

    assert rendered["max_tokens"] == capacity.per_slot_tokens
    assert not _fits_share(rendered["max_tokens"], capacity)
