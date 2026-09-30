# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19432: a lane overlay placement carries ``use_for`` and ``weight`` to the routing authority.

RULING 2026-09-30T15:09:11Z: no model is reserved as a fallback rung; each takes traffic in
proportion to what it measures. The routing authority (omnimarket) reads two placement keys
the lane overlay could not carry, because the placement model refuses unknown keys: ``use_for``
narrows the classes a placed backend serves, ``weight`` sets its share of a spread group.

This drives the real renderer over a copy of the committed dev overlay whose placed backend
declares both, and shows each key is written only when it differs from the default, so a
routing authority older than the keys still parses every existing placement byte for byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_bifrost_lane_backend_placement import (
    ModelBifrostLaneBackendPlacement,
)
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_DEV_OVERLAY = _ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
_PLACED = "local-omnipc2-chat"
_ENDPOINT_URL_ENV = {
    "local-coder": "LLM_CODER_URL",
    "local-heavy-reasoning": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
    "local-embedding": "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
}


def _overlay() -> dict[str, Any]:
    loaded = yaml.safe_load(_DEV_OVERLAY.read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _placement(overlay: dict[str, Any]) -> dict[str, Any]:
    row = next(b for b in overlay["backends"] if b["backend_id"] == _PLACED)
    return row["placement"]


def _render(tmp_path: Path, overlay: dict[str, Any]) -> dict[str, Any]:
    overlay_path = tmp_path / "overlay.bifrost.yaml"
    overlay_path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    served = {b["backend_id"]: b["served_model_id"] for b in overlay["backends"]}
    source = tmp_path / "base.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": backend_id,
                        "model_name": served[backend_id],
                        "endpoint_url_env": env_name,
                        "required": True,
                    }
                    for backend_id, env_name in _ENDPOINT_URL_ENV.items()
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    target = tmp_path / "rendered.yaml"
    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=overlay_path,
        target_path=target,
        environ={},
    )
    rendered = yaml.safe_load(target.read_text(encoding="utf-8"))
    return {b["backend_id"]: b for b in rendered["backends"]}[_PLACED]["placement"]


def test_a_declared_weight_and_class_list_reach_the_rendered_contract(
    tmp_path: Path,
) -> None:
    overlay = _overlay()
    _placement(overlay).update({"weight": 2.5, "use_for": ["review", "reasoning"]})
    rendered = _render(tmp_path, overlay)
    assert rendered["weight"] == 2.5
    assert rendered["use_for"] == ["review", "reasoning"]


def test_the_defaults_render_neither_key(tmp_path: Path) -> None:
    """CONTROL: the committed placement declares neither, so an older authority sees no new key."""
    rendered = _render(tmp_path, _overlay())
    assert "weight" not in rendered
    assert "use_for" not in rendered


def test_a_weight_of_one_is_the_default_and_is_not_written(tmp_path: Path) -> None:
    overlay = _overlay()
    _placement(overlay)["weight"] = 1.0
    assert "weight" not in _render(tmp_path, overlay)


@pytest.mark.parametrize("weight", [0, -1.0])
def test_a_weight_that_is_not_positive_is_refused(weight: float) -> None:
    with pytest.raises(ValidationError):
        ModelBifrostLaneBackendPlacement(
            tier="local", fallback_for=("a",), max_context_tokens=1, weight=weight
        )


def test_an_empty_class_list_is_refused() -> None:
    with pytest.raises(ValidationError):
        ModelBifrostLaneBackendPlacement(
            tier="local", fallback_for=("a",), max_context_tokens=1, use_for=()
        )
