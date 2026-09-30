# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19432: the dev overlay places gpt-oss-120b as a spread peer for the long-thinking classes.

RULING 2026-09-30T15:07:51Z: no model is reserved as a fallback rung. The Mac
Studio's gpt-oss-120b takes a share of the local-heavy-reasoning rung's first
choice traffic, but only for the classes it measured well on and where its
latency (p50 69 s) is normal. This drives the COMMITTED dev overlay through the
real renderer and proves the placement, its class list and its weight arrive at
the routing authority; the controls show each of the new keys is written only
when it differs from the default, so an older routing authority still parses the
older placements byte for byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_DEV_OVERLAY = _ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
_PLACED = "local-studio-planner"
_SHORT_PROSE = {"document", "documentation", "summarization"}
_ENDPOINT_URL_ENV = {
    "local-coder": "LLM_CODER_URL",
    "local-heavy-reasoning": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
    "local-embedding": "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
}


def _dev_overlay() -> dict[str, Any]:
    loaded = yaml.safe_load(_DEV_OVERLAY.read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


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
    return {backend["backend_id"]: backend for backend in rendered["backends"]}


def _placement_row(overlay: dict[str, Any]) -> dict[str, Any]:
    return next(b for b in overlay["backends"] if b["backend_id"] == _PLACED)[
        "placement"
    ]


def test_the_committed_overlay_renders_gpt_oss_as_a_narrowed_spread_peer(
    tmp_path: Path,
) -> None:
    rendered = _render(tmp_path, _dev_overlay())[_PLACED]
    placement = rendered["placement"]
    assert (
        rendered["endpoint_url"] == "http://192.168.86.200:8130/v1/chat/completions"
    )  # onex-allow-internal-ip
    assert rendered["model_name"] == "gpt-oss-120b"
    assert placement["mode"] == "spread"
    assert placement["tier"] == "local"
    assert placement["fallback_for"] == ["local-heavy-reasoning"]
    assert placement["max_context_tokens"] == 131072
    assert {"review", "reasoning", "planning", "research"} <= set(placement["use_for"])
    # The short prose classes, where a minute-long answer is not normal, stay off it.
    assert not set(placement["use_for"]) & _SHORT_PROSE


def test_the_default_weight_renders_no_weight_key(tmp_path: Path) -> None:
    """CONTROL: weight 1.0 is the default, so an older routing authority sees no new key."""
    assert "weight" not in _render(tmp_path, _dev_overlay())[_PLACED]["placement"]


def test_a_declared_weight_and_class_list_reach_the_rendered_contract(
    tmp_path: Path,
) -> None:
    overlay = _dev_overlay()
    _placement_row(overlay)["weight"] = 2.5
    placement = _render(tmp_path, overlay)[_PLACED]["placement"]
    assert placement["weight"] == 2.5


def test_a_placement_without_a_class_list_renders_none(tmp_path: Path) -> None:
    """CONTROL: no ``use_for`` means the rung's whole list, and no key is written."""
    overlay = _dev_overlay()
    del _placement_row(overlay)["use_for"]
    assert "use_for" not in _render(tmp_path, overlay)[_PLACED]["placement"]
