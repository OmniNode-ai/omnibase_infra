# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19432: the committed dev overlay places the .200 Studio planner as a spread peer.

The rung was placed for gpt-oss-120b (OMN-19432), served Qwen3.6-35B-A3B from
OMN-17427 (2026-10-08), and since OMN-20422 (2026-10-09) serves Qwen3.8-27B as
``Qwen3.8-27B`` on the same endpoint, with the same placement.

The deployed dev lane reads this overlay, not the operator's host overlay: of 447
delegation runs in the 24 hours to 2026-09-30, 438 ran on the deployed lane and
none reached the .200 planner, because the planner was bound only in the host
overlay. This drives the COMMITTED dev overlay through the real renderer and
proves the planner arrives in the rendered contract as a weighted spread peer of
the reasoning rung, for the classes it was measured on, within its window.

Properties pinned, each with a control:

* it is a spread peer with a weight below the rung's own, for the classes it was
  measured on (RULING 2026-09-30T15:07:51Z: no model is a fallback-only rung);
* it backs the reasoning rung only, never the code rungs, and not document or
  summarization;
* the window is enforced against the real overlay row (negative control).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_DEV_OVERLAY = _ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
_PLACED = "local-studio-planner"
_RUNG = "local-heavy-reasoning"

_ENDPOINT_URL_ENV = {
    "local-coder": "LLM_CODER_URL",
    "local-heavy-reasoning": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
    "local-embedding": "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
}


def _dev_overlay() -> dict[str, Any]:
    loaded = yaml.safe_load(_DEV_OVERLAY.read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_base_contract(path: Path) -> None:
    served = {
        backend["backend_id"]: backend["served_model_id"]
        for backend in _dev_overlay()["backends"]
    }
    path.write_text(
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


def _render(tmp_path: Path, overlay_path: Path) -> dict[str, Any]:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)
    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=overlay_path,
        target_path=target,
        environ={},
    )
    rendered = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert isinstance(rendered, dict)
    return {backend["backend_id"]: backend for backend in rendered["backends"]}


def test_the_committed_dev_overlay_renders_the_planner_behind_the_reasoning_rung(
    tmp_path: Path,
) -> None:
    by_id = _render(tmp_path, _DEV_OVERLAY)
    declared = next(b for b in _dev_overlay()["backends"] if b["backend_id"] == _PLACED)
    backend = by_id[_PLACED]

    assert backend["model_name"] == "Qwen3.8-27B"
    assert backend["endpoint_url"] == declared["endpoint_url"]
    placement = backend["placement"]
    assert placement["tier"] == "local"
    # It backs the reasoning rung and nothing else, and that rung is bound in
    # the same rendered contract.
    assert placement["fallback_for"] == [_RUNG]
    assert by_id[_RUNG]["endpoint_url"], f"{_RUNG} is not bound in the render"
    assert placement["max_context_tokens"] <= declared["context_window"]
    # Its routing window is the rung's own 8192, never the pool the server holds:
    # on 9000 to 15500-token plans it grounded worse than the Qwen rung (unsupported
    # claims in 65 of 177 against 33 of 185, worse on 8 of 10 documents), so it is
    # not a long-context rung. Widening it needs a new measurement.
    assert placement["max_context_tokens"] == 8192
    assert declared["context_window"] == 262144


def test_the_planner_is_a_weighted_spread_peer_for_its_measured_classes(
    tmp_path: Path,
) -> None:
    """RULING 2026-09-30T15:07:51Z: no fallback-only rung. It shares first-choice traffic."""
    placement = _render(tmp_path, _DEV_OVERLAY)[_PLACED]["placement"]
    assert placement["mode"] == "spread"
    assert placement["weight"] == 0.25
    assert placement["use_for"] == [
        "review",
        "reasoning",
        "complex_reasoning",
        "planning",
        "research",
        "escalation",
    ]
    # Never the classes it grounds worse on, and never the code rungs.
    assert "document" not in placement["use_for"]
    assert "summarization" not in placement["use_for"]
    assert not any("code" in task for task in placement["use_for"])


def test_the_planner_weight_is_below_the_rungs_own(tmp_path: Path) -> None:
    """Its share is capacity-set: a fraction of the rung's 1.0, never above it."""
    weight = _render(tmp_path, _DEV_OVERLAY)[_PLACED]["placement"]["weight"]
    assert 0 < weight < 1.0


def test_a_fallback_mode_renders_no_mode_key(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: the committed spread mode is what the render carries."""
    overlay = _dev_overlay()
    row = next(b for b in overlay["backends"] if b["backend_id"] == _PLACED)
    row["placement"]["mode"] = "fallback"
    poisoned = tmp_path / "fallback.bifrost.yaml"
    poisoned.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")

    assert "mode" not in _render(tmp_path, poisoned)[_PLACED]["placement"]


def test_an_oversized_planner_placement_is_refused(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: the window is enforced against the real overlay row."""
    overlay = _dev_overlay()
    row = next(b for b in overlay["backends"] if b["backend_id"] == _PLACED)
    row["placement"]["max_context_tokens"] = row["context_window"] + 1

    poisoned = tmp_path / "poisoned.bifrost.yaml"
    poisoned.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)

    with pytest.raises(ProtocolConfigurationError, match=_PLACED):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=poisoned,
            target_path=target,
            environ={},
        )
    assert not target.exists()
