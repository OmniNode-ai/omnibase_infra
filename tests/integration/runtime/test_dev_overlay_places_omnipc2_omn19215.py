# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19215: the committed dev overlay's placement reaches the rendered contract.

The routing authority in omnimarket mirrors a rendered backend that carries a
``placement`` into the routing tier it names, after the rungs it backs. This
drives the COMMITTED dev overlay through the real renderer and proves the
second lab host's placement arrives intact, naming rungs the same rendered
contract binds, within the window the backend declares.

The negative control is a copy of the real dev overlay whose placement offers
more context than the backend declares. It must be refused and leave no
artifact; without it a renderer that stopped checking would pass the positive
case and prove nothing.
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
_PLACED = "local-omnipc2-chat"

#: The base-declared local rungs the dev overlay binds, with the env hint the
#: base contract carries for each (the renderer strips it).
_ENDPOINT_URL_ENV = {
    "local-coder": "LLM_CODER_URL",
    "local-heavy-reasoning": "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
}


def _dev_overlay() -> dict[str, Any]:
    loaded = yaml.safe_load(_DEV_OVERLAY.read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_base_contract(path: Path) -> None:
    """A minimal base contract carrying the base rungs the dev overlay binds."""
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


def test_the_committed_dev_overlay_renders_the_omnipc2_placement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract(source)

    render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=_DEV_OVERLAY,
        target_path=target,
        environ={},
    )

    rendered = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in rendered["backends"]}
    declared = next(b for b in _dev_overlay()["backends"] if b["backend_id"] == _PLACED)
    placement = by_id[_PLACED]["placement"]

    assert placement == declared["placement"]
    assert placement["tier"] == "local"
    # Every rung it backs is a backend this same rendered contract binds.
    for rung in placement["fallback_for"]:
        assert by_id[rung]["endpoint_url"], f"{rung} is not bound in the render"
    assert placement["max_context_tokens"] <= declared["context_window"]


def test_an_oversized_placement_in_the_dev_overlay_is_refused(tmp_path: Path) -> None:
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
