# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19215: the env-resolved render refuses a placement the pinned ladder lacks.

The entrypoint names the target, the lane overlay and the routing ladder by
environment variable only. This drives the render the same way, with no path
argument, so the ladder check runs from the variables the runtime sets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract_with_source,
)

pytestmark = pytest.mark.integration

_PLACEMENT = {
    "tier": "local",
    "fallback_for": ["local-coder"],
    "max_context_tokens": 32_768,
}


def _write(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _environ(tmp_path: Path, *, tier: str) -> dict[str, str]:
    overlay = {
        "schema_version": "bifrost_lane_overlay.v3",
        "lane": "dev",
        "locale": "lab",
        "backends": [
            {
                "backend_id": "local-coder",
                "endpoint_url": "http://192.0.2.10:8000/v1/chat/completions",
                "served_model_id": "served-model",
                "parameter_count": "27B",
                "context_window": 131_072,
                "max_tokens": 65_536,
                "timeout_ms": 300_000,
            },
            {
                "backend_id": "local-second-host",
                "endpoint_url": "http://192.0.2.20:8000/v1/chat/completions",
                "served_model_id": "served-model",
                "parameter_count": "27B",
                "context_window": 32_768,
                "max_tokens": 16_384,
                "timeout_ms": 300_000,
                "provider": "local",
                "tier": "local",
                "credential": {"kind": "none"},
                "placement": {**_PLACEMENT, "tier": tier},
            },
        ],
    }
    ladder = {
        "tiers": [
            {"name": "local", "models": [{"backend_id": "local-coder"}]},
            {"name": "cheap_cloud", "models": [{"backend_id": "cloud-flash"}]},
        ]
    }
    return {
        "BIFROST_CONTRACT_PATH": str(tmp_path / "rendered.yaml"),
        "BIFROST_LANE_OVERLAY_PATH": str(_write(tmp_path / "lane.yaml", overlay)),
        "DELEGATION_ROUTING_TIERS_PATH": str(_write(tmp_path / "tiers.yaml", ladder)),
    }


def _base(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "base.yaml",
        {
            "backends": [
                {
                    "backend_id": "local-coder",
                    "provider": "local",
                    "model_name": "base-model",
                    "tier": "local",
                }
            ],
            "routing_rules": [
                {"task_class": "code_generation", "backend_ids": ["local-coder"]}
            ],
            "default_backends": ["local-coder"],
        },
    )


def test_env_resolved_render_passes_a_placement_the_ladder_holds(
    tmp_path: Path,
) -> None:
    outcome = render_bifrost_delegation_contract_with_source(
        source_path=_base(tmp_path),
        environ=_environ(tmp_path, tier="local"),
        verify_endpoints=False,
    )
    assert outcome.path == tmp_path / "rendered.yaml"
    rendered = yaml.safe_load(outcome.path.read_text(encoding="utf-8"))
    placed = [b for b in rendered["backends"] if "placement" in b]
    assert [b["backend_id"] for b in placed] == ["local-second-host"]


def test_env_resolved_render_refuses_a_tier_the_ladder_lacks(tmp_path: Path) -> None:
    with pytest.raises(
        ProtocolConfigurationError, match=r"local-second-host.*unknown tier .lokal."
    ):
        render_bifrost_delegation_contract_with_source(
            source_path=_base(tmp_path),
            environ=_environ(tmp_path, tier="lokal"),
            verify_endpoints=False,
        )
    assert not (tmp_path / "rendered.yaml").exists()
