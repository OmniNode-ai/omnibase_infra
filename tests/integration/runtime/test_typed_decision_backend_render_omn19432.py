# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19432: typed-decision admission through fixture lane overlays.

Render local, cloud and typed-decision backends together through lab and cloud
fixture overlays. Local bindings are substituted or disabled, cloud endpoints
pass through, and the typed-decision operation URL survives the full render.

The negative control uses the same overlays with a bare-base typed-decision URL
and requires refusal without a partial artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)
from tests.helpers.lane_overlay_fixtures import (
    cloud_overlay,
    lab_overlay,
    write_overlay,
)

pytestmark = pytest.mark.integration


@pytest.fixture(
    params=["example-lab", "example-cloud"],
    ids=["lab-locale-overlay", "cloud-locale-overlay"],
)
def overlay_path(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    lane = str(request.param)
    overlay = (
        lab_overlay(lane=lane) if lane == "example-lab" else cloud_overlay(lane=lane)
    )
    return write_overlay(tmp_path / f"{lane}.bifrost.yaml", overlay)


_CLOUD_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
#: TypeSafe Jev's System One endpoint (the companion omnimarket PR's backend).
_TYPED_DECISION_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def _write_base_contract_with_typed_decision_backend(
    path: Path, *, decision_endpoint: str
) -> None:
    """A base contract shaped like the packaged omnimarket one: one local
    backend the fixture overlays bind or disable, one cloud backend, and one
    typed-decision backend (OMN-19432) resolved straight from this file."""
    path.write_text(
        yaml.safe_dump(
            {
                "backends": [
                    {
                        "backend_id": "local-coder",
                        "model_name": "Qwen3.8-27B",
                        "endpoint_url_env": "BIFROST_LOCAL_CODER_ENDPOINT_URL",
                        "endpoint_url": None,
                        "tier": "local",
                    },
                    {
                        "backend_id": "local-heavy-reasoning",
                        "model_name": "Qwen3.8-27B",
                        "endpoint_url_env": "BIFROST_LOCAL_CODER_ENDPOINT_URL",
                        "endpoint_url": None,
                        "tier": "local",
                    },
                    {
                        "backend_id": "cloud-gemini-pro",
                        "model_name": "gemini-2.5-flash",
                        "endpoint_url": _CLOUD_ENDPOINT,
                        "tier": "frontier_api",
                    },
                    {
                        "backend_id": "cloud-typed-decision",
                        "provider": "typesafe",
                        "model_name": "jev-latest",
                        "endpoint_url": decision_endpoint,
                        "secret_ref": "llm.typesafe.api_key",
                        "tier": "typed_decision",
                    },
                ],
                "routing_rules": [
                    {
                        "rule_id": "d4e5f6a7-0002-4000-8000-000000000001",
                        "task_class": "code_generation",
                        "backend_ids": ["local-coder", "cloud-gemini-pro"],
                    }
                ],
                "default_backends": ["local-coder", "cloud-gemini-pro"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_typed_decision_backend_renders_verbatim_through_fixture_overlay(
    overlay_path: Path, tmp_path: Path
) -> None:
    """The typed-decision backend's URL passes through untouched by either the
    fixture lab overlay (which binds ``local-coder``) or the fixture cloud overlay
    (which disables it), and it renders alongside both other backend kinds."""
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract_with_typed_decision_backend(
        source, decision_endpoint=_TYPED_DECISION_ENDPOINT
    )

    rendered = render_bifrost_delegation_contract(
        source_path=source,
        overlay_path=overlay_path,
        target_path=target,
        environ={},
    )
    assert rendered == target

    contract = yaml.safe_load(target.read_text(encoding="utf-8"))
    by_id = {backend["backend_id"]: backend for backend in contract["backends"]}
    assert by_id["cloud-typed-decision"]["endpoint_url"] == _TYPED_DECISION_ENDPOINT
    assert by_id["cloud-gemini-pro"]["endpoint_url"] == _CLOUD_ENDPOINT


def test_bare_base_typed_decision_endpoint_is_refused_through_fixture_overlay(
    overlay_path: Path, tmp_path: Path
) -> None:
    """NEGATIVE CONTROL: a typed-decision endpoint that names only an API
    version, not its own operation, is refused end to end through the fixture
    overlay — proving the OMN-19432 admission check fires inside the
    full render path, not only against an isolated tmp_path fixture."""
    source = tmp_path / "base.yaml"
    target = tmp_path / "rendered.yaml"
    _write_base_contract_with_typed_decision_backend(
        source, decision_endpoint="https://api.typesafe.ai/v1"
    )

    with pytest.raises(ProtocolConfigurationError, match="typed-decision endpoint"):
        render_bifrost_delegation_contract(
            source_path=source,
            overlay_path=overlay_path,
            target_path=target,
            environ={},
        )
    assert not target.exists(), (
        "a refused render must leave no rendered contract behind — a partial "
        "write here would be read by the next process to start"
    )
