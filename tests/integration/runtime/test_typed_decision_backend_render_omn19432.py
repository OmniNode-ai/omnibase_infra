# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19432: drive the typed-decision backend admission through the REAL
committed lane overlays.

``tests/unit/runtime/test_render_bifrost_delegation_contract.py`` proves the
admission logic (``_is_typed_decision_backend`` /
``_typed_decision_endpoint_is_complete``) against synthetic ``tmp_path``
fixtures in isolation. This module is the integration half, in the same spirit
as ``test_committed_lab_overlays_render_omn18570.py`` next to it: it proves the
new admission logic still holds once a typed-decision backend is rendered
alongside the real committed lab overlay (local rungs bound from
``dev.bifrost.yaml``) and the real committed cloud overlay
(``onex-dev.bifrost.yaml``), not only against a hand-built fixture that has no
other backend to interact with.

Why this matters here specifically: the renderer resolves a typed-decision
backend's ``endpoint_url`` straight from the base contract (the overlay never
touches it, unlike a ``local`` backend), so the one way to prove it survives
the FULL render path — local-backend overlay substitution, cloud-backend
pass-through, then the typed-decision admission check — is to render all three
kinds together through a real overlay file, which is exactly what production
does when the container starts (see this ticket's PR body).

The negative control renders the same fixture through the same real overlay
with a bare-base typed-decision URL, and asserts the render is REFUSED end to
end with no partial artifact — the admission check firing inside the full
committed-overlay path, not only against an isolated tmp_path fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_OVERLAY_DIR = _ROOT / "docker" / "lane-overlays"
_DEV_OVERLAY = _OVERLAY_DIR / "dev.bifrost.yaml"
_CLOUD_OVERLAY = _OVERLAY_DIR / "onex-dev.bifrost.yaml"

_CLOUD_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
#: TypeSafe Jev's System One endpoint (the companion omnimarket PR's backend).
_TYPED_DECISION_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def _write_base_contract_with_typed_decision_backend(
    path: Path, *, decision_endpoint: str
) -> None:
    """A base contract shaped like the packaged omnimarket one: one local
    backend the committed overlays bind or disable, one cloud backend, and one
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
                        # The real dev.bifrost.yaml overlay also binds this id
                        # without declaring provider/tier/credential itself, so
                        # the base contract must declare it (OMN-17099) for
                        # that overlay to render at all.
                        "backend_id": "local-heavy-reasoning",
                        "model_name": "Qwen3.8-27B",
                        "endpoint_url_env": "BIFROST_LOCAL_CODER_ENDPOINT_URL",
                        "endpoint_url": None,
                        "tier": "local",
                    },
                    {
                        # OMN-17099 added this id to the real dev.bifrost.yaml
                        # overlay without declaring provider/tier/credential
                        # itself either (same shape as local-heavy-reasoning
                        # above), so the base contract must declare it too, or
                        # this module's positive case fails on every dev-lane
                        # render with "not declared by the base contract" and
                        # the negative-control case fails on the SAME error
                        # instead of the typed-decision refusal it asserts on
                        # (matching the sibling precedent's declaration in
                        # test_committed_lab_overlays_render_omn18570.py).
                        "backend_id": "local-embedding",
                        "model_name": "text-embedding-qwen3",
                        "endpoint_url_env": "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
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


@pytest.mark.parametrize(
    "overlay_path",
    [_DEV_OVERLAY, _CLOUD_OVERLAY],
    ids=["lab-locale-dev-overlay", "cloud-locale-onex-dev-overlay"],
)
def test_typed_decision_backend_renders_verbatim_through_committed_overlay(
    overlay_path: Path, tmp_path: Path
) -> None:
    """The typed-decision backend's URL passes through untouched by either the
    real lab overlay (which binds ``local-coder``) or the real cloud overlay
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


@pytest.mark.parametrize(
    "overlay_path",
    [_DEV_OVERLAY, _CLOUD_OVERLAY],
    ids=["lab-locale-dev-overlay", "cloud-locale-onex-dev-overlay"],
)
def test_bare_base_typed_decision_endpoint_is_refused_through_committed_overlay(
    overlay_path: Path, tmp_path: Path
) -> None:
    """NEGATIVE CONTROL: a typed-decision endpoint that names only an API
    version, not its own operation, is refused end to end through the real
    committed overlay — proving the OMN-19432 admission check fires inside the
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
