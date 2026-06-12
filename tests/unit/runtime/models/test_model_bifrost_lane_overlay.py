# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelBifrostLaneOverlay (OMN-12864).

Validates the typed deployment binding model:
- URL completeness enforcement (OMN-12815)
- env-var mapping
- reject bare-base URLs
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)


def _complete_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1/chat/completions"


def _valid_overlay(**overrides: str) -> dict[str, str]:
    base = {
        "lane": "dev",
        "coder_endpoint_url": _complete_url("192.168.86.201", 8000),
        "reasoner_endpoint_url": _complete_url("192.168.86.201", 8001),
        "embedding_endpoint_url": _complete_url("192.168.86.201", 8100),
        "ds4_flash_endpoint_url": _complete_url("192.168.86.200", 8101),
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_valid_overlay_loads() -> None:
    overlay = ModelBifrostLaneOverlay.model_validate(_valid_overlay())
    assert overlay.lane == "dev"
    assert overlay.coder_endpoint_url == _complete_url("192.168.86.201", 8000)
    assert overlay.reasoner_endpoint_url == _complete_url("192.168.86.201", 8001)
    assert overlay.embedding_endpoint_url == _complete_url("192.168.86.201", 8100)
    assert overlay.ds4_flash_endpoint_url == _complete_url("192.168.86.200", 8101)


@pytest.mark.unit
def test_as_env_dict_returns_correct_keys() -> None:
    overlay = ModelBifrostLaneOverlay.model_validate(_valid_overlay())
    env = overlay.as_env_dict()
    assert set(env.keys()) == {
        "BIFROST_LOCAL_CODER_ENDPOINT_URL",
        "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
        "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
        "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL",
    }
    assert env["BIFROST_LOCAL_CODER_ENDPOINT_URL"] == _complete_url(
        "192.168.86.201", 8000
    )
    assert env["BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL"] == _complete_url(
        "192.168.86.200", 8101
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "bare_url"),
    [
        ("coder_endpoint_url", "http://192.168.86.201:8000"),
        ("coder_endpoint_url", "http://192.168.86.201:8000/"),
        ("coder_endpoint_url", "http://192.168.86.201:8000/v1"),
        ("reasoner_endpoint_url", "http://192.168.86.201:8001/v1"),
        ("embedding_endpoint_url", "http://192.168.86.201:8100/v1"),
        ("ds4_flash_endpoint_url", "http://192.168.86.200:8101"),
    ],
)
def test_bare_base_url_rejected(field: str, bare_url: str) -> None:
    """Bare-base URLs without /chat/completions path must be rejected (OMN-12815)."""
    with pytest.raises(ValidationError, match="endpoint_url must end in"):
        ModelBifrostLaneOverlay.model_validate(_valid_overlay(**{field: bare_url}))


@pytest.mark.unit
def test_alternate_chat_completions_path_accepted() -> None:
    """URLs ending in /chat/completions (without /v1 prefix) are also valid."""
    overlay = ModelBifrostLaneOverlay.model_validate(
        _valid_overlay(coder_endpoint_url="http://192.168.86.201:8000/chat/completions")
    )
    assert overlay.coder_endpoint_url == "http://192.168.86.201:8000/chat/completions"


@pytest.mark.unit
def test_empty_lane_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelBifrostLaneOverlay.model_validate(_valid_overlay(lane=""))


@pytest.mark.unit
def test_empty_url_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelBifrostLaneOverlay.model_validate(_valid_overlay(coder_endpoint_url=""))


@pytest.mark.unit
def test_extra_fields_rejected() -> None:
    """Extra fields must be forbidden (strict schema)."""
    with pytest.raises(ValidationError):
        ModelBifrostLaneOverlay.model_validate(
            {**_valid_overlay(), "unexpected_field": "value"}
        )
