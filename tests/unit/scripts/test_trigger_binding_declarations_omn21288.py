# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The rebuild publisher accepts typed client declarations without rerouting."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from scripts.trigger_rebuild_on_merge import ModelCiBusOverlay

pytestmark = pytest.mark.unit

_LANE = {
    "broker": "broker.example.invalid:19092",
    "security_protocol": "SASL_PLAINTEXT",
    "sasl_mechanism": "SCRAM-SHA-256",
    "binding_fallback": ["inmemory"],
    "secret_resolver": {
        "enable_convention_fallback": False,
        "mappings": [
            {
                "logical_name": "GITHUB_TOKEN",
                "source": {
                    "source_type": "infisical",
                    "source_path": "/shared/env/GITHUB_TOKEN",
                },
            }
        ],
    },
}


def test_binding_declarations_preserve_publisher_transport() -> None:
    overlay = ModelCiBusOverlay.model_validate(
        {"default": "inmemory", "lanes": {"dev": copy.deepcopy(_LANE)}}
    )
    lane = overlay.lanes["dev"]
    assert lane.binding_fallback == ("inmemory",)
    assert lane.broker == _LANE["broker"]
    assert lane.security_protocol == "SASL_PLAINTEXT"
    assert lane.sasl_mechanism == "SCRAM-SHA-256"
    assert lane.secret_resolver is not None
    assert not lane.secret_resolver.enable_convention_fallback
    assert lane.secret_resolver.mappings[0].logical_name == "GITHUB_TOKEN"
    assert lane.secret_resolver.mappings[0].source.source_type == "infisical"


@pytest.mark.parametrize("invalid", ["fallback", "resolver", "source", "lane"])
def test_binding_declarations_still_refuse_unknown_fields(invalid: str) -> None:
    lane = copy.deepcopy(_LANE)
    if invalid == "fallback":
        lane["binding_fallback"] = ["inmemroy"]
    elif invalid == "resolver":
        lane["secret_resolver"]["enable_convention_falback"] = True
    elif invalid == "source":
        lane["secret_resolver"]["mappings"][0]["source"]["source_pth"] = "/wrong"
    else:
        lane["unknown_binding"] = True
    with pytest.raises(ValidationError):
        ModelCiBusOverlay.model_validate(
            {"default": "inmemory", "lanes": {"dev": lane}}
        )


def test_binding_declarations_remain_optional() -> None:
    overlay = ModelCiBusOverlay.model_validate(
        {"default": "inmemory", "lanes": {"local": {"broker": "inmemory"}}}
    )
    assert overlay.lanes["local"].binding_fallback == ()
    assert overlay.lanes["local"].secret_resolver is None
