# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolution-equivalence tests for the vector-store Qdrant endpoint descriptor.

OMN-13558 Wave-1 endpoint→overlay migration. Proves the overlay-resolved
``descriptor.qdrant_url`` returns exactly the value the old direct
``os.environ["QDRANT_URL"]`` read returned for the same env, across dev /
stability / prod lane values, and that resolution fails closed when the var is
unset (no silent ``http://localhost:6333`` fallback).
"""

from __future__ import annotations

import os

import pytest

from omnibase_infra.nodes.node_vector_store_effect.contract_descriptor import (
    contract_qdrant_url,
)

pytestmark = pytest.mark.unit


# Representative per-lane QDRANT_URL values (the same shape an operator overlay /
# the per-lane service env supplies). Dev, stability-test, and prod each point at
# a distinct endpoint; the overlay must resolve each identically to a raw env read.
_LANE_ENDPOINTS = [
    "http://localhost:6333",  # dev
    "http://qdrant.stability-test.svc:6333",  # stability-test
    "http://qdrant.prod.svc:6333",  # prod
]


@pytest.mark.parametrize("endpoint", _LANE_ENDPOINTS)
def test_overlay_resolution_equals_direct_env_read(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlay descriptor resolves the same value the old env read produced."""
    monkeypatch.setenv("QDRANT_URL", endpoint)

    # The value the pre-migration code read directly.
    direct = os.environ["QDRANT_URL"]
    # The value the migrated overlay seam resolves.
    resolved = contract_qdrant_url()

    assert resolved == direct == endpoint


def test_fails_closed_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset QDRANT_URL raises rather than defaulting to localhost."""
    monkeypatch.delenv("QDRANT_URL", raising=False)

    with pytest.raises(ValueError, match=r"descriptor\.qdrant_url resolved empty"):
        contract_qdrant_url()


def test_fails_closed_when_env_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only QDRANT_URL is treated as unset and fails closed."""
    monkeypatch.setenv("QDRANT_URL", "   ")

    with pytest.raises(ValueError, match=r"descriptor\.qdrant_url resolved empty"):
        contract_qdrant_url()


def test_qdrant_handler_config_default_factory_uses_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelQdrantHandlerConfig.url default resolves through the same overlay seam."""
    from omnibase_infra.handlers.models.qdrant import ModelQdrantHandlerConfig

    endpoint = "http://qdrant.example:6333"
    monkeypatch.setenv("QDRANT_URL", endpoint)

    config = ModelQdrantHandlerConfig()
    assert config.url == endpoint
