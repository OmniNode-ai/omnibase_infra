# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolution-equivalence tests for the graph handler Bolt endpoint descriptor.

OMN-13558 Wave-1 endpoint→overlay migration. Proves the overlay-resolved
``descriptor.graph_bolt_uri`` returns exactly the value the old direct
``os.environ["GRAPH_BOLT_URI"]`` read returned for the same env, across dev /
stability / prod lane values, and that resolution fails closed when the var is
unset (no silent ``bolt://localhost:7687`` fallback).
"""

from __future__ import annotations

import os

import pytest

from omnibase_infra.handlers.models.graph.contract_descriptor import (
    contract_graph_bolt_uri,
)

pytestmark = pytest.mark.unit


# Representative per-lane GRAPH_BOLT_URI values (the same shape an operator
# overlay / the per-lane service env supplies). Dev, stability-test, and prod
# each point at a distinct Bolt endpoint; the overlay must resolve each
# identically to a raw env read.
_LANE_ENDPOINTS = [
    "bolt://localhost:7687",  # dev
    "bolt://memgraph.stability-test.svc:7687",  # stability-test
    "bolt://memgraph.prod.svc:7687",  # prod
]


@pytest.mark.parametrize("endpoint", _LANE_ENDPOINTS)
def test_overlay_resolution_equals_direct_env_read(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overlay descriptor resolves the same value the old env read produced."""
    monkeypatch.setenv("GRAPH_BOLT_URI", endpoint)

    # The value the pre-migration code read directly.
    direct = os.environ["GRAPH_BOLT_URI"]
    # The value the migrated overlay seam resolves.
    resolved = contract_graph_bolt_uri()

    assert resolved == direct == endpoint


def test_fails_closed_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset GRAPH_BOLT_URI raises rather than defaulting to localhost."""
    monkeypatch.delenv("GRAPH_BOLT_URI", raising=False)

    with pytest.raises(ValueError, match=r"descriptor\.graph_bolt_uri resolved empty"):
        contract_graph_bolt_uri()


def test_fails_closed_when_env_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only GRAPH_BOLT_URI is treated as unset and fails closed."""
    monkeypatch.setenv("GRAPH_BOLT_URI", "   ")

    with pytest.raises(ValueError, match=r"descriptor\.graph_bolt_uri resolved empty"):
        contract_graph_bolt_uri()


def test_graph_handler_config_default_factory_uses_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelGraphHandlerConfig.uri default resolves through the same overlay seam."""
    from omnibase_infra.handlers.models.graph import ModelGraphHandlerConfig

    endpoint = "bolt://memgraph.example:7687"
    monkeypatch.setenv("GRAPH_BOLT_URI", endpoint)

    config = ModelGraphHandlerConfig()
    assert config.uri == endpoint


def test_graph_handler_config_fails_closed_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelGraphHandlerConfig() with no uri + unset env fails closed."""
    monkeypatch.delenv("GRAPH_BOLT_URI", raising=False)

    with pytest.raises(ValueError, match=r"descriptor\.graph_bolt_uri resolved empty"):
        from omnibase_infra.handlers.models.graph import ModelGraphHandlerConfig

        ModelGraphHandlerConfig()
