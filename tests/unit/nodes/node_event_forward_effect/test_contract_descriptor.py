# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the event-forward contract-owned endpoint descriptor."""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_event_forward_effect.contract_descriptor import (
    contract_event_forward_backend_url,
)

pytestmark = pytest.mark.unit


def test_resolves_backend_url_from_contract_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handler receives its endpoint solely through the contract overlay."""
    endpoint = "https://event-backend.example.invalid"
    monkeypatch.setenv("EVENT_FORWARD_BACKEND_URL", endpoint)

    assert contract_event_forward_backend_url() == endpoint


def test_fails_closed_when_backend_url_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset endpoint must not fall back to an undeclared local service."""
    monkeypatch.delenv("EVENT_FORWARD_BACKEND_URL", raising=False)

    with pytest.raises(ValueError, match=r"descriptor\.backend_url resolved empty"):
        contract_event_forward_backend_url()
