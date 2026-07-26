# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the OMN-15181 prod-lane redeploy dispatch locality guard.

Live-verified 2026-07-26 (ledger 09:05Z / 09:25Z): the prod redpanda broker's
advertised external listener is a raw-LAN address that only routes from
``omninode-pc`` itself. A prod-lane ``node_redeploy_orchestrator`` dispatch
from any other host silently passes bootstrap and then hangs on the
produce-path leader reconnect until the Kafka flush times out. This guard
fails fast, client-side, before that dispatch is attempted.
"""

from __future__ import annotations

import click
import pytest

from omnibase_infra.cli.prod_dispatch_locality_guard import (
    REQUIRED_PROD_DISPATCH_HOSTNAME,
    enforce_prod_dispatch_locality,
)


def test_noop_for_non_redeploy_node(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only node_redeploy_orchestrator is guarded; every other node passes."""
    monkeypatch.setattr(
        "omnibase_infra.cli.prod_dispatch_locality_guard.socket.gethostname",
        lambda: "some-macbook.local",
    )
    enforce_prod_dispatch_locality(
        "node_compliance_sweep", {"runtime_lane": "prod"}
    )  # must not raise


def test_noop_for_dev_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev-lane redeploy dispatch is never restricted by locality."""
    monkeypatch.setattr(
        "omnibase_infra.cli.prod_dispatch_locality_guard.socket.gethostname",
        lambda: "some-macbook.local",
    )
    enforce_prod_dispatch_locality(
        "node_redeploy_orchestrator", {"runtime_lane": "dev"}
    )  # must not raise


def test_noop_for_stability_test_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stability-test-lane redeploy dispatch is never restricted by locality."""
    monkeypatch.setattr(
        "omnibase_infra.cli.prod_dispatch_locality_guard.socket.gethostname",
        lambda: "some-macbook.local",
    )
    enforce_prod_dispatch_locality(
        "node_redeploy_orchestrator", {"runtime_lane": "stability-test"}
    )  # must not raise


def test_noop_when_lane_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing runtime_lane field (never expected in practice) fails open."""
    monkeypatch.setattr(
        "omnibase_infra.cli.prod_dispatch_locality_guard.socket.gethostname",
        lambda: "some-macbook.local",
    )
    enforce_prod_dispatch_locality("node_redeploy_orchestrator", {})  # must not raise


def test_passes_on_the_required_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prod-lane dispatch running on omninode-pc itself is allowed through."""
    monkeypatch.setattr(
        "omnibase_infra.cli.prod_dispatch_locality_guard.socket.gethostname",
        lambda: REQUIRED_PROD_DISPATCH_HOSTNAME,
    )
    enforce_prod_dispatch_locality(
        "node_redeploy_orchestrator", {"runtime_lane": "prod"}
    )  # must not raise


def test_blocks_prod_lane_from_the_wrong_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prod-lane dispatch from any other host fails closed with a clear message."""
    monkeypatch.setattr(
        "omnibase_infra.cli.prod_dispatch_locality_guard.socket.gethostname",
        lambda: "Jonahs-MacBook-Pro.local",
    )
    with pytest.raises(click.ClickException) as excinfo:
        enforce_prod_dispatch_locality(
            "node_redeploy_orchestrator", {"runtime_lane": "prod"}
        )
    message = str(excinfo.value)
    assert REQUIRED_PROD_DISPATCH_HOSTNAME in message
    assert "Jonahs-MacBook-Pro.local" in message
    assert "OMN-15181" in message


def test_blocks_prod_lane_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lane matching is case-insensitive -- 'Prod' must not slip past the guard."""
    monkeypatch.setattr(
        "omnibase_infra.cli.prod_dispatch_locality_guard.socket.gethostname",
        lambda: "some-macbook.local",
    )
    with pytest.raises(click.ClickException):
        enforce_prod_dispatch_locality(
            "node_redeploy_orchestrator", {"runtime_lane": "Prod"}
        )
