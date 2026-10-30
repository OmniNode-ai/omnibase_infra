# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The laptop bundle's main kernel publishes the runtime tick (OMN-19988).

The local environment observer runs on the runtime tick, which only a ``main``
kernel whose lane sets ``ONEX_RUNTIME_SCHEDULER_ENABLED`` publishes. These tests
pin that the ``local`` render turns the producer on for exactly one kernel, and
that no other bundle gains the switch, since the producer is opt-in per lane.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.runtime.runtime_profile import resolve_runtime_scheduler_enabled
from omnibase_infra.runtime.runtime_scheduler import TICK_PRODUCER_PROFILE

pytestmark = pytest.mark.unit

_CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")
_FLAG = "ONEX_RUNTIME_SCHEDULER_ENABLED"
_MAIN_KERNEL = "omninode-runtime"
_LAPTOP_BUNDLES = ("local", "local-market", "local-lane-mirror")


def _services(bundle: str) -> dict[str, dict[str, object]]:
    compose = generate_compose(
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([bundle])
    )
    services = compose["services"]
    assert isinstance(services, dict)
    return services


def _env(service: dict[str, object]) -> dict[str, str]:
    env = service.get("environment", {})
    assert isinstance(env, dict)
    return env


@pytest.mark.parametrize("bundle", _LAPTOP_BUNDLES)
def test_laptop_main_kernel_turns_the_tick_producer_on(
    bundle: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _env(_services(bundle)[_MAIN_KERNEL])
    assert env["RUNTIME_PROFILE"] == TICK_PRODUCER_PROFILE
    # The kernel's own reader decides, so a misspelt value fails here, not at boot.
    monkeypatch.setenv(_FLAG, env[_FLAG])
    assert resolve_runtime_scheduler_enabled() is True


@pytest.mark.parametrize("bundle", _LAPTOP_BUNDLES)
def test_only_the_main_kernel_owns_the_tick_in_the_laptop_render(bundle: str) -> None:
    """Two producers on one bus would double every tick.

    A service without its own ``command`` boots the image's kernel, and a kernel
    with no ``RUNTIME_PROFILE`` resolves to ``main``.
    """
    owners = sorted(
        name
        for name, service in _services(bundle).items()
        if _env(service).get(_FLAG) == "true"
        and "command" not in service
        and (_env(service).get("RUNTIME_PROFILE") or TICK_PRODUCER_PROFILE)
        == TICK_PRODUCER_PROFILE
    )
    assert owners == [_MAIN_KERNEL]


def test_no_other_bundle_gains_the_tick_switch() -> None:
    """The producer is opt-in per lane; lab lanes set it in their own overlay."""
    bundles = yaml.safe_load((Path(_CATALOG_DIR) / "bundles.yaml").read_text())
    carriers = sorted(
        bundle
        for bundle in bundles
        if bundle not in _LAPTOP_BUNDLES
        and any(_FLAG in _env(service) for service in _services(bundle).values())
    )
    assert carriers == []
