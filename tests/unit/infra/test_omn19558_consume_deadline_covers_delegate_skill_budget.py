# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19558 AC1 -- the consume deadline never undercuts the delegate-skill wait.

node_delegate_skill_orchestrator declares max_handler_duration_seconds 240 and
its handler waits that plus a 60 s terminal margin. A runtime whose Kafka poll
interval is the 300 s library default applies a 255 s dispatch deadline, so the
consume loop dead-letters the command before the handler can publish its
terminal. Every runtime service of every catalog bundle must run a config whose
effective dispatch deadline is at least the handler budget plus the margin.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)

pytestmark = pytest.mark.unit

_CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")
_HANDLER_BUDGET_SECONDS = 240.0
_TERMINAL_DELIVERY_MARGIN_SECONDS = 60.0
_REQUIRED_DEADLINE = _HANDLER_BUDGET_SECONDS + _TERMINAL_DELIVERY_MARGIN_SECONDS
_CONSUMING_SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")


def _bundle_names() -> list[str]:
    raw = yaml.safe_load(Path(_CATALOG_DIR, "bundles.yaml").read_text())
    return sorted(raw)


def _cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for bundle in _bundle_names():
        try:
            resolved = CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([bundle])
        except Exception:  # noqa: BLE001 -- a bundle that cannot resolve alone
            continue
        cases.extend(
            (bundle, svc) for svc in _CONSUMING_SERVICES if svc in resolved.manifests
        )
    return cases


def test_the_matrix_covers_the_laptop_profile() -> None:
    """Positive control: the case list is not empty and includes `local`."""
    assert ("local", "runtime-effects") in _cases()


@pytest.mark.parametrize(("bundle", "service"), _cases())
def test_effective_dispatch_deadline_covers_handler_budget_plus_margin(
    bundle: str, service: str
) -> None:
    manifest = (
        CatalogResolver(catalog_dir=_CATALOG_DIR).resolve([bundle]).manifests[service]
    )
    env = {**manifest.operational_defaults, **manifest.hardcoded_env}
    kwargs: dict[str, object] = {}
    if "KAFKA_MAX_POLL_INTERVAL_MS" in env:
        kwargs["max_poll_interval_ms"] = int(env["KAFKA_MAX_POLL_INTERVAL_MS"])
    config = ModelKafkaEventBusConfig(**kwargs)
    assert config.effective_dispatch_deadline_seconds >= _REQUIRED_DEADLINE
