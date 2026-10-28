# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21433: a pre-PR slot service that cannot start stops instead of looping.

Measured on the lab host on 2026-10-28. The slot's projection writers inherit
``restart: unless-stopped`` from the dev-lane overlay. Three of them exit on
start in a slot because the standalone binding they open expects the dev
principal ``omninode_runtime`` and the slot's DSN names ``omninode_runtime_<slot>``.
The slot's savings writer had restarted 9047 times, and the log of that one
container held 172978 refusals ``too many connections for role`` on the slot's
intelligence role: every restart opens its pools and its schema stamp on the dev
lane's shared Postgres, against a slot role that carries a budget of 6. The budget did its job (the refusals were the slot's own), but a
loop with no end is a standing load on the shared server and a slot that never
reports its failure.

A bounded policy turns the loop into a finding: three restarts, then the container
stays exited, ``docker ps`` shows it, and the slot's health probe fails on it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERLAY = REPO_ROOT / "docker" / "docker-compose.prepr.yml"

SLOT_SERVICES = (
    "omninode-runtime",
    "runtime-effects",
    "runtime-worker",
    "projection-api",
    "tenant-projection-writer",
    "projection-tenant-registry-writer",
    "projection-delegation-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-live-events-writer",
    "onex-api",
)


# The slot is rendered by prepr_verify_lane.sh from these three files, in this
# order; a later file's ``restart`` replaces an earlier one's, and ``!reset``
# drops it back to the default (no restart).
COMPOSE_CHAIN = (
    REPO_ROOT / "docker" / "docker-compose.infra.yml",
    REPO_ROOT / "docker" / "docker-compose.dev-lane.yml",
    OVERLAY,
)

_RESET = object()


class _TagTolerantLoader(yaml.SafeLoader):
    """Load a compose file, marking ``!reset`` and passing other tags through."""


def _tag(loader: Any, suffix: str, node: Any) -> Any:
    if suffix == "reset":
        return _RESET
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_TagTolerantLoader.add_multi_constructor("!", _tag)


def _services(path: Path) -> dict[str, Any]:
    text = re.sub(r"\$\{[A-Z_]+[^}]*\}", "1", path.read_text(encoding="utf-8"))
    loader = _TagTolerantLoader(text)
    try:
        return loader.get_single_data().get("services") or {}
    finally:
        loader.dispose()


def _rendered_restart(name: str) -> object:
    """The ``restart`` value the chain renders for ``name``, last file wins."""
    policy: object = None
    for path in COMPOSE_CHAIN:
        service = _services(path).get(name)
        if not isinstance(service, dict) or "restart" not in service:
            continue
        value = service["restart"]
        policy = None if value is _RESET else value
    return policy


@pytest.mark.parametrize("name", SLOT_SERVICES)
def test_every_slot_service_has_a_bounded_restart_policy(name: str) -> None:
    policy = _rendered_restart(name)
    assert policy == "on-failure:3", (
        f"{name} renders restart policy {policy!r} from the infra, dev-lane and "
        f"prepr chain. The slot must stop after three failed restarts instead of "
        f"looping on the shared Postgres (OMN-21433)."
    )
