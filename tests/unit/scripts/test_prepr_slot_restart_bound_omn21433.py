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


class _TagTolerantLoader(yaml.SafeLoader):
    """Load the overlay without resolving compose's ``!override``/``!reset``."""


def _passthrough(loader: Any, _suffix: str, node: Any) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_TagTolerantLoader.add_multi_constructor("!", _passthrough)
_TagTolerantLoader.add_multi_constructor("tag:yaml.org,2002:merge", _passthrough)


def _services() -> dict[str, Any]:
    text = re.sub(r"\$\{[A-Z_]+[^}]*\}", "1", OVERLAY.read_text(encoding="utf-8"))
    loader = _TagTolerantLoader(text)
    try:
        return loader.get_single_data()["services"]
    finally:
        loader.dispose()


@pytest.mark.parametrize("name", SLOT_SERVICES)
def test_every_slot_service_has_a_bounded_restart_policy(name: str) -> None:
    policy = str(_services()[name].get("restart", ""))
    assert re.fullmatch(r"on-failure:[1-5]", policy), (
        f"{name} restart policy is {policy!r}. The slot inherits unless-stopped "
        f"from the dev-lane overlay, which loops a failing service forever on "
        f"the shared Postgres (OMN-21433)."
    )
