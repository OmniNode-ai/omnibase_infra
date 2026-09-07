# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18012 -- binding a credential on a container nobody recreates is inert.

Phase B binds ``KAFKA_SASL_*`` on all 15 dev-lane Kafka clients in
``docker/docker-compose.dev-lane.yml``. A compose ``environment:`` entry only
takes effect when the container is RECREATED, and the only sanctioned path that
recreates anything on this lane is ``scripts/deploy-runtime.sh --execute``,
which restarts exactly the services its own arrays name.

``context-audit-consumer`` was in NEITHER array. It is declared in the base
compose file (so it is not "dev-lane only") and it is absent from
``RUNTIME_SERVICES`` (so it is in no lane's restart set at all). On the dev lane
it had been up 2 weeks across every intervening deploy, measured 2026-09-07.
That was harmless while the broker accepted PLAINTEXT. After the flip it is the
precise failure this ticket exists to remove: a client that keeps speaking
PLAINTEXT to a listener that now refuses it, invisible because the deploy
reports success.

This test is the RED guard for that class in general, not for one service name:
it derives the client set from the compose overlay and the restart set from the
script, and asserts the second covers the first.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DEV_LANE_OVERLAY = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"

AUTH_KEY = "KAFKA_SASL_USERNAME"


class _TolerantLoader(yaml.SafeLoader):
    """compose uses `!override` / `!!merge`, which SafeLoader refuses."""


_TolerantLoader.add_multi_constructor(  # type: ignore[no-untyped-call]
    "",
    lambda loader, suffix, node: (
        loader.construct_mapping(node)
        if isinstance(node, yaml.MappingNode)
        else (
            loader.construct_sequence(node)
            if isinstance(node, yaml.SequenceNode)
            else loader.construct_scalar(node)
        )
    ),
)


def _authenticated_dev_lane_clients() -> set[str]:
    """Services the overlay gives broker credentials to."""
    with open(DEV_LANE_OVERLAY, encoding="utf-8") as handle:
        # S506: _TolerantLoader subclasses SafeLoader; the only widening is a
        # multi-constructor for compose's tags, which resolve to plain
        # mappings/sequences/scalars. No arbitrary object can be built.
        parsed = yaml.load(handle, Loader=_TolerantLoader)  # noqa: S506
    services = dict(parsed.get("services") or {})
    return {
        name
        for name, body in services.items()
        if isinstance(body, dict)
        and isinstance(body.get("environment"), dict)
        and AUTH_KEY in body["environment"]
    }


def _bash_array(name: str) -> set[str]:
    """Read a `readonly NAME=( ... )` array out of the deploy script."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    match = re.search(rf"^readonly {re.escape(name)}=\((.*?)^\)", text, re.M | re.S)
    assert match is not None, f"{name} not found in {DEPLOY_SCRIPT.name}"
    return {
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _dev_lane_restart_set() -> set[str]:
    return (
        _bash_array("RUNTIME_SERVICES")
        | _bash_array("DEV_LANE_ONLY_RUNTIME_SERVICES")
        | _bash_array("DEV_LANE_EXTRA_BROKER_CLIENTS")
    )


def test_positive_control_the_arrays_and_the_overlay_both_parse() -> None:
    """Every assertion below is a set difference. An empty set on either side
    is a vacuous pass, so prove both sides are populated first."""
    clients = _authenticated_dev_lane_clients()
    restarts = _dev_lane_restart_set()
    assert len(clients) >= 15, (
        f"only {len(clients)} authenticated clients parsed out of the overlay; "
        "the coverage assertion below would be near-vacuous"
    )
    assert len(restarts) >= 15, (
        f"only {len(restarts)} services parsed out of the deploy script arrays"
    )


def test_every_authenticated_dev_lane_client_is_in_the_deploy_restart_set() -> None:
    clients = _authenticated_dev_lane_clients()
    uncovered = sorted(clients - _dev_lane_restart_set())
    assert not uncovered, (
        f"these dev-lane services are given broker credentials but no deploy "
        f"path recreates them: {uncovered}. The credential never reaches the "
        "process; each keeps speaking PLAINTEXT to a listener that refuses it, "
        "and the deploy reports success. Add them to "
        "DEV_LANE_EXTRA_BROKER_CLIENTS in scripts/deploy-runtime.sh."
    )


def test_the_extra_array_is_appended_only_on_the_dev_lane_branch() -> None:
    """Blast radius: prod, stability-test and judge restart sets must be
    byte-identical to before this change."""
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    usages = [
        line for line in text.splitlines() if "DEV_LANE_EXTRA_BROKER_CLIENTS[@]" in line
    ]
    assert len(usages) == 1, (
        f"DEV_LANE_EXTRA_BROKER_CLIENTS is expanded {len(usages)} times; it "
        "must be expanded exactly once, in the dev branch of "
        "resolve_lane_runtime_services"
    )
    dev_branch = text[
        text.index("lane_only_services=(") : text.index(
            "docker-compose.stability-test.yml)"
        )
    ]
    assert "DEV_LANE_EXTRA_BROKER_CLIENTS[@]" in dev_branch, (
        "the extra broker clients are appended outside the dev-lane branch, so "
        "another lane would be handed a service name to restart"
    )
