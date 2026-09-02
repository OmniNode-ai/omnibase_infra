# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17531: the three lab lanes bind strict auto-wiring; prod and judge do not.

``ONEX_WIRING_STRICT_MODE`` is ``${ONEX_WIRING_STRICT_MODE:-0}`` in
``docker-compose.infra.yml`` and, before OMN-17531, no lane overlay bound it. At
``0`` a handler the resolver cannot construct is quarantined and the runtime
still reports healthy; at ``1`` it re-raises and kills boot. onex-dev (staging)
binds ``"1"`` on all three runtime Deployments, so every lab lane ran the weaker
semantics and could not reproduce a staging boot failure -- the mechanism behind
OMN-15623, OMN-17502 and OMN-17510, each of which was first seen on staging.

These assertions are the enforcement half of that change. Without them the
binding is a one-time edit that the next overlay refactor can drop silently, and
the lane goes back to reporting healthy while wiring nothing.

Scope is pinned in BOTH directions on purpose:

* dev, lakshman and stability-test MUST bind ``"1"``, as a literal.
* prod (``omnibase-infra-prod``) and judge MUST NOT be bound here. Judge is
  declared read-only and prod is not in OMN-17531's mandate; flipping either
  from this ticket's surface would be scope creep with a live blast radius.
* The base file keeps its ``:-0`` default, which is what makes an unlisted or
  future lane opt IN explicitly rather than inherit strict semantics it was
  never proven under.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = ROOT / "docker"

BASE = DOCKER_DIR / "docker-compose.infra.yml"
DEV_LANE = DOCKER_DIR / "docker-compose.dev-lane.yml"
LAKSHMAN = DOCKER_DIR / "docker-compose.lakshman.yml"
STABILITY = DOCKER_DIR / "docker-compose.stability-test.yml"
PROD = DOCKER_DIR / "docker-compose.prod.yml"
JUDGE = DOCKER_DIR / "docker-compose.judge.yml"

STRICT_KEY = "ONEX_WIRING_STRICT_MODE"

# The runtime services that actually run the auto-wiring kernel. projection-api
# and omninode-contract-resolver are uvicorn servers that never reach
# wire_from_manifest, so the flag is inert on them and is deliberately not bound.
STRICT_LANES: dict[str, tuple[Path, tuple[str, ...]]] = {
    "dev": (DEV_LANE, ("omninode-runtime", "runtime-effects", "runtime-worker")),
    # This lane runs no runtime-worker (docker-compose.lakshman.yml is
    # standalone and declares main + effects only).
    "lakshman": (LAKSHMAN, ("omninode-runtime", "runtime-effects")),
    "stability-test": (
        STABILITY,
        ("omninode-runtime", "runtime-effects", "runtime-worker"),
    ),
}

pytestmark = pytest.mark.unit


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    """Resolve Compose's `!override` / `!reset` tags to their plain value."""
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """Test-local YAML loader with Docker Compose tag support."""


_ComposeLoader.add_constructor("!override", _construct_compose_value)
_ComposeLoader.add_constructor("!reset", _construct_compose_value)


def _load(path: Path) -> dict[str, Any]:
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_ComposeLoader)  # noqa: S506
    assert isinstance(loaded, dict)
    return loaded


def _environment(compose: dict[str, Any], service: str) -> dict[str, Any]:
    services = compose.get("services") or {}
    assert service in services, f"service {service!r} is absent from the overlay"
    return dict((services[service] or {}).get("environment") or {})


@pytest.mark.parametrize("lane", sorted(STRICT_LANES))
def test_lab_lane_runtime_services_bind_strict_wiring(lane: str) -> None:
    """OMN-17531 AC-1: every runtime service on the three lab lanes is strict."""
    path, services = STRICT_LANES[lane]
    compose = _load(path)

    for service in services:
        env = _environment(compose, service)
        assert STRICT_KEY in env, (
            f"{lane}/{service}: {STRICT_KEY} is unbound, so this service falls "
            f"through to the base `:-0` default and quarantines unwireable "
            f"handlers while still reporting healthy (OMN-17531)"
        )
        assert env[STRICT_KEY] == "1", (
            f"{lane}/{service}: {STRICT_KEY} is {env[STRICT_KEY]!r}, not '1'"
        )


@pytest.mark.parametrize("lane", sorted(STRICT_LANES))
def test_lab_lane_strict_wiring_is_literal_not_interpolated(lane: str) -> None:
    """A `${VAR:-1}` form is silently downgradable from an operator's shell.

    Same locus as the OMN-12864 BIFROST_CONTRACT_PATH footgun: an exported
    variable in the deploying shell would propagate into every runtime container
    and return the lane to the quarantining semantics with no diff anywhere.
    """
    path, services = STRICT_LANES[lane]
    compose = _load(path)

    for service in services:
        raw = _environment(compose, service)[STRICT_KEY]
        assert "$" not in str(raw), (
            f"{lane}/{service}: {STRICT_KEY} is interpolated ({raw!r}); an "
            f"ambient shell export would silently downgrade this lane"
        )


@pytest.mark.parametrize("path", [PROD, JUDGE])
def test_prod_and_judge_are_not_bound_by_this_ticket(path: Path) -> None:
    """OMN-17531 is explicitly scoped away from prod and judge.

    Judge is declared read-only in the lane map; prod is not touched. If either
    is ever moved to strict mode it must be its own ticket with its own boot
    evidence, not a side effect of a lab-lane change.
    """
    compose = _load(path)
    for name, service in (compose.get("services") or {}).items():
        env = dict((service or {}).get("environment") or {})
        assert STRICT_KEY not in env, (
            f"{path.name}/{name} binds {STRICT_KEY}; prod and judge are out of "
            f"OMN-17531's scope and need their own ticket and boot evidence"
        )


def test_base_keeps_the_quarantining_default() -> None:
    """The opt-in stays opt-in.

    Flipping the base default to `1` would move prod and judge too, and would
    make every future lane strict before anyone has booted it once. The base
    default is what forces a new lane to make the choice explicitly.
    """
    assert f"{STRICT_KEY}: ${{{STRICT_KEY}:-0}}" in BASE.read_text(encoding="utf-8")
