# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15504 -- the dev lane must not be the only lane on the library default.

``max_poll_interval_ms`` is the deadline aiokafka gives a handler to return to
the poll loop. Miss it and the consumer is evicted mid-handle; the handler then
finishes, its ``OffsetCommit`` is refused with ``UnknownMemberIdError``, the
committed offset never advances, and the same records are redelivered on every
rejoin. That is a livelock, not a slowdown: it does not self-heal, and a
container restart does not clear it because the committed offset lives in the
broker rather than in the container.

judge, stability-test and lakshman have all carried ``1800000`` for some time.
The dev lane was never brought along, so it ran the ``300000`` library default
-- and ``node_delegate_skill_orchestrator``'s worst-case handler duration is its
contract's ``wait_timeout_seconds`` of 300, which landed on that deadline
exactly. On 2026-09-10 the delegation chain livelocked from 11:56:43Z through
four identical ~300 s join-then-evict cycles, and every delegation submitted on
the lab after that instant went undelivered.

This test is the gate for the class, not for one lane's current value: it
derives the runtime-consumer services from each lane overlay and asserts the
dev lane declares the interval on the same services stability-test does, at a
value no smaller. A regression that silently drops the override -- an anchor
rename, a merge list edited without it -- fails here rather than on the lab.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"
DEV_LANE_OVERLAY = DOCKER_DIR / "docker-compose.dev-lane.yml"
STABILITY_OVERLAY = DOCKER_DIR / "docker-compose.stability-test.yml"

POLL_KEY = "KAFKA_MAX_POLL_INTERVAL_MS"

# The services that host auto-wired consumers. Named explicitly rather than
# derived, because "which services run consumers" is the fact under test: a
# derivation that read the same compose file would agree with itself no matter
# what the file said.
RUNTIME_CONSUMER_SERVICES = frozenset(
    {"omninode-runtime", "runtime-effects", "runtime-worker"}
)


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


def _poll_intervals(overlay: Path) -> dict[str, int]:
    """Map service name -> declared poll interval, for services that set one."""
    with open(overlay, encoding="utf-8") as handle:
        # S506: _TolerantLoader subclasses SafeLoader; the only widening is a
        # multi-constructor for compose's tags, which resolve to plain
        # mappings/sequences/scalars. No arbitrary object can be built.
        parsed = yaml.load(handle, Loader=_TolerantLoader)  # noqa: S506
    services = dict(parsed.get("services") or {})
    found: dict[str, int] = {}
    for name, body in services.items():
        if not isinstance(body, dict):
            continue
        env = body.get("environment")
        if isinstance(env, dict) and POLL_KEY in env:
            found[name] = int(str(env[POLL_KEY]))
    return found


def test_dev_lane_declares_the_poll_interval_on_every_consumer_service() -> None:
    """The override must reach all three, not just the one that was noticed."""
    declared = _poll_intervals(DEV_LANE_OVERLAY)
    missing = sorted(RUNTIME_CONSUMER_SERVICES - declared.keys())

    assert not missing, (
        f"dev lane services {missing} do not declare {POLL_KEY}, so they run "
        "aiokafka's 300000 ms default. A handler that outruns it is evicted "
        "mid-handle and its commit is refused with UnknownMemberIdError -- the "
        "livelock that took the .201 delegation chain down on 2026-09-10 "
        "(OMN-15504). Merge *dev_lane_consumer_poll_env into the service's "
        "environment in docker/docker-compose.dev-lane.yml."
    )


def test_dev_lane_poll_interval_is_not_below_the_stability_lane() -> None:
    """The positive control: compare against a lane that already had it right.

    Pinning a literal here would pass just as happily if every lane drifted
    together. Comparing lanes means the assertion still has a referent.
    """
    stability = _poll_intervals(STABILITY_OVERLAY)
    dev = _poll_intervals(DEV_LANE_OVERLAY)

    assert stability, (
        f"no service in {STABILITY_OVERLAY.name} declares {POLL_KEY}; this "
        "test's reference lane is gone, so its comparison proves nothing"
    )
    reference = min(stability.values())

    for service in sorted(RUNTIME_CONSUMER_SERVICES):
        assert service in dev, (
            f"dev lane {service} declares no {POLL_KEY} at all, so it runs the "
            "300000 ms default -- see the sibling test in this module"
        )
        assert dev[service] >= reference, (
            f"dev lane {service} declares {POLL_KEY}={dev[service]}, below the "
            f"stability lane's {reference}. The dev lane is where changes are "
            "proven first, so a shorter deadline there hides an eviction that "
            "staging and prod would not have."
        )
