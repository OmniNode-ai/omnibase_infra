# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19144 -- the declared lane must survive all the way onto the wire.

The unit gate beside this one (``tests/unit/scripts/
test_omn19144_dev_lane_runtime_lane_identity.py``) proves the dev lane overlay
declares ``ONEX_RUNTIME_LANE`` and that the resolver accepts the spelling. That
is one end of the seam. It would still pass if the monitor stopped stamping the
field onto the event, or if the field stopped serialising onto the payload the
consumer actually reads -- and either of those reproduces the original defect
exactly, because the consumer's only signal is a key that is absent.

So this exercises the whole seam in one pass, against the real components: the
lane value is read out of the compose overlay on disk, put into the process
environment the way the deployment does, and a real ``ServiceRuntimeHealthMonitor``
is run for one cycle. The assertion is on the SERIALISED payload, because
``payload.get("lane")`` on the bare event dict is precisely what the lane-keyed
projection does, and a field that exists on the model but not on the wire is
indistinguishable from the failure this ticket is about.

No broker and no database. ``run_once`` is driven with no event bus and empty
bootstrap servers, which is a supported shape -- the broker-dependent dimensions
grade themselves degraded and the cycle still returns its event, in about a
second. That matters for more than speed: a test that needs a service would be
deselected or skipped on a runner, and a gate that does not run where the merge
happens is not a gate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.health import runtime_lane_identity
from omnibase_infra.runtime.health.runtime_lane_identity import (
    ENV_RUNTIME_LANE,
    KNOWN_LANES,
)
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_LANE_OVERLAY = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
LANE_SPEAKING_SERVICE = "omninode-runtime"


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


def _declared_lane() -> str:
    """The lane the dev-lane overlay declares for its health-emitting service.

    Read from the file rather than pinned here on purpose. A literal would make
    this test agree with itself while the deployment said something else, which
    is the one disagreement it exists to catch.
    """
    with open(DEV_LANE_OVERLAY, encoding="utf-8") as handle:
        # S506: _TolerantLoader subclasses SafeLoader; the only widening is a
        # multi-constructor for compose's tags, which resolve to plain
        # mappings/sequences/scalars. No arbitrary object can be built.
        parsed = yaml.load(handle, Loader=_TolerantLoader)  # noqa: S506
    environment = parsed["services"][LANE_SPEAKING_SERVICE]["environment"]
    return str(environment[ENV_RUNTIME_LANE])


def _monitor() -> ServiceRuntimeHealthMonitor:
    """One offline monitor, boot grace off so a cycle is not suppressed."""
    return ServiceRuntimeHealthMonitor(
        event_bus=None,
        bootstrap_servers="",
        check_interval_seconds=60.0,
        boot_grace_seconds=0.0,
    )


@pytest.mark.asyncio
async def test_the_declared_lane_reaches_the_serialised_health_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deployment -> environment -> monitor -> wire, with nothing stubbed."""
    lane = _declared_lane()
    assert lane in KNOWN_LANES, (
        f"the overlay declares {ENV_RUNTIME_LANE}={lane!r}, which the resolver "
        f"refuses: it is not in {sorted(KNOWN_LANES)}. A refused value reaches "
        "the consumer as no value at all."
    )
    monkeypatch.setenv(ENV_RUNTIME_LANE, lane)

    event = await _monitor().run_once()

    assert event.lane == lane, (
        f"the monitor emitted lane={event.lane!r} while the deployment declared "
        f"{lane!r}; the stamp between the resolver and the event is broken"
    )
    payload = event.model_dump(mode="json")
    assert payload.get("lane") == lane, (
        "the lane is on the model but not on the serialised payload. A "
        "lane-keyed consumer reads this dict and nothing else, so an absent "
        "key here is the same total silent loss as an unset variable "
        f"(payload keys: {sorted(payload)})"
    )


@pytest.mark.asyncio
async def test_without_the_declaration_the_payload_carries_no_lane(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The pre-fix state, pinned as the negative control.

    This is the whole defect reproduced through the real emitter: with nothing
    declared, the event is valid, the cycle succeeds, the payload is complete
    in every other respect -- and the one key a lane-keyed consumer needs is
    null. Nothing raises. Pinning it here means the assertion above has a
    referent: it distinguishes "the lane made it through" from "this test would
    pass either way".

    It also holds the emitter's side of the honesty repair: the process that
    cannot name its lane says so, once, rather than emitting in silence.
    """
    monkeypatch.delenv(ENV_RUNTIME_LANE, raising=False)
    runtime_lane_identity._warn_absent_lane.cache_clear()

    try:
        with caplog.at_level(logging.WARNING):
            event = await _monitor().run_once()
    finally:
        runtime_lane_identity._warn_absent_lane.cache_clear()

    assert event.lane is None, (
        "an undeclared lane produced a lane on the event -- something is "
        "guessing, and a guessed lane puts one deployment's verdict on "
        "another lane's row"
    )
    assert event.model_dump(mode="json").get("lane") is None

    announced = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and ENV_RUNTIME_LANE in record.getMessage()
    ]
    assert announced, (
        "the monitor emitted a lane-less health event and nothing in the "
        f"process mentioned {ENV_RUNTIME_LANE}. That silence is what let this "
        "run undetected: the consumer drops the event and cannot tell a "
        "lane-less emitter from a lane it does not hold, and the emitter is "
        "the only thing in a position to say which it was."
    )
