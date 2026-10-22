# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19812: every runtime-image container declares its open-file limit.

The dev-lane host's Docker daemon hands containers a soft RLIMIT_NOFILE of 1024
(hard 524288). ``omninode-runtime`` runs one Kafka consumer per subscribed
group -- 437 groups on 2026-10-19 -- at about two broker connections each, so
a few minutes after boot it crossed 1024 and every Kafka connect, file open
and Postgres connect failed with ``[Errno 24] Too many open files``. The kernel
then shut down, the container restarted, and the C28 consumer-flow probe found
the lane ``starting`` past both settle windows and graded it INDETERMINATE.

The limit is a property of the deployment, so it is declared in compose rather
than inherited from whatever the host daemon's default happens to be.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pytest

from tests.unit.infra.test_broker_profile_omn19419 import _render

pytestmark = pytest.mark.unit

# A floor with headroom over the measured need (~1000 fds at 437 groups),
# well under the daemon's hard limit of 524288.
_NOFILE_FLOOR = 65536


@cache
def _services(lane: str) -> dict[str, dict[str, Any]]:
    services: dict[str, dict[str, Any]] = _render(lane)["services"]
    return services


def _runtime_image_services(lane: str) -> list[str]:
    return sorted(
        name
        for name, service in _services(lane).items()
        if str((service.get("build") or {}).get("dockerfile", "")).endswith(
            "Dockerfile.runtime"
        )
    )


@pytest.mark.parametrize("lane", ["infra", "dev"])
def test_lane_renders_the_runtime_kernel(lane: str) -> None:
    # Positive control: the selection below is not vacuously empty.
    assert "omninode-runtime" in _runtime_image_services(lane)


@pytest.mark.parametrize("lane", ["infra", "dev"])
def test_every_runtime_image_service_raises_nofile(lane: str) -> None:
    short: dict[str, object] = {}
    for name in _runtime_image_services(lane):
        nofile = (_services(lane)[name].get("ulimits") or {}).get("nofile")
        soft = nofile.get("soft") if isinstance(nofile, dict) else nofile
        if not isinstance(soft, int) or soft < _NOFILE_FLOOR:
            short[name] = nofile
    assert not short, (
        f"runtime-image services on the {lane} render declare no nofile "
        f"ulimit of at least {_NOFILE_FLOOR} and inherit the daemon's soft "
        f"1024: {short}"
    )
