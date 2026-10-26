# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21279: the dev broker holds a memory reservation on the shared host.

Why this exists. On 2026-10-26 the host (94 GB, swap 57 GB used) ran the dev
Redpanda broker with no memory reservation (``memory.low`` 0) and 10.8 GB of its
arena in swap. Lane sessions were not the source: the lane cgroup
(``user.slice``) refaulted 0.5 MB of anonymous pages in 10 s against 55 MB for
``system.slice``, and a lane runs with ``MemorySwapMax=0``. What protects a
container from having its resident pages swapped out for another cgroup's is its
reservation, so these tests pin one on the dev lane. The runtime containers
already carry a 256 MiB reservation from ``docker/catalog/services``, a layer
other lanes share, so raising it is a separate change.

The reservation lives in ``docker-compose.dev-lane.yml`` and not in the base
``docker-compose.infra.yml``: every other lane merges the base (see
``test_redpanda_memory_per_lane_dial_omn19085``), and a base-file value would
reach their brokers at their next recreate.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.infra.test_broker_profile_omn19419 import _render

_GIB = 1024**3
_DEV_BROKER_FLOOR = 4 * _GIB


def _broker(lane: str) -> dict[str, Any]:
    service: dict[str, Any] = _render(lane)["services"]["redpanda"]
    return service


@pytest.mark.unit
@pytest.mark.lab_overlay
def test_dev_broker_holds_a_reservation_covering_its_resident_set() -> None:
    """The broker's non-reclaimable peak was about 4.4 GB; its RSS 1.7 to 2.2 GB."""
    reservation = int(_broker("dev").get("mem_reservation", 0))

    assert reservation >= _DEV_BROKER_FLOOR, (
        f"dev broker reservation is {reservation} bytes; it needs at least "
        f"{_DEV_BROKER_FLOOR} so its resident pages are not swapped out for a "
        "lane or a build"
    )


@pytest.mark.unit
@pytest.mark.lab_overlay
def test_base_file_broker_carries_no_reservation() -> None:
    """A reservation in the base file would land on every other lane's broker."""
    assert "mem_reservation" not in _broker("infra"), (
        "docker-compose.infra.yml declares a broker memory reservation; it "
        "belongs in docker-compose.dev-lane.yml"
    )
