# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-25194: the dev lane's omninode-runtime log holds at least 26 hours.

The hourly integration pass reads merged changes running on the lane from the
``omninode-runtime`` container log. The base file's json-file options, 10m x 3,
held 18 to 27 minutes of it, so an hourly or daily node run never appeared and
its line stayed NOT RUNNING.

MEASURED VOLUME (the dev lane host, 2026-11-17, read with ``docker logs
--timestamps omninode-runtime``): 101,974 lines and 17,573,347 bytes of log
text over 16.4 minutes (03:38:59 to 03:55:25). The json-file envelope adds about
72 bytes a line (``{"log":"...\\n","stream":"stderr","time":"<30 chars>"}``), so
the on-disk volume is about 22 MB over that window, 79.4 MB an hour. The floor
below takes it as 80 MB an hour.

json-file drops its oldest file whole when ``max-file`` is reached, so the
window it GUARANTEES is ``(max-file - 1) * max-size``, not ``max-file *
max-size``. The test holds that product to the 26 hour volume.

Two layers, as in tests/ci/test_compose_service_memory_limits.py: the static
layer reads the overlay as YAML and needs no docker CLI; the render layer runs
``docker compose config`` over each lane stack that layers the overlay, so
compose's own merge decides the effective options, and it skips only when the
docker CLI is absent.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.ci._compose_render_omn21099 import (
    BORROWER_STACKS,
    DEV_LANE_OVERLAY,
    DEV_LANE_STACK,
    docker_compose_available,
    load_compose,
    render_stack,
)

pytestmark = [pytest.mark.unit]

SERVICE = "omninode-runtime"
MEASURED_MB_PER_HOUR = 80
FLOOR_HOURS = 26
FLOOR_MB = MEASURED_MB_PER_HOUR * FLOOR_HOURS

_SIZE_RE = re.compile(r"^(\d+)([kmg]?)$")
_UNIT_MB = {"": 1 / (1024 * 1024), "k": 1 / 1024, "m": 1.0, "g": 1024.0}

LANE_STACKS: dict[str, tuple[str, ...]] = {
    "dev-lane": DEV_LANE_STACK,
    **BORROWER_STACKS,
}


def _megabytes(size: str) -> float:
    match = _SIZE_RE.match(size.strip().lower())
    assert match, f"unreadable json-file max-size {size!r}"
    return int(match.group(1)) * _UNIT_MB[match.group(2)]


def _guaranteed_window_mb(logging: dict[str, Any]) -> float:
    assert logging.get("driver") == "json-file", logging
    options = logging.get("options") or {}
    assert "max-size" in options and "max-file" in options, (
        f"{SERVICE} has no bounded json-file rotation: {logging}"
    )
    max_file = int(options["max-file"])
    assert max_file >= 2, f"max-file {max_file} keeps no rotated file at all"
    return (max_file - 1) * _megabytes(str(options["max-size"]))


def _assert_floor(logging: dict[str, Any], where: str) -> None:
    window = _guaranteed_window_mb(logging)
    assert window >= FLOOR_MB, (
        f"{where}: {SERVICE} logging {logging['options']} guarantees {window:.0f} MB, "
        f"{window / MEASURED_MB_PER_HOUR:.1f} h at {MEASURED_MB_PER_HOUR} MB/h; the "
        f"integration pass needs {FLOOR_HOURS} h = {FLOOR_MB} MB (OMN-25194)"
    )


def test_overlay_declares_runtime_log_retention_above_the_floor() -> None:
    services = load_compose(DEV_LANE_OVERLAY)["services"]
    logging = services[SERVICE].get("logging")
    assert logging, f"{DEV_LANE_OVERLAY} gives {SERVICE} no logging of its own"
    _assert_floor(logging, DEV_LANE_OVERLAY)


@pytest.mark.skipif(
    not docker_compose_available(), reason="docker compose CLI is not installed"
)
@pytest.mark.parametrize("lane", sorted(LANE_STACKS))
def test_rendered_runtime_log_retention_is_above_the_floor(lane: str) -> None:
    services = render_stack(LANE_STACKS[lane])
    if SERVICE not in services:
        pytest.skip(f"{lane} renders no {SERVICE}")
    _assert_floor(services[SERVICE]["logging"], lane)


def test_floor_rejects_the_retention_that_held_twenty_minutes() -> None:
    """Positive control: the base file's 10m x 3 is below the floor."""
    with pytest.raises(AssertionError, match="guarantees 20 MB"):
        _assert_floor(
            {"driver": "json-file", "options": {"max-size": "10m", "max-file": "3"}},
            "base",
        )
