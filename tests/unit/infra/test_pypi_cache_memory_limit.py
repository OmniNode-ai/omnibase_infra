# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pypi-cache service must hold a memory floor that covers its measured working set (OMN-19998).

`devpi-server` is the pull-through PyPI cache on the main lab host. Its
resident-plus-swapped working set exceeded a 2g `mem_limit`. Measured on 2026-11-13: the container
sat at its 2g limit (cgroup peak 2049 MB against a 2048 MB cap), its swap peak
hit the 2048 MB swap cap, it logged 18210 limit hits in 21 h, and 51.8 GB of
anon pages were read back from swap. The working set read 2.7 GB resident
plus swapped, so 4g clears it with headroom; the host holds more than 60 GB
available. This test pins the floor so a future edit cannot drop the limit back
below the measured working set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_FILE = (
    Path(__file__).resolve().parents[3] / "docker" / "docker-compose.pypi-cache.yml"
)

SERVICE_NAME = "omninode-pypi-cache"


def _docker_size_to_bytes(value: object) -> int:
    """Convert a docker size string to bytes.

    Accepts digits followed by one of b, k, m, g (case-insensitive) or plain
    digits as bytes. Docker compose reports `mem_limit` / `memswap_limit` in
    this form (e.g. `4g`, `2048m`, `2097152000`).
    """
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TypeError(f"cannot convert {value!r} to a docker size in bytes")
    text = value.strip()
    if not text:
        raise ValueError("empty docker size string")
    multiplier = 1
    suffix = text[-1].lower()
    if suffix in {"b", "k", "m", "g"}:
        multiplier = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[suffix]
        text = text[:-1]
    return int(text) * multiplier


@pytest.mark.unit
def test_pypi_cache_memory_limit_covers_measured_working_set() -> None:
    """`mem_limit` must be at least 4 GiB to cover the measured working set."""
    with open(COMPOSE_FILE) as f:
        compose = yaml.safe_load(f)

    service = compose["services"][SERVICE_NAME]

    mem_limit_bytes = _docker_size_to_bytes(service["mem_limit"])

    assert mem_limit_bytes >= 4 * 1024**3, (
        f"{SERVICE_NAME} `mem_limit` is {service['mem_limit']} ({mem_limit_bytes} bytes), "
        "below the 4 GiB floor. OMN-19998 measured on 2026-11-13: the "
        "container ran at its 2g limit (cgroup peak 2049 MB against a 2048 MB cap, "
        "18210 limit hits in 21 h, 51.8 GB of anon pages read back from swap). The "
        "measured working set is about 2.7 GB resident plus swapped, so 4g is the floor."
    )

    if "memswap_limit" in service:
        memswap_limit_bytes = _docker_size_to_bytes(service["memswap_limit"])
        assert memswap_limit_bytes >= mem_limit_bytes, (
            f"{SERVICE_NAME} `memswap_limit` ({service['memswap_limit']}) must be at least "
            f"`mem_limit` ({service['mem_limit']})."
        )
