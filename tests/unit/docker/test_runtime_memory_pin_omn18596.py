# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime image installs omninode-memory 0.18.4 or later.

The plugin pin cascade raised the ``omninode-memory`` floor in
``docker/Dockerfile.runtime`` after omnimemory released v0.18.4. The test reads
the shipped pin and fails while the floor is still below 0.18.4. A later
cascade that raises the floor again keeps it passing.

Ticket: OMN-18596
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_MEMORY_PIN_RE = re.compile(
    r'"omninode-memory>=(?P<floor>\d+\.\d+\.\d+),<(?P<ceiling>\d+\.\d+\.\d+)"'
)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_runtime_image_memory_floor_is_at_least_0_18_4(dockerfile_path: Path) -> None:
    pins = _MEMORY_PIN_RE.findall(dockerfile_path.read_text(encoding="utf-8"))
    assert len(pins) == 1, f"expected one omninode-memory pin, found {pins}"
    floor, ceiling = pins[0]
    assert _version(floor) >= (0, 18, 4), floor
    assert _version(floor) < _version(ceiling), (floor, ceiling)
