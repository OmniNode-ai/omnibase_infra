# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime image pins each plugin at or above its recorded floor.

The plugin pin cascade rewrites the plugin pins in ``docker/Dockerfile.runtime``
and records the resulting floors in
``tests/ci/fixtures/runtime_plugin_pin_floors_omn20864.yaml``. This test reads
both: it passes when every recorded plugin has exactly one pin whose floor is at
least the recorded one and below its ceiling, and fails on the merge base's
Dockerfile once the cascade has raised a floor. A later cascade that raises a
floor again keeps it passing.

Ticket: OMN-20864
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO / "docker" / "Dockerfile.runtime"
_FLOORS = (
    _REPO / "tests" / "ci" / "fixtures" / "runtime_plugin_pin_floors_omn20864.yaml"
)
_PIN_RE = re.compile(
    r'"(?P<package>omninode-[A-Za-z0-9-]+)'
    r"(?:>=(?P<floor>\d+\.\d+\.\d+),<(?P<ceiling>\d+\.\d+\.\d+)|==(?P<exact>\d+\.\d+\.\d+))"
    r'"'
)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _recorded() -> dict[str, str]:
    floors = yaml.safe_load(_FLOORS.read_text(encoding="utf-8"))["floors"]
    assert floors, "the floors fixture records no plugin"
    return {str(package): str(floor) for package, floor in floors.items()}


@pytest.mark.parametrize("package", sorted(_recorded()))
def test_runtime_image_pins_the_plugin_at_or_above_its_recorded_floor(
    package: str,
) -> None:
    pins = [
        m
        for m in _PIN_RE.finditer(_DOCKERFILE.read_text(encoding="utf-8"))
        if m["package"] == package
    ]
    assert len(pins) == 1, f"expected one {package} pin, found {len(pins)}"
    pin = pins[0]
    floor = pin["floor"] or pin["exact"]
    assert _version(floor) >= _version(_recorded()[package]), (package, floor)
    if pin["ceiling"]:
        assert _version(floor) < _version(pin["ceiling"]), (package, floor)
