# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-24562 AC15: the tracked cpu-boost-off unit carries its ordering lines."""

from __future__ import annotations

from pathlib import Path

import pytest

UNIT = Path(__file__).resolve().parents[2] / "deploy/systemd/cpu-boost-off.service"


def _section(name: str) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw in UNIT.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
        elif current == name and line and not line.startswith("#"):
            lines.append(line)
    return lines


@pytest.mark.unit
def test_unit_orders_after_sysinit_target() -> None:
    assert "After=sysinit.target" in _section("Unit")


@pytest.mark.unit
def test_unit_is_wanted_by_multi_user_target() -> None:
    assert "WantedBy=multi-user.target" in _section("Install")


@pytest.mark.unit
def test_unit_is_a_oneshot_writing_boost_zero() -> None:
    service = _section("Service")
    assert "Type=oneshot" in service
    assert any("echo 0 > /sys/devices/system/cpu/cpufreq/boost" in s for s in service)
