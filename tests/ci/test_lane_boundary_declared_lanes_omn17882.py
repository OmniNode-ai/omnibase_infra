# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A lane the lab declares for a host is accepted by that host's boundary tests.

The host boundary tests read the lanes on the host from the lab manifest rather
than naming them. This exercises the reader over a neutral manifest, so it runs
without the private overlay: an extra lane declared for the host is read.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers.lane_census_fixtures import (
    lanes_declared_on_host,
)

pytestmark = pytest.mark.ci


def _manifest() -> dict[str, Any]:
    return {
        "lanes": {
            "own-lane": {"hosts": ["host-b"]},
            "extra-lane": {"hosts": ["host-b"]},
            "shared-lane": {"hosts": ["host-a", "host-b"]},
            "a-only-lane": {"hosts": ["host-a"]},
        }
    }


def test_extra_lane_declared_for_the_host_is_read_from_the_manifest() -> None:
    assert lanes_declared_on_host(_manifest(), "host-b") == [
        "extra-lane",
        "own-lane",
        "shared-lane",
    ]


def test_lane_declared_for_another_host_alone_is_not_read() -> None:
    assert "a-only-lane" not in lanes_declared_on_host(_manifest(), "host-b")
