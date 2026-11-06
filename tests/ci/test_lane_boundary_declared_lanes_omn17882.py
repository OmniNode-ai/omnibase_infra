# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A lane the lab lists for a host is accepted by that host's boundary tests.

The host boundary tests read the lanes on the host from the lab host registry
(OMN-22152) rather than naming them. This exercises the reader over a neutral
registry, so it runs without the private overlay: an extra lane listed for the
host is read.
"""

from __future__ import annotations

import pytest

from omnibase_infra.models.lane_census import ModelLaneCensusRegistry
from tests.helpers.lane_census_fixtures import (
    lanes_declared_on_host,
)

pytestmark = pytest.mark.ci


def _registry() -> ModelLaneCensusRegistry:
    return ModelLaneCensusRegistry.from_rows(
        [
            {"name": "host-a", "aliases": [], "lanes": ["shared-lane", "a-only-lane"]},
            {
                "name": "host-b",
                "aliases": [],
                "lanes": ["own-lane", "extra-lane", "shared-lane"],
            },
        ]
    )


def test_extra_lane_listed_for_the_host_is_read_from_the_registry() -> None:
    assert lanes_declared_on_host(_registry(), "host-b") == [
        "extra-lane",
        "own-lane",
        "shared-lane",
    ]


def test_lane_listed_for_another_host_alone_is_not_read() -> None:
    assert "a-only-lane" not in lanes_declared_on_host(_registry(), "host-b")
