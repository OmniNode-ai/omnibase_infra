# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Each expansion of the node-announcement topic template passes core's topic validator (OMN-21590, A2 AC6)."""

from __future__ import annotations

import pytest

from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.models.node_announce.model_node_announcement_topic import (
    ModelNodeAnnouncementTopic,
)
from omnibase_infra.runtime.runtime_profile import _PROFILES


@pytest.mark.unit
@pytest.mark.parametrize("workload", sorted(_PROFILES))
def test_every_workload_expansion_passes_the_topic_validator(workload: str) -> None:
    suffix = ModelNodeAnnouncementTopic.suffix_for(workload)

    result = validate_topic_suffix(suffix)

    assert result.is_valid, result.error
    assert suffix == ModelNodeAnnouncementTopic.TEMPLATE.format(workload=workload)
    assert suffix.count(".") == 4


@pytest.mark.unit
def test_the_workload_sits_in_the_event_name_segment() -> None:
    parts = ModelNodeAnnouncementTopic.suffix_for("projection-api").split(".")
    assert parts == [
        "onex",
        "evt",
        "platform",
        "node-announcement-from-projection-api",
        "v1",
    ]


@pytest.mark.unit
def test_the_profile_set_is_not_empty() -> None:
    """Positive control for the parametrisation above."""
    assert len(_PROFILES) > 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "workload",
    ["", "  ", "Has Capitals", "dot.ted", "-leading", "a_b", " main", "main "],
)
def test_planted_violation_a_workload_that_is_not_kebab_case_is_refused(
    workload: str,
) -> None:
    with pytest.raises(ValueError):
        ModelNodeAnnouncementTopic.suffix_for(workload)
