# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The per-workload hot-loading topics and their declared specs (OMN-21226, H1).

Every expansion of the applied and state templates for every lane workload passes
core's topic validator, and the state topic is declared compacted.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from omnibase_core.constants.constants_runtime_profiles import (
    REGISTERED_RUNTIME_PROFILES,
)
from omnibase_core.validation import validate_topic_suffix
from omnibase_infra.models.contract_hotload.model_contract_hotload_workload_topic import (
    ModelContractHotloadWorkloadTopic,
)
from omnibase_infra.runtime.runtime_profile import load_runtime_profile

pytestmark = pytest.mark.unit

# Profiles are routing identities, while catalog service names identify deployed
# workloads. Exercise both sets rather than treating a private profile map as
# the workload inventory.
_CATALOG = Path(__file__).resolve().parents[4] / "docker" / "catalog" / "services"
_LANE_WORKLOADS = {
    service["name"]
    for path in _CATALOG.glob("*.yaml")
    if (service := yaml.safe_load(path.read_text(encoding="utf-8")))["layer"]
    == "runtime"
}
_TOPIC_WRITERS = sorted(_LANE_WORKLOADS | REGISTERED_RUNTIME_PROFILES)


@pytest.mark.parametrize("workload", _TOPIC_WRITERS)
@pytest.mark.parametrize(
    "expand",
    [
        ModelContractHotloadWorkloadTopic.applied,
        ModelContractHotloadWorkloadTopic.state,
    ],
    ids=["applied", "state"],
)
def test_every_workload_expansion_passes_the_topic_validator(
    workload: str, expand: Callable[[str], ModelContractHotloadWorkloadTopic]
) -> None:
    topic = expand(workload)

    result = validate_topic_suffix(topic.suffix)

    assert result.is_valid, result.error
    assert topic.suffix.count(".") == 4
    assert workload in topic.suffix.split(".")[3]


def test_the_profile_set_is_not_empty() -> None:
    """Positive control for the parametrisation above."""
    assert "omninode-runtime" in _LANE_WORKLOADS
    assert "runtime-effects" in _LANE_WORKLOADS
    assert "main" in REGISTERED_RUNTIME_PROFILES


@pytest.mark.parametrize("profile", sorted(REGISTERED_RUNTIME_PROFILES))
def test_registered_profiles_resolve_through_the_public_api(profile: str) -> None:
    assert load_runtime_profile(profile).name == profile


def test_the_workload_sits_in_the_event_name_segment() -> None:
    assert ModelContractHotloadWorkloadTopic.state("projection-api").suffix.split(
        "."
    ) == ["onex", "evt", "platform", "contract-hotload-state-from-projection-api", "v1"]
    assert ModelContractHotloadWorkloadTopic.applied("projection-api").suffix.split(
        "."
    ) == [
        "onex",
        "evt",
        "platform",
        "contract-hotload-applied-from-projection-api",
        "v1",
    ]


@pytest.mark.parametrize("workload", _TOPIC_WRITERS)
def test_the_state_topic_is_declared_compacted(workload: str) -> None:
    spec = ModelContractHotloadWorkloadTopic.state(workload).spec

    assert spec.kafka_config is not None
    assert spec.kafka_config["cleanup.policy"] == "compact"


@pytest.mark.parametrize("workload", _TOPIC_WRITERS)
def test_the_applied_topic_is_declared_time_retained(workload: str) -> None:
    spec = ModelContractHotloadWorkloadTopic.applied(workload).spec

    assert spec.kafka_config is not None
    assert spec.kafka_config["cleanup.policy"] == "delete"


@pytest.mark.parametrize(
    "workload",
    [
        "",
        "  ",
        "Has Capitals",
        "dot.ted",
        "-leading",
        "trailing-",
        "two--parts",
        "a_b",
        " main",
        "main ",
        "main\n",
        "UPPER",
        "café",
    ],
)
@pytest.mark.parametrize("channel", ["applied", "state"])
def test_planted_violation_a_workload_that_is_not_kebab_case_is_refused(
    workload: str,
    channel: str,
) -> None:
    with pytest.raises(ValueError):
        ModelContractHotloadWorkloadTopic(workload=workload, channel=channel)
