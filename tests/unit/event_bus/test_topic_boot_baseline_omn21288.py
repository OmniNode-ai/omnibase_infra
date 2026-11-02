# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A boot that provisions more topic specs than the recorded baseline fails (OMN-21288).

The dev lane's runtime image logged ``total: 2300`` topic specs at boot on
2026-11-01 and creates every one of them in every process, whether or not the
process ever serves a message on them. The count is held in
``config/boot_topic_specs.yaml``; this file builds the real provisioning list the
way the kernel does (``TopicProvisioner`` over the repository's contracts and
standalone manifests) and compares it with that record.

The planted spec goes through ``_build_topic_specs``, the method the
constructor calls at boot, so the failing case is the boot path and not a
parallel list. Each failing case has its passing control beside it, and the
record is changed in a temporary data file (no code edit) to show the verdict
follows the data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.models.event_bus.model_topic_boot_baseline import (
    ModelTopicBootBaseline,
)
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_FILE = REPO_ROOT / "config" / "boot_topic_specs.yaml"
INFRA_SRC = REPO_ROOT / "src" / "omnibase_infra"
PLANTED_TOPIC = (
    "onex.evt.t29-planted.extra-spec.v1"  # onex-topic-allow: planted fixture
)


def _boot_spec_count() -> int:
    """Build the provisioning list the way the kernel does and count it."""
    provisioner = TopicProvisioner(
        bootstrap_servers="localhost:9092",
        contracts_root=INFRA_SRC / "nodes",
        skill_manifests_roots=[INFRA_SRC / "cli", INFRA_SRC / "services"],
    )
    return len(provisioner._topic_specs)


def _boot_spec_count_with_planted_spec() -> int:
    """The same boot, with one extra spec appended by ``_build_topic_specs``."""
    real_build = TopicProvisioner._build_topic_specs

    def build_plus_one(self: TopicProvisioner) -> tuple[ModelTopicSpec, ...]:
        return (*real_build(self), ModelTopicSpec(suffix=PLANTED_TOPIC))

    with patch.object(TopicProvisioner, "_build_topic_specs", build_plus_one):
        return _boot_spec_count()


def _baseline_file_with(tmp_path: Path, repo_specs: int) -> Path:
    document = yaml.safe_load(BASELINE_FILE.read_text(encoding="utf-8"))
    document["lanes"]["dev"]["repo_specs"] = repo_specs
    path = tmp_path / "boot_topic_specs.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def baseline() -> ModelTopicBootBaseline:
    return ModelTopicBootBaseline.from_yaml_file(BASELINE_FILE, lane="dev")


@pytest.fixture(scope="module")
def boot_spec_count() -> int:
    return _boot_spec_count()


def test_boot_list_is_within_the_recorded_baseline(
    baseline: ModelTopicBootBaseline, boot_spec_count: int
) -> None:
    """AC-T29-2 control: the unmodified boot list does not exceed the record."""
    assert baseline.repo_specs_over(boot_spec_count) == 0, (
        f"boot provisioning builds {boot_spec_count} topic specs, above the "
        f"recorded {baseline.repo_specs} in config/boot_topic_specs.yaml. A contract "
        "added a topic that every boot now creates; remove it, or raise the record "
        "in the change that means to add it."
    )


def test_planted_extra_spec_at_boot_fails_the_check(
    baseline: ModelTopicBootBaseline, boot_spec_count: int
) -> None:
    """AC-T29-2: one extra spec on the boot path puts the list over the record."""
    planted_count = _boot_spec_count_with_planted_spec()

    assert planted_count == boot_spec_count + 1
    assert baseline.repo_specs_over(boot_spec_count) == 0
    assert baseline.repo_specs_over(planted_count) == 1


def test_changing_the_declared_value_changes_the_verdict_without_a_code_edit(
    tmp_path: Path, boot_spec_count: int
) -> None:
    """AC-T29-3: the same boot list passes or fails by the data file alone."""
    at_count = ModelTopicBootBaseline.from_yaml_file(
        _baseline_file_with(tmp_path, boot_spec_count), lane="dev"
    )
    one_below = ModelTopicBootBaseline.from_yaml_file(
        _baseline_file_with(tmp_path, boot_spec_count - 1), lane="dev"
    )

    assert at_count.repo_specs_over(boot_spec_count) == 0
    assert one_below.repo_specs_over(boot_spec_count) == 1


def test_baseline_record_is_typed_and_refuses_unknown_keys(tmp_path: Path) -> None:
    document = yaml.safe_load(BASELINE_FILE.read_text(encoding="utf-8"))
    document["lanes"]["dev"]["extra_figure"] = 1
    path = tmp_path / "boot_topic_specs.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValidationError):
        ModelTopicBootBaseline.from_yaml_file(path, lane="dev")


def test_baseline_record_carries_its_read_time_and_source(
    baseline: ModelTopicBootBaseline,
) -> None:
    assert baseline.read_at.endswith("Z")
    assert "topic provisioning" in baseline.source
    assert baseline.specs_per_boot >= baseline.repo_specs


def test_repo_list_larger_than_the_image_list_is_refused(tmp_path: Path) -> None:
    document = yaml.safe_load(BASELINE_FILE.read_text(encoding="utf-8"))
    document["lanes"]["dev"]["repo_specs"] = (
        document["lanes"]["dev"]["specs_per_boot"] + 1
    )
    path = tmp_path / "boot_topic_specs.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValidationError):
        ModelTopicBootBaseline.from_yaml_file(path, lane="dev")


def test_undeclared_lane_is_refused() -> None:
    with pytest.raises(ValueError, match="no topic baseline for lane"):
        ModelTopicBootBaseline.from_yaml_file(BASELINE_FILE, lane="nonexistent")
