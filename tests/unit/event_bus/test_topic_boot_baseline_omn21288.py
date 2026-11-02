# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A boot that provisions more topic specs than the recorded baseline fails (OMN-21288).

The dev lane's runtime image logged ``total: 2300`` topic specs at boot on
2026-11-01 and creates every one of them in every process, whether or not the
process ever serves a message on them. The count is held in
``config/boot_topic_specs.yaml``; this file builds the real provisioning list the
way the kernel does (``TopicProvisioner`` over the repository's contracts and
standalone manifests) and compares it with that record.

The planted spec is declared in a real standalone manifest discovered by the
constructor at boot, so the failing case is the boot path and not a parallel
list. Each failing case has its passing control beside it, and the
record is changed in a temporary data file (no code edit) to show the verdict
follows the data.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.models.event_bus.model_topic_boot_baseline import (
    ModelTopicBootBaseline,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_FILE = REPO_ROOT / "config" / "boot_topic_specs.yaml"
INFRA_SRC = REPO_ROOT / "src" / "omnibase_infra"
PLANTED_TOPIC = (
    "onex.evt.t29-planted.extra-spec.v1"  # onex-topic-allow: planted fixture
)


def _boot_spec_count(
    caplog: pytest.LogCaptureFixture, extra_manifest_root: Path | None = None
) -> int:
    """Build the boot list and read its count from the provisioner's boot log."""
    roots = [INFRA_SRC / "cli", INFRA_SRC / "services"]
    if extra_manifest_root is not None:
        roots.append(extra_manifest_root)
    first_record = len(caplog.records)
    with caplog.at_level(logging.INFO, logger=TopicProvisioner.__module__):
        TopicProvisioner(
            bootstrap_servers="localhost:9092",
            contracts_root=INFRA_SRC / "nodes",
            skill_manifests_roots=roots,
        )
    counts = [
        match[1]
        for record in caplog.records[first_record:]
        if (
            match := re.search(
                r"topic provisioning \(contract-first\).*total: (\d+),",
                record.getMessage(),
            )
        )
    ]
    assert len(counts) == 1, "expected one provisioning count from this boot"
    return int(counts[0])


def _boot_spec_count_with_planted_spec(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> int:
    """Discover an extra spec through the real manifest extraction path."""
    (tmp_path / "topics.yaml").write_text(
        yaml.safe_dump({"topics": [PLANTED_TOPIC]}), encoding="utf-8"
    )
    return _boot_spec_count(caplog, tmp_path)


def _baseline_file_with(tmp_path: Path, repo_specs: int) -> Path:
    document = yaml.safe_load(BASELINE_FILE.read_text(encoding="utf-8"))
    document["lanes"]["dev"]["repo_specs"] = repo_specs
    path = tmp_path / "boot_topic_specs.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def baseline() -> ModelTopicBootBaseline:
    return ModelTopicBootBaseline.from_yaml_file(BASELINE_FILE, lane="dev")


@pytest.fixture
def boot_spec_count(caplog: pytest.LogCaptureFixture) -> int:
    return _boot_spec_count(caplog)


def test_boot_list_is_within_the_recorded_baseline(
    baseline: ModelTopicBootBaseline, boot_spec_count: int
) -> None:
    """AC-T29-2 control: the unmodified boot list does not exceed the record."""
    baseline.check_repo_specs(boot_spec_count)
    assert baseline.repo_specs_over(boot_spec_count) == 0, (
        f"boot provisioning builds {boot_spec_count} topic specs, above the "
        f"recorded {baseline.repo_specs} in config/boot_topic_specs.yaml. A contract "
        "added a topic that every boot now creates; remove it, or raise the record "
        "in the change that means to add it."
    )


def test_committed_baseline_equals_the_boot_list(
    baseline: ModelTopicBootBaseline, boot_spec_count: int
) -> None:
    """The committed figure is exact, so one planted spec is always over it."""
    assert baseline.repo_specs == boot_spec_count


def test_planted_extra_spec_at_boot_fails_the_committed_baseline(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    baseline: ModelTopicBootBaseline,
    boot_spec_count: int,
) -> None:
    """AC-T29-2: one extra spec on the boot path exceeds the committed record."""
    planted_count = _boot_spec_count_with_planted_spec(tmp_path, caplog)

    assert planted_count == boot_spec_count + 1
    assert baseline.repo_specs_over(boot_spec_count) == 0
    assert baseline.repo_specs_over(planted_count) == 1
    with pytest.raises(ValueError, match="exceeding the recorded"):
        baseline.check_repo_specs(planted_count)


def test_duplicate_manifest_topic_does_not_grow_the_boot_list(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, boot_spec_count: int
) -> None:
    document = yaml.safe_load((INFRA_SRC / "cli" / "topics.yaml").read_text())
    (tmp_path / "topics.yaml").write_text(
        yaml.safe_dump({"topics": [document["topics"][0]]}), encoding="utf-8"
    )
    assert _boot_spec_count(caplog, tmp_path) == boot_spec_count


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
    at_count.check_repo_specs(boot_spec_count)
    with pytest.raises(ValueError, match="exceeding the recorded"):
        one_below.check_repo_specs(boot_spec_count)


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


@pytest.mark.parametrize(
    "text",
    [
        "schema_version: '1'\nlanes:\n  dev: 5\n",
        "schema_version:\nlanes: {}\n",
        "- a\n- b\n",
        "a: [unclosed\n",
    ],
)
def test_malformed_documents_raise_value_error(tmp_path: Path, text: str) -> None:
    path = tmp_path / "boot_topic_specs.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        ModelTopicBootBaseline.from_yaml_file(path, lane="dev")


def test_missing_file_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ModelTopicBootBaseline.from_yaml_file(tmp_path / "absent.yaml", lane="dev")


def test_measured_at_must_be_a_commit_sha(tmp_path: Path) -> None:
    document = yaml.safe_load(BASELINE_FILE.read_text(encoding="utf-8"))
    document["lanes"]["dev"]["repo_specs_measured_at"] = "yesterday"
    path = tmp_path / "boot_topic_specs.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValidationError):
        ModelTopicBootBaseline.from_yaml_file(path, lane="dev")


@pytest.mark.parametrize(
    "read_at",
    [
        "2026-99-01T23:29:24Z",
        "2026-02-29T23:29:24Z",
        "2026-11-01T24:29:24Z",
        "2026-11-01T23:29:24+01:00",
    ],
)
def test_read_time_refuses_invalid_dates_and_non_utc_values(
    baseline: ModelTopicBootBaseline, read_at: str
) -> None:
    record = baseline.model_dump()
    record["read_at"] = read_at
    with pytest.raises(ValidationError):
        ModelTopicBootBaseline.model_validate(record)


def test_read_time_accepts_a_valid_leap_day(baseline: ModelTopicBootBaseline) -> None:
    record = baseline.model_dump()
    record["read_at"] = "2024-02-29T23:29:24Z"
    assert ModelTopicBootBaseline.model_validate(record).read_at == record["read_at"]
