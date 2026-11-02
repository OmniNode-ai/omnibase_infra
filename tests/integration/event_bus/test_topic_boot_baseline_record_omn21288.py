# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The committed boot topic baseline equals the real provisioning list (OMN-21288).

Builds the provisioning list through ``TopicProvisioner`` over this repository's
real contracts and standalone manifests, and compares it with the committed
``config/boot_topic_specs.yaml``. Both suites read the committed record, so a
record that drifted from the real list (up or down) fails. The count comes from
the boot log rather than private provisioner members.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.models.event_bus.model_topic_boot_baseline import (
    ModelTopicBootBaseline,
)

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_FILE = REPO_ROOT / "config" / "boot_topic_specs.yaml"
INFRA_SRC = REPO_ROOT / "src" / "omnibase_infra"


def test_committed_repo_specs_equals_the_real_boot_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    baseline = ModelTopicBootBaseline.from_yaml_file(BASELINE_FILE, lane="dev")
    with caplog.at_level(logging.INFO, logger=TopicProvisioner.__module__):
        TopicProvisioner(
            bootstrap_servers="localhost:9092",
            contracts_root=INFRA_SRC / "nodes",
            skill_manifests_roots=[INFRA_SRC / "cli", INFRA_SRC / "services"],
        )
    counts = re.findall(
        r"topic provisioning \(contract-first\).*total: (\d+),", caplog.text
    )
    assert len(counts) == 1, "expected one provisioning count from this boot"
    boot_count = int(counts[0])
    baseline.check_repo_specs(boot_count)

    assert boot_count == baseline.repo_specs, (
        f"config/boot_topic_specs.yaml records repo_specs={baseline.repo_specs} but "
        f"the real boot builds {boot_count}; update the record in this change."
    )
