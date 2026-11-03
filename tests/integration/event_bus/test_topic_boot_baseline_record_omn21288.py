# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every lane in the committed boot topic baseline loads as a typed record (OMN-21288).

Reads ``config/boot_topic_specs.yaml`` as shipped, the way a gate adapter would,
and loads each lane it declares through ``ModelTopicBootBaseline``. The unit
suite proves the boot list equals the record for ``dev``; this proves the
shipped file as a whole is loadable, so a lane added with a malformed record
fails here rather than at its first use.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.models.event_bus.model_topic_boot_baseline import (
    ModelTopicBootBaseline,
)

pytestmark = [pytest.mark.integration]

BASELINE_FILE = Path(__file__).resolve().parents[3] / "config" / "boot_topic_specs.yaml"


def test_every_declared_lane_loads_as_a_typed_record() -> None:
    lanes = yaml.safe_load(BASELINE_FILE.read_text(encoding="utf-8"))["lanes"]
    assert "dev" in lanes
    for lane in lanes:
        record = ModelTopicBootBaseline.from_yaml_file(BASELINE_FILE, lane=lane)
        assert record.lane == lane
        assert record.repo_specs <= record.specs_per_boot
