# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24474: the dev-201 lane takes only a deliberate promotion while frozen."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from deploy_agent.routing import is_promotion, lane_flags_for_instance

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
THIRD_ROLL = datetime(2026, 11, 14, 18, 0, tzinfo=UTC)
FLAGS = lane_flags_for_instance(REPO_ROOT, "dev-201")


def test_committed_table_freezes_the_dev_lane() -> None:
    freeze = FLAGS.freeze
    assert freeze is not None
    assert "OMN-24474" in freeze.reason
    assert freeze.until.tzinfo is not None
    assert freeze.until > THIRD_ROLL


def test_the_freeze_keeps_the_other_dev_201_flags() -> None:
    assert FLAGS.max_dev_lag_minutes == 20
    assert "gateway-deploy" in FLAGS.disabled_phases


def test_a_merge_driven_command_is_frozen_before_expiry_and_promotion_is_not() -> None:
    freeze = FLAGS.freeze
    assert freeze is not None
    mid_freeze = FLAGS.frozen_reason(datetime(2026, 11, 14, 19, 0, tzinfo=UTC))
    assert mid_freeze is not None and "FROZEN" in mid_freeze
    assert FLAGS.frozen_reason(freeze.until + timedelta(seconds=1)) is None
    assert is_promotion("promotion/operator") is True
    assert is_promotion("gha/omnibase_infra/pr-1") is False
