# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for learning store adapter."""

from datetime import UTC, datetime

import pytest

from omnibase_infra.services.agent_learning_extraction.adapter_learning_store import (
    compute_freshness_score,
)


@pytest.mark.unit
class TestFreshnessScore:
    def test_today_is_1(self) -> None:
        now = datetime.now(tz=UTC)
        assert compute_freshness_score(now, now) == pytest.approx(1.0)

    def test_one_week_decays(self) -> None:
        from datetime import timedelta

        now = datetime.now(tz=UTC)
        one_week_ago = now - timedelta(weeks=1)
        score = compute_freshness_score(one_week_ago, now)
        assert 0.85 < score < 0.95  # ~10% decay per week

    def test_four_weeks_decays_more(self) -> None:
        from datetime import timedelta

        now = datetime.now(tz=UTC)
        four_weeks_ago = now - timedelta(weeks=4)
        score = compute_freshness_score(four_weeks_ago, now)
        assert 0.55 < score < 0.70  # ~40% total decay

    def test_never_below_zero(self) -> None:
        from datetime import timedelta

        now = datetime.now(tz=UTC)
        ancient = now - timedelta(weeks=52)
        score = compute_freshness_score(ancient, now)
        assert score >= 0.0
