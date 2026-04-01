# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for baseline passthrough verification [OMN-6774]."""

from __future__ import annotations

import os

import pytest

from omnibase_infra.services.eval.baseline_passthrough import (
    ONEX_FEATURE_FLAGS,
    get_flag_states,
    is_baseline_mode,
    verify_baseline_passthrough,
)


@pytest.mark.unit
class TestGetFlagStates:
    def test_returns_all_flags(self) -> None:
        states = get_flag_states()
        for flag in ONEX_FEATURE_FLAGS:
            assert flag in states

    def test_truthy_values(self) -> None:
        flag = ONEX_FEATURE_FLAGS[0]
        os.environ[flag] = "true"
        states = get_flag_states()
        assert states[flag] is True
        os.environ.pop(flag, None)

    def test_falsy_values(self) -> None:
        flag = ONEX_FEATURE_FLAGS[0]
        os.environ[flag] = "false"
        states = get_flag_states()
        assert states[flag] is False
        os.environ.pop(flag, None)


@pytest.mark.unit
class TestIsBaselineMode:
    def test_baseline_when_all_off(self) -> None:
        for flag in ONEX_FEATURE_FLAGS:
            os.environ[flag] = "false"
        assert is_baseline_mode() is True
        for flag in ONEX_FEATURE_FLAGS:
            os.environ.pop(flag, None)

    def test_not_baseline_when_one_on(self) -> None:
        for flag in ONEX_FEATURE_FLAGS:
            os.environ[flag] = "false"
        os.environ[ONEX_FEATURE_FLAGS[0]] = "true"
        assert is_baseline_mode() is False
        for flag in ONEX_FEATURE_FLAGS:
            os.environ.pop(flag, None)


@pytest.mark.unit
class TestVerifyBaselinePassthrough:
    def test_returns_verification_dict(self) -> None:
        for flag in ONEX_FEATURE_FLAGS:
            os.environ[flag] = "false"
        result = verify_baseline_passthrough()
        assert result["all_flags_off"] is True
        assert "kafka_bootstrap" in result
        for flag in ONEX_FEATURE_FLAGS:
            os.environ.pop(flag, None)
