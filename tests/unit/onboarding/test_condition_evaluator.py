# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for condition_evaluator — OMN-10779, OMN-10769."""

from __future__ import annotations

import pytest

from omnibase_infra.onboarding.condition_evaluator import (
    ConditionEvaluationError,
    evaluate_condition,
)

pytestmark = pytest.mark.unit


class TestNoneCondition:
    def test_none_returns_true(self) -> None:
        assert evaluate_condition(None, {}) is True

    def test_none_ignores_state(self) -> None:
        assert evaluate_condition(None, {"foo": "bar"}) is True


class TestEquality:
    def test_eq_matching_value(self) -> None:
        assert evaluate_condition("env == local", {"env": "local"}) is True

    def test_eq_non_matching_value(self) -> None:
        assert evaluate_condition("env == cloud", {"env": "local"}) is False

    def test_eq_quoted_string_value(self) -> None:
        assert (
            evaluate_condition('mode == "production"', {"mode": "production"}) is True
        )

    def test_eq_unknown_key_raises(self) -> None:
        with pytest.raises(ConditionEvaluationError, match="unknown_key"):
            evaluate_condition("unknown_key == foo", {})


class TestInList:
    def test_in_list_match(self) -> None:
        assert evaluate_condition("env in [local, dev]", {"env": "local"}) is True

    def test_in_list_no_match(self) -> None:
        assert evaluate_condition("env in [local, dev]", {"env": "cloud"}) is False

    def test_in_list_unknown_key_raises(self) -> None:
        with pytest.raises(ConditionEvaluationError, match="env"):
            evaluate_condition("env in [local, dev]", {})

    def test_in_list_single_item(self) -> None:
        assert (
            evaluate_condition("mode in [production]", {"mode": "production"}) is True
        )


class TestNotIn:
    def test_not_in_list_match(self) -> None:
        assert (
            evaluate_condition("env not in [cloud, staging]", {"env": "local"}) is True
        )

    def test_not_in_list_no_match(self) -> None:
        assert evaluate_condition("env not in [local, dev]", {"env": "local"}) is False

    def test_not_in_list_unknown_key_raises(self) -> None:
        with pytest.raises(ConditionEvaluationError, match="env"):
            evaluate_condition("env not in [local, dev]", {})

    def test_not_in_literal_list_true(self) -> None:
        assert (
            evaluate_condition(
                "deployment_mode not in [local, hybrid]",
                {"deployment_mode": "cloud"},
            )
            is True
        )

    def test_not_in_literal_list_false(self) -> None:
        assert (
            evaluate_condition(
                "deployment_mode not in [local, hybrid]",
                {"deployment_mode": "local"},
            )
            is False
        )


class TestInStateKey:
    def test_in_state_key_match(self) -> None:
        assert (
            evaluate_condition(
                "env in allowed_envs",
                {"env": "local", "allowed_envs": ["local", "dev"]},
            )
            is True
        )

    def test_in_state_key_no_match(self) -> None:
        assert (
            evaluate_condition(
                "env in allowed_envs",
                {"env": "cloud", "allowed_envs": ["local", "dev"]},
            )
            is False
        )

    def test_in_state_key_lhs_unknown_raises(self) -> None:
        with pytest.raises(ConditionEvaluationError, match="env"):
            evaluate_condition("env in allowed_envs", {"allowed_envs": ["local"]})

    def test_in_state_key_rhs_unknown_raises(self) -> None:
        with pytest.raises(ConditionEvaluationError, match="allowed_envs"):
            evaluate_condition("env in allowed_envs", {"env": "local"})


class TestAnd:
    def test_and_both_true(self) -> None:
        assert (
            evaluate_condition(
                "env == local and mode == dev", {"env": "local", "mode": "dev"}
            )
            is True
        )

    def test_and_first_false(self) -> None:
        assert (
            evaluate_condition(
                "env == cloud and mode == dev", {"env": "local", "mode": "dev"}
            )
            is False
        )

    def test_and_second_false(self) -> None:
        assert (
            evaluate_condition(
                "env == local and mode == prod", {"env": "local", "mode": "dev"}
            )
            is False
        )

    def test_and_both_false(self) -> None:
        assert (
            evaluate_condition(
                "env == cloud and mode == prod", {"env": "local", "mode": "dev"}
            )
            is False
        )

    def test_and_unknown_key_raises(self) -> None:
        with pytest.raises(ConditionEvaluationError):
            evaluate_condition("env == local and missing_key == x", {"env": "local"})

    def test_and_inside_quoted_value_is_not_clause_split(self) -> None:
        assert (
            evaluate_condition(
                'mode == "research and development" and env == local',
                {"mode": "research and development", "env": "local"},
            )
            is True
        )


class TestCanonicalGraphConditions:
    """Tests based on real condition patterns that would appear in onboarding contracts."""

    def test_deployment_mode_condition(self) -> None:
        assert (
            evaluate_condition(
                "deployment_mode in [cloud, hybrid]",
                {"deployment_mode": "cloud"},
            )
            is True
        )

    def test_deployment_mode_local_excluded(self) -> None:
        assert (
            evaluate_condition(
                "deployment_mode in [cloud, hybrid]",
                {"deployment_mode": "local"},
            )
            is False
        )

    def test_user_role_not_in(self) -> None:
        assert (
            evaluate_condition(
                "user_role not in [guest, viewer]",
                {"user_role": "admin"},
            )
            is True
        )

    def test_combined_role_and_env(self) -> None:
        assert (
            evaluate_condition(
                "user_role == admin and env == production",
                {"user_role": "admin", "env": "production"},
            )
            is True
        )

    def test_value_in_state_list(self) -> None:
        assert (
            evaluate_condition(
                "selected_track in available_tracks",
                {
                    "selected_track": "omnimarket",
                    "available_tracks": ["omnimarket", "standalone"],
                },
            )
            is True
        )


class TestInteractiveOnboardingConditions:
    """Tests from OMN-10769 covering interactive onboarding-specific patterns."""

    def test_compound_deployment_and_service_not_in(self) -> None:
        state = {"deployment_mode": "local", "selected_local_services": ["kafka"]}
        assert (
            evaluate_condition(
                "deployment_mode == local and llm_inference not in selected_local_services",
                state,
            )
            is True
        )

    def test_response_membership_check(self) -> None:
        state = {
            "selected_local_services": ["kafka", "postgres"],
            "response": ["kafka", "postgres"],
        }
        assert evaluate_condition("llm_inference in response", state) is False
        assert evaluate_condition("kafka in response", state) is True


class TestUnknownKeyErrors:
    def test_unknown_key_message_includes_key_name(self) -> None:
        with pytest.raises(ConditionEvaluationError, match="my_missing_key"):
            evaluate_condition("my_missing_key == foo", {})

    def test_unknown_key_is_not_eval_error(self) -> None:
        exc = None
        try:
            evaluate_condition("missing == x", {})
        except ConditionEvaluationError as e:
            exc = e
        assert exc is not None
        assert not isinstance(exc, SyntaxError)

    def test_unknown_key_in_membership_raises(self) -> None:
        with pytest.raises(ConditionEvaluationError, match="nonexistent_list"):
            evaluate_condition("llm_inference in nonexistent_list", {})
