# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Parity tests for the canonical ``handle()`` of node_architecture_validator.

OMN-21924 gives ``HandlerArchitectureValidation`` a ``handle`` entry that takes the
one typed request model and returns the typed result, so the node dispatches
through its contract. For a request that carries no stack-shape snapshot, ``handle``
must return exactly what ``validate_architecture`` returned before the change. These
tests drive the handler through that entry. Against the merge-base handler, which has
no ``handle``, each test turns the ``AttributeError`` into its own assertion failure.

The scenarios double as the candidate inputs the adequacy receipts under
``scripts/ci/adequacy_receipts/`` were recorded over.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums import EnumValidationSeverity
from omnibase_infra.nodes.node_architecture_validator.handlers.handler_architecture_validation import (
    HandlerArchitectureValidation,
)
from omnibase_infra.nodes.node_architecture_validator.models import (
    ModelArchitectureValidationRequest,
    ModelArchitectureValidationResult,
)
from tests.unit.nodes.node_architecture_validator.test_node_architecture_validator import (
    MockRule,
)

pytestmark = pytest.mark.unit


class SampleNode:
    """Validation target standing in for a node."""

    __name__ = "SampleNode"


class SampleHandler:
    """Validation target standing in for a handler."""

    __name__ = "SampleHandler"


class ModelValidationScenario(BaseModel):
    """One validation request and the rules the handler is built with."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_count: int
    handler_count: int
    failing_rule: bool
    fail_fast: bool
    rule_ids: tuple[str, ...] | None


SCENARIOS: tuple[ModelValidationScenario, ...] = (
    ModelValidationScenario(
        node_count=0,
        handler_count=0,
        failing_rule=False,
        fail_fast=False,
        rule_ids=None,
    ),
    ModelValidationScenario(
        node_count=1,
        handler_count=1,
        failing_rule=False,
        fail_fast=False,
        rule_ids=None,
    ),
    ModelValidationScenario(
        node_count=2,
        handler_count=1,
        failing_rule=True,
        fail_fast=False,
        rule_ids=None,
    ),
    ModelValidationScenario(
        node_count=2,
        handler_count=2,
        failing_rule=True,
        fail_fast=True,
        rule_ids=None,
    ),
    ModelValidationScenario(
        node_count=1,
        handler_count=0,
        failing_rule=True,
        fail_fast=False,
        rule_ids=("NO_HANDLER_PUBLISHING",),
    ),
)


def build_handler(scenario: ModelValidationScenario) -> HandlerArchitectureValidation:
    """The handler under test, with one passing and optionally one failing rule."""
    rules = [
        MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )
    ]
    if scenario.failing_rule:
        rules.append(
            MockRule(
                rule_id="PURE_REDUCERS",
                name="Pure Reducers Rule",
                severity=EnumValidationSeverity.ERROR,
                should_pass=False,
                message="This rule always fails",
            )
        )
    return HandlerArchitectureValidation(rules=tuple(rules))


def build_request(
    scenario: ModelValidationScenario,
) -> ModelArchitectureValidationRequest:
    """The request a scenario describes."""
    return ModelArchitectureValidationRequest(
        nodes=tuple(SampleNode() for _ in range(scenario.node_count)),
        handlers=tuple(SampleHandler() for _ in range(scenario.handler_count)),
        rule_ids=scenario.rule_ids,
        fail_fast=scenario.fail_fast,
    )


def run_scenario(
    scenario: ModelValidationScenario,
) -> ModelArchitectureValidationResult:
    """Call the handler the way the runtime does: one request argument."""
    handler = build_handler(scenario)
    try:
        return handler.handle(build_request(scenario))
    except AttributeError as exc:
        pytest.fail(f"handler has no canonical handle() entry: {exc}")


class TestArchitectureValidationHandle:
    """HandlerArchitectureValidation.handle takes one request model."""

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_handle_returns_what_validate_architecture_returns(
        self, scenario: ModelValidationScenario
    ) -> None:
        result = run_scenario(scenario)
        expected = build_handler(scenario).validate_architecture(
            build_request(scenario)
        )
        assert isinstance(result, ModelArchitectureValidationResult)
        assert result == expected

    def test_empty_request_is_valid(self) -> None:
        result = run_scenario(SCENARIOS[0])
        assert result.valid is True
        assert result.nodes_checked == 0
        assert result.handlers_checked == 0

    def test_failing_rule_reports_its_violations(self) -> None:
        result = run_scenario(SCENARIOS[2])
        assert result.valid is False
        assert result.violation_count > 0

    def test_fail_fast_stops_at_the_first_violation(self) -> None:
        result = run_scenario(SCENARIOS[3])
        assert result.valid is False
        assert result.violation_count == 1
