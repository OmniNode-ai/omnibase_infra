# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""An absent execution-budget map defaults; a present broken one refuses (OMN-18924).

THE DEFECT. ``resolve_task_class_execution_budget`` was added by OMN-15504 and
fails closed when the task-class contract declares no ``execution_budgets``
map. No merged omnimarket branch has ever declared that map -- its producer is
still an open pull request -- so on 2026-09-20 every ``onex delegate`` through
the sanctioned wrapper was refused BEFORE any dispatch, with
``task-class contract at ... declares no execution_budgets map``. Reproduced at
18:2x UTC with an ordinary prose prompt opening "Summarize".

It merged green for two compounding reasons worth recording, because either
alone would have hidden it. The same change added the map to two fixtures in
this repo, so this suite had a map while the shipped contract did not; and the
dispatch venv installs this package by LOCAL PATH from the canonical clone, so
the requirement reached the host the moment the clone carried the merge, with
no release in between to notice at.

THE RULE IT BROKE. A new contract field lands consumer-first, or is excluded
when unset. A consumer that makes an un-defaulted field mandatory inverts that,
and its blast radius is every caller rather than the one feature the field
serves. So the fail-closed branch is the defect, not the missing map.

WHAT CHANGES, AND WHAT DELIBERATELY DOES NOT. An ABSENT map resolves to the
default the introducing change's own fixtures already declare for all eleven
public classes -- 240 second ceiling, 60 second delivery margin -- and says so
in one log line naming the contract path, so a defaulted budget is never
mistaken for a declared one. A PRESENT map keeps every refusal exactly as
written: malformed, missing the resolved class, or carrying an invalid entry
all still raise. A contract that declares the map is unaffected value for
value, so when the producer lands, the declared budgets take effect and none
of this needs reverting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from omnibase_infra.cli.model_task_class_execution_budget import (
    ModelTaskClassExecutionBudget,
)
from omnibase_infra.cli.task_class_selection import (
    DEFAULT_EXECUTION_BUDGET,
    TaskClassContractError,
    resolve_task_class_execution_budget,
)

pytestmark = pytest.mark.unit

#: The shape the SHIPPED contract actually has: public classes with selection
#: predicates and no `execution_budgets` key anywhere. This is the reproduction.
_SHIPPED_SHAPE = """\
version: "omn18924-shipped-shape"
task_classes:
  document:
    gateway_exposure: public
    selection:
      priority: 62
      phrases: ["document"]
  summarization:
    gateway_exposure: public
    selection:
      priority: 80
      min_words: 120
      phrases: ["summarize"]
"""

#: The same contract WITH the map, at the values the introducing change's own
#: fixtures declare. Used as the positive control that a declared budget wins.
_WITH_MAP = (
    _SHIPPED_SHAPE
    + """\
execution_budgets:
  document: {task_class_timeout_ceiling_seconds: 90, terminal_delivery_margin_seconds: 30}
"""
)


def _contract(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "task_class_contracts.v1.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestTheShippedContractResolves:
    """AC1 and AC2. The reproduction, as the test that should have existed."""

    def test_an_absent_map_resolves_to_the_default(self, tmp_path: Path) -> None:
        budget = resolve_task_class_execution_budget(
            _contract(tmp_path, _SHIPPED_SHAPE), task_type="document"
        )
        assert budget == DEFAULT_EXECUTION_BUDGET

    def test_the_default_is_the_value_the_fixtures_already_declared(self) -> None:
        """Not a number invented here: it is what all eleven classes declare."""
        assert DEFAULT_EXECUTION_BUDGET.task_class_timeout_ceiling_seconds == 240
        assert DEFAULT_EXECUTION_BUDGET.terminal_delivery_margin_seconds == 60

    def test_the_fallback_is_recorded_rather_than_silent(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A defaulted budget must be distinguishable from a declared one."""
        contract = _contract(tmp_path, _SHIPPED_SHAPE)
        with caplog.at_level(logging.INFO):
            resolve_task_class_execution_budget(contract, task_type="document")
        assert any(str(contract) in record.getMessage() for record in caplog.records)

    def test_every_public_class_resolves_on_the_shipped_shape(
        self, tmp_path: Path
    ) -> None:
        """The refusal was per-call, so one class passing proves little."""
        contract = _contract(tmp_path, _SHIPPED_SHAPE)
        for task_type in ("document", "summarization"):
            assert (
                resolve_task_class_execution_budget(contract, task_type=task_type)
                == DEFAULT_EXECUTION_BUDGET
            )


class TestADeclaredMapStillWins:
    """AC4. The default must never override a contract that states a value."""

    def test_a_declared_budget_is_used_verbatim(self, tmp_path: Path) -> None:
        budget = resolve_task_class_execution_budget(
            _contract(tmp_path, _WITH_MAP), task_type="document"
        )
        assert budget == ModelTaskClassExecutionBudget(
            task_class_timeout_ceiling_seconds=90,
            terminal_delivery_margin_seconds=30,
        )
        assert budget != DEFAULT_EXECUTION_BUDGET

    def test_a_declared_budget_is_not_logged_as_a_fallback(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Positive control on the log line: it marks the default, not every read."""
        contract = _contract(tmp_path, _WITH_MAP)
        with caplog.at_level(logging.INFO):
            resolve_task_class_execution_budget(contract, task_type="document")
        assert not [
            record
            for record in caplog.records
            if "execution_budgets" in record.getMessage()
        ]


class TestAPresentMapStillFailsClosed:
    """AC3. Tolerance is for ABSENCE only; a stated-but-wrong budget is a defect."""

    def test_a_present_map_that_is_not_a_mapping_refuses(self, tmp_path: Path) -> None:
        contract = _contract(
            tmp_path, _SHIPPED_SHAPE + "execution_budgets: not-a-mapping\n"
        )
        with pytest.raises(TaskClassContractError, match="execution_budgets"):
            resolve_task_class_execution_budget(contract, task_type="document")

    def test_a_present_map_missing_the_resolved_class_refuses(
        self, tmp_path: Path
    ) -> None:
        contract = _contract(tmp_path, _WITH_MAP)
        with pytest.raises(TaskClassContractError, match="summarization"):
            resolve_task_class_execution_budget(contract, task_type="summarization")

    def test_a_present_entry_with_an_invalid_value_refuses(
        self, tmp_path: Path
    ) -> None:
        """The ceiling is capped; a value past it is a contract defect, not a default."""
        contract = _contract(
            tmp_path,
            _SHIPPED_SHAPE
            + "execution_budgets:\n"
            + "  document: {task_class_timeout_ceiling_seconds: 6000, "
            + "terminal_delivery_margin_seconds: 60}\n",
        )
        with pytest.raises(TaskClassContractError, match="invalid execution budget"):
            resolve_task_class_execution_budget(contract, task_type="document")

    def test_an_unreadable_contract_still_refuses(self, tmp_path: Path) -> None:
        """Tolerating an absent FIELD must not tolerate an absent FILE."""
        with pytest.raises(TaskClassContractError):
            resolve_task_class_execution_budget(
                tmp_path / "nope.yaml", task_type="document"
            )
