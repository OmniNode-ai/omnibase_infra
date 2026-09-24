# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The CLI admits every task class the contract declares (OMN-13966).

THE DEFECT. ``task_class_contracts.v1.yaml`` declares fifteen task classes.
``onex delegate --task-type`` validated an explicit class against the
contract's ``gateway_exposure: public`` projection only, so it refused the four
internal classes -- ``documentation``, ``validator_generation``,
``agent_delegation`` and ``escalation`` -- with ``unknown task type``. None of
them is unknown. All four are declared Market classes (OMN-15651) that the
wire request model admits, three of them route, and ``agent_delegation`` is
unroutable ON PURPOSE (OMN-15961), which the contract says in its own
``routing_availability`` block and the CLI never repeated.

THE RULE NOW. Auto-selection still reads the public projection only, so an
internal class is never chosen for a prompt. An explicit class is checked
against the whole contract: admitted unless the contract declares it
unroutable, and then refused with the contract's own words.

THE OTHER HALF OF THIS ASSERTION LIVES IN OMNIMARKET:
``tests/unit/inference/test_task_class_admission_omn13966.py`` pins the LIVE
contract's class partition to the same two lists as below. Neither suite can
import the other's half (repo layering one way, a registry pin the other, and
this repo's venv-purity gate refuses to run with omnimarket installed), so the
seam is two pinned halves, the same shape as ``TestTaskTypeVocabulary``. A
class added to, removed from, or made unroutable in the contract turns
omnimarket's half red first; updating these lists and the stand-in contract
is the fix, and this file then proves the CLI admits exactly that partition.
"""

from __future__ import annotations

import pytest
import yaml

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.task_class_selection import (
    EnumTaskTypeResolution,
    TaskClassContractError,
    load_task_class_admission,
)
from tests.unit.cli.conftest import STAND_IN_TASK_CLASS_CONTRACT

pytestmark = pytest.mark.unit

#: Every class the live contract declares that ``--task-type`` must admit.
EXPECTED_ADMITTED_CLASSES = (
    "code_generation",
    "code_review",
    "complex_reasoning",
    "document",
    "documentation",
    "escalation",
    "planning",
    "reasoning",
    "refactor",
    "research",
    "review",
    "summarization",
    "test",
    "validator_generation",
)

#: Every class the live contract declares unroutable (``routing_availability``).
EXPECTED_UNAVAILABLE_CLASSES = ("agent_delegation",)


def _declared_classes() -> set[str]:
    raw = yaml.safe_load(STAND_IN_TASK_CLASS_CONTRACT.read_text(encoding="utf-8"))
    return set(raw["task_classes"])


class TestTheCliAdmitsTheContractsWholeClassSet:
    def test_the_pinned_partition_is_total_and_disjoint(self) -> None:
        admitted = set(EXPECTED_ADMITTED_CLASSES)
        unavailable = set(EXPECTED_UNAVAILABLE_CLASSES)
        assert admitted.isdisjoint(unavailable)
        assert admitted | unavailable == _declared_classes()

    def test_the_stand_in_admission_equals_the_pinned_partition(self) -> None:
        admission = load_task_class_admission(STAND_IN_TASK_CLASS_CONTRACT)
        assert sorted(admission.admitted) == sorted(EXPECTED_ADMITTED_CLASSES)
        assert sorted(entry.name for entry in admission.unavailable) == sorted(
            EXPECTED_UNAVAILABLE_CLASSES
        )

    def test_the_public_help_mirror_is_a_subset_of_the_admitted_set(self) -> None:
        assert set(cli_delegate.TASK_TYPE_CHOICES) <= set(EXPECTED_ADMITTED_CLASSES)

    @pytest.mark.parametrize("task_class", EXPECTED_ADMITTED_CLASSES)
    def test_every_admitted_class_resolves_explicitly_through_the_cli(
        self, task_class: str
    ) -> None:
        resolved = cli_delegate.resolve_task_class("any prompt", explicit=task_class)
        assert resolved.task_type == task_class
        assert resolved.resolution is EnumTaskTypeResolution.EXPLICIT

    def test_agent_delegation_is_refused_with_the_contracts_reason(self) -> None:
        with pytest.raises(TaskClassContractError) as refused:
            cli_delegate.resolve_task_class(
                "split this objective across agents", explicit="agent_delegation"
            )
        message = str(refused.value)
        assert "unknown" not in message
        for fragment in (
            "agent_delegation",
            "pending_capability",
            "agent_orchestration",
            "OMN-15961 WS-4/C6",
            "No routing tier can execute agentic/tool-using work",
        ):
            assert fragment in message

    def test_an_undeclared_class_is_still_unknown(self) -> None:
        with pytest.raises(TaskClassContractError, match="unknown task type"):
            cli_delegate.resolve_task_class("anything", explicit="classification")
