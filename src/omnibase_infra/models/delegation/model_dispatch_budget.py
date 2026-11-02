# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The time limit one delegation dispatch runs under (OMN-21604).

The ceiling and delivery margin are the task-class authority's (the contract, or
the site overlay that raises it); nothing here holds a number of its own. An
unset figure resolves to the class's declared one. A class whose ceiling plus
margin passes the consumer dispatch deadline is refused when the dispatch port
loads, rather than cancelled mid-run.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.cli.task_class_registry import (
    TaskClassContractError,
    load_task_class_authority,
)
from omnibase_infra.errors import ProtocolConfigurationError

logger = logging.getLogger(__name__)


class ModelDispatchBudget(BaseModel):
    """The execution window and delivery margin a dispatch runs under, in seconds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_timeout_seconds: int = Field(gt=0)
    terminal_delivery_margin_seconds: int = Field(gt=0)

    @classmethod
    def resolve(
        cls,
        task_type: str,
        execution_timeout_seconds: int | None,
        terminal_delivery_margin_seconds: int | None,
    ) -> tuple[int, int]:
        """Return (execution window, delivery margin) in seconds.

        Keeps each figure the caller passed; an unset one is the class's declared figure.
        """
        if (
            execution_timeout_seconds is None
            or terminal_delivery_margin_seconds is None
        ):
            declared = load_task_class_authority().execution_budget(task_type)
            if execution_timeout_seconds is None:
                execution_timeout_seconds = declared.task_class_timeout_ceiling_seconds
            if terminal_delivery_margin_seconds is None:
                terminal_delivery_margin_seconds = (
                    declared.terminal_delivery_margin_seconds
                )
        budget = cls(
            execution_timeout_seconds=execution_timeout_seconds,
            terminal_delivery_margin_seconds=terminal_delivery_margin_seconds,
        )
        return budget.execution_timeout_seconds, budget.terminal_delivery_margin_seconds

    @classmethod
    def refuse_past_dispatch_deadline(cls, event_bus: object) -> None:
        """Refuse a declared class the bus's consumer dispatch deadline would cancel.

        Names each class and both figures. A bus that exposes no dispatch deadline
        carries no check, and where no task-class authority is installed there is
        no budget to compare.
        """
        config = getattr(event_bus, "config", None)
        deadline = getattr(config, "effective_dispatch_deadline_seconds", None)
        if not isinstance(deadline, int | float):
            return
        try:
            authority = load_task_class_authority()
        except TaskClassContractError:
            logger.warning(
                "delegation budget vs dispatch deadline check skipped: no "
                "task-class authority is installed"
            )
            return
        offenders: list[str] = []
        for task_class in sorted(
            authority.public_task_classes | authority.internal_task_classes
        ):
            try:
                declared = authority.execution_budget(task_class)
            except ValueError:
                continue  # a class with no declared budget is refused at dispatch
            ceiling = declared.task_class_timeout_ceiling_seconds
            margin = declared.terminal_delivery_margin_seconds
            if ceiling + margin > deadline:
                offenders.append(
                    f"{task_class} ({ceiling} + {margin} = {ceiling + margin} s)"
                )
        if offenders:
            raise ProtocolConfigurationError(
                "task classes whose execution ceiling plus delivery margin exceed "
                f"the consumer dispatch deadline of {deadline:g} s would be "
                f"cancelled mid-run: {', '.join(offenders)}. Lower the overlay "
                "ceiling or raise the consumer dispatch deadline."
            )
