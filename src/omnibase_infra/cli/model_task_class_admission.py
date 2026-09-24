# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Which task classes an explicit ``--task-type`` may name (OMN-13966)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.cli.model_unavailable_task_class import (
    ModelUnavailableTaskClass,
)

__all__ = ["ModelTaskClassAdmission"]


class ModelTaskClassAdmission(BaseModel):
    """The contract's whole class set, split into admitted and unroutable.

    ``admitted`` and the names in ``unavailable`` are disjoint and together
    equal every class the contract declares, public and internal alike. The
    split is derived from the contract on every run, so it cannot fall behind
    it; the two pinned halves in ``test_task_class_admission_omn13966.py``
    (here) and omnimarket's test of the same name hold the live contract to it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    admitted: frozenset[str]
    unavailable: tuple[ModelUnavailableTaskClass, ...]

    def unavailable_named(self, name: str) -> ModelUnavailableTaskClass | None:
        """Return the unroutable declaration for ``name``, or ``None``."""
        for entry in self.unavailable:
            if entry.name == name:
                return entry
        return None
