# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract-compatible predicate container for scope applicability."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelScopePredicate(BaseModel):
    """Contract-compatible predicate container for scope applicability."""

    model_config = ConfigDict(frozen=True, extra="allow")

    def is_universal(self) -> bool:
        return not self.model_dump(mode="json", exclude_none=True)


__all__ = ["ModelScopePredicate"]
