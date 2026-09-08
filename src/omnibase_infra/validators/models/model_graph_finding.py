# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A single static defect on the contract graph (OMN-18013)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.validators.models.enum_defect_class import DefectClass


class ModelGraphFinding(BaseModel):
    """A single static defect on the contract graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    defect: DefectClass
    topic: str | None = None
    node: str | None = None
    package: str = ""
    detail: str = ""

    def key(self) -> str:
        """Stable identity used for baseline matching."""
        return f"{self.defect}::{self.node or '-'}::{self.topic or '-'}"


__all__ = ["ModelGraphFinding"]
