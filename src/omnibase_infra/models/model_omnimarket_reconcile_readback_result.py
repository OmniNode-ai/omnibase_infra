# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fresh package and canonical-clone identities after reconciliation."""

from pydantic import BaseModel, ConfigDict


class ModelOmnimarketReconcileReadbackResult(BaseModel):
    """An unavailable side never establishes equality."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    installed: str = ""
    canonical: str = ""

    @property
    def matches(self) -> bool:
        """Require both identities to be readable and equal."""
        return bool(self.canonical) and self.installed == self.canonical
