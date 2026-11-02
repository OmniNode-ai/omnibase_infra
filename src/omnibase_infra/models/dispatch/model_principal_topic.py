# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A contract-declared publish topic that holds one ``{principal}`` slot (OMN-22103)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

SLOT = "{principal}"
#: A principal as the lab issuer names one: lower-case, digits and hyphens.
PRINCIPAL_PATTERN = "[a-z0-9][a-z0-9-]{0,62}"


class ModelPrincipalTopic(BaseModel):
    """A declared topic with exactly one ``{principal}`` slot, and the names it admits.

    A per-principal reply topic cannot be a static contract entry: the requester's
    principal is in its name. The contract declares the template, and an emitted topic
    is admitted only when it is that template with a well-formed principal in the slot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    template: str

    def matches(self, candidate: str) -> bool:
        """True when ``candidate`` is this template with a principal filling the slot."""
        if self.template.count(SLOT) != 1:
            return False
        pattern = PRINCIPAL_PATTERN.join(
            re.escape(part) for part in self.template.split(SLOT)
        )
        return re.fullmatch(pattern, candidate) is not None

    @staticmethod
    def admits(candidate: str, declared: Iterable[str]) -> bool:
        """True when ``candidate`` is a declared literal topic or a declared template filled.

        An unfilled template is no topic: the slot is never published to as written.
        """
        topics = tuple(declared)
        if SLOT not in candidate and candidate in topics:
            return True
        return any(ModelPrincipalTopic(template=t).matches(candidate) for t in topics)
