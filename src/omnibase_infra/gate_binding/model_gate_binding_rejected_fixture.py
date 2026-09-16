# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A binding the contract declares unreadable."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelGateBindingRejectedFixture(BaseModel):
    """A worked counter-example.

    A rejected fixture is not an absence of evidence: it is the contract
    stating that this text must NOT be read as a binding, so a resolver that
    grows lenient enough to accept it turns the suite red.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    binding: str
