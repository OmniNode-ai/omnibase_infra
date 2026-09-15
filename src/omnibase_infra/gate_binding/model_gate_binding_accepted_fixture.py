# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A binding the contract declares readable, and the form it must be."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelGateBindingAcceptedFixture(BaseModel):
    """A worked example both consumers run against their own resolver."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    binding: str
    form: str
