# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One line, and which of the two line patterns must match it."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelGateBindingLineFixture(BaseModel):
    """Pins the containment between the two line patterns.

    The authoring pattern is strict and the reading pattern tolerant, on
    purpose: a bulleted binding line is refused at creation because a bullet is
    how a line ends up inside a checklist that binds nothing, while a reader
    faces descriptions already written. What this fixture pins is that a line
    admitted at authoring time is readable at reading time -- otherwise a
    ticket can be created and then never read.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    line: str
    authoring: bool
    reading: bool
