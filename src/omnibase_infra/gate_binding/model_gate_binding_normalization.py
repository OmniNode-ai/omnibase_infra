# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One rewrite applied to a binding's text before any form is matched."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelGateBindingNormalization(BaseModel):
    """A declared rewrite of a binding's raw text.

    These exist because the text a reader gets back from Linear is not the text
    the author typed: Linear rewrites issue identifiers into mention elements,
    and editors auto-link them. A form pattern built against the authored
    spelling matches nothing on a live ticket.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    pattern: str
    replacement: str
