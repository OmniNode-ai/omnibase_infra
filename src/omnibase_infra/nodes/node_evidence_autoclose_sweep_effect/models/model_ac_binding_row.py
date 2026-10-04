# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One row of the acceptance-criterion / evidence-check binding table.

OMN-20368. The model now lives in ``omnibase_core`` beside the shared Done-write
receipt gate that produces it; this module keeps the closer's import path.
"""

from __future__ import annotations

from omnibase_core.models.ticket.model_ac_binding_row import ModelAcBindingRow

__all__ = ["ModelAcBindingRow"]
