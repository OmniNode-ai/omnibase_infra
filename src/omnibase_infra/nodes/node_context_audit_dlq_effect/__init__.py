# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Context audit DLQ effect — contract stub for topic provisioning."""

from pydantic import BaseModel


class ModelContextAuditDlqInput(BaseModel):
    """Stub input model for context audit DLQ processing."""


class ModelContextAuditDlqOutput(BaseModel):
    """Stub output model for context audit DLQ processing."""


__all__ = ["ModelContextAuditDlqInput", "ModelContextAuditDlqOutput"]
