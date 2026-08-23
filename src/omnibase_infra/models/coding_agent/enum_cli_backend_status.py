# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structured CLI subprocess failure classes for coding-agent invocations.

OMN-15959: the sibling copy of this vocabulary in
``node_llm_inference_effect.handlers.handler_llm_cli_subprocess`` was deleted
(OMN-13250's Phase D DoD claimed that collapse had already happened; it had
not — the handler was still live with 40 tests). This module is now the sole
owner of the CLI-backend failure vocabulary; the coding-agent surface does not
import a sibling node's private handler module.
"""

from __future__ import annotations

from enum import Enum


class EnumCliBackendStatus(str, Enum):
    """Structured failure classes for a coding-agent CLI subprocess."""

    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT = "timeout"
    SUBPROCESS_ERROR = "subprocess_error"
    EMPTY_RESPONSE = "empty_response"
