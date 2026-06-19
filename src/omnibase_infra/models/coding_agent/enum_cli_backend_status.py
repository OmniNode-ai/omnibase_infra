# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structured CLI subprocess failure classes for coding-agent invocations.

Mirrors the failure vocabulary in
``node_llm_inference_effect.handlers.handler_llm_cli_subprocess.EnumCliBackendStatus``.
Phase D (OMN-13250) collapses the duplicate by relocating the inference handler;
until then the coding-agent surface owns its own copy so the invoke effect does
not import a sibling node's private handler module.
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
