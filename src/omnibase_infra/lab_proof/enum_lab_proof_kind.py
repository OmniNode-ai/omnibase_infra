# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""How a profile proves a pull request head on a lab host.

Ticket: OMN-19565
"""

from __future__ import annotations

from enum import StrEnum


class EnumLabProofKind(StrEnum):
    """How a profile proves a pull request head on a lab host."""

    RUNTIME_IMAGE = "runtime_image"
    FOUNDATION_OVERRIDE = "foundation_override"
    PYPI_SIBLING_OVERRIDE = "pypi_sibling_override"
    PLUGIN_SESSION = "plugin_session"
    WEB_RENDER = "web_render"
    K8S_NAMESPACE = "k8s_namespace"
    SCRIPT_REPLAY = "script_replay"
    EXEMPT = "exempt"


__all__ = ["EnumLabProofKind"]
