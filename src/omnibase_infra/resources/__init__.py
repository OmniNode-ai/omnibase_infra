# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Infrastructure-layer resource managers.

Manages runtime resources (e.g. httpx clients) outside the ONEX graph.
Callers control instantiation and lifetime.
"""

from omnibase_infra.resources.handler_resource_manager import HandlerResourceManager

__all__ = ["HandlerResourceManager"]
