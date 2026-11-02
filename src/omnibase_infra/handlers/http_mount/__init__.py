# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Serve contract-declared ASGI applications on the runtime's HTTP port (OMN-21288)."""

from omnibase_infra.handlers.http_mount.asgi_mounts import AsgiMounts

__all__ = ["AsgiMounts"]
