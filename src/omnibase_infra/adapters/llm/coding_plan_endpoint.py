# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Identify Coding Plan endpoints reserved for Claude Code (OMN-20173)."""

from urllib.parse import urlsplit


def is_coding_plan_endpoint(url: str) -> bool:
    """Return whether the URL path addresses the Coding Plan API."""
    path = urlsplit(url).path
    return "/api/coding/" in path or path.endswith("/api/coding")
