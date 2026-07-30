# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hermetic test double for omniclaude's canonical deploy-path classifier."""

from __future__ import annotations


def find_runtime_paths(changed_files: list[str]) -> list[str]:
    """Return the runtime paths exercised by rebuild-trigger unit tests."""
    runtime_prefixes = ("src/omnimarket/", "src/omnibase_infra/nodes/")
    return [
        path
        for path in changed_files
        if path.startswith(runtime_prefixes) or path.startswith("docker/docker-compose")
    ]
