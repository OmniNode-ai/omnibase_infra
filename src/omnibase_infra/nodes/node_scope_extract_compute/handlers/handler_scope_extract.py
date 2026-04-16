# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that extracts scope items from plan file content.

This is a COMPUTE handler - pure transformation, no I/O.
"""

from __future__ import annotations

import logging
import re
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extracted import (
    ModelScopeExtracted,
)

logger = logging.getLogger(__name__)

# Known OmniNode repositories for repo detection
KNOWN_REPOS: frozenset[str] = frozenset(
    {
        "omniclaude",
        "omnibase_core",
        "omnibase_infra",
        "omnibase_spi",
        "omniintelligence",
        "omnimemory",
        "omnidash",
        "omninode_infra",
        "omniweb",
        "onex_change_control",
        "omnibase_compat",
    }
)


class HandlerScopeExtract:
    """Extracts scope items (files, directories, repos, systems) from plan content."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        content: str,
        plan_file_path: str,
        correlation_id: UUID,
        output_path: str = "~/.claude/scope-manifest.json",
    ) -> ModelScopeExtracted:
        """Extract scope items from plan content.

        Extraction heuristics:
            - File paths in backticks (e.g., `src/foo.py`)
            - "Files affected" / "Files Affected" sections
            - Known repo names
            - Directory paths ending in /

        Args:
            content: Plan file content.
            plan_file_path: Original plan file path.
            correlation_id: Workflow correlation ID.
            output_path: Caller-specified output path to carry forward.

        Returns:
            ModelScopeExtracted with extracted scope items.
        """
        logger.info(
            "Extracting scope from plan file (correlation_id=%s)",
            correlation_id,
        )

        files: list[str] = []
        directories: list[str] = []
        repos: list[str] = []
        systems: list[str] = []

        # Extract paths from backticks
        backtick_paths = re.findall(r"`([^`]+)`", content)
        for path in backtick_paths:
            # Skip things that look like code, not paths
            if " " in path or "=" in path or "(" in path:
                continue
            if path.endswith("/"):
                directories.append(path)
            elif "." in path.split("/")[-1] if "/" in path else "." in path:
                # Has a file extension
                files.append(path)
            elif "/" in path:
                # Path-like but no extension - treat as directory
                directories.append(path)

        # Extract repos from known names (word-boundary aware to avoid substring matches)
        for repo in KNOWN_REPOS:
            if re.search(rf"\b{re.escape(repo)}\b", content):
                repos.append(repo)

        # Extract "Files affected" or "Files Affected" sections
        files_section = re.search(
            r"(?:Files?\s+[Aa]ffected|Scope):?\s*\n((?:\s*[-*]\s+.+\n)+)",
            content,
        )
        if files_section:
            for line in files_section.group(1).splitlines():
                item = re.sub(r"^\s*[-*]\s+", "", line).strip()
                if item:
                    item = item.strip("`")
                    if item.endswith("/"):
                        directories.append(item)
                    else:
                        files.append(item)

        # Extract systems from common keywords
        system_keywords = [
            "hooks",
            "skills",
            "CLAUDE.md",
            "CI pipeline",
            "Docker",
            "Kafka",
            "PostgreSQL",
            "runtime",
            "dashboard",
        ]
        for kw in system_keywords:
            if kw.lower() in content.lower():
                systems.append(kw)

        # Deduplicate while preserving order
        files = list(dict.fromkeys(files))
        directories = list(dict.fromkeys(directories))
        repos = list(dict.fromkeys(repos))
        systems = list(dict.fromkeys(systems))

        logger.info(
            "Extracted scope: %d files, %d dirs, %d repos, %d systems",
            len(files),
            len(directories),
            len(repos),
            len(systems),
        )

        return ModelScopeExtracted(
            correlation_id=correlation_id,
            plan_file_path=plan_file_path,
            output_path=output_path,
            files=tuple(files),
            directories=tuple(directories),
            repos=tuple(repos),
            systems=tuple(systems),
        )
