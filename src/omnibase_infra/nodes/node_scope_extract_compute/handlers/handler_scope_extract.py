# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that extracts scope items from plan file content.

This is a COMPUTE handler - pure transformation, no I/O.
"""

from __future__ import annotations

import logging
import re

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extract_input import (
    ModelScopeExtractInput,
)
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

    async def handle(self, request: ModelScopeExtractInput) -> ModelScopeExtracted:
        """Extract scope items from plan content (canonical definition B).

        The shared runtime adapter validates the wire payload into the
        contract-declared ``ModelScopeExtractInput`` and hands it here — the
        envelope boundary lives in the runtime adapter, not in this compute core
        (definition B, OMN-14355). The regex extraction body below is preserved
        behaviorally identical to the pre-flip multi-positional handler; only the
        entry signature is unpacked from the request model.

        Extraction heuristics:
            - File paths in backticks (e.g., `src/foo.py`)
            - "Files affected" / "Files Affected" sections
            - Known repo names
            - Directory paths ending in /

        Args:
            request: Validated scope-extract input carrying the plan file
                content, original plan file path, workflow correlation ID, and
                the caller-specified output path to carry forward.

        Returns:
            ModelScopeExtracted with extracted scope items.
        """
        content = request.content
        plan_file_path = request.plan_file_path
        correlation_id = request.correlation_id
        output_path = request.output_path

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
