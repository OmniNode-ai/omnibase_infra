# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL and file operations within the registration bootstrap effect."""

from __future__ import annotations

import logging
from pathlib import Path

import asyncpg
import yaml
from asyncpg import Pool

from omnibase_infra.runtime.contract_topic_router import (
    build_topic_router_from_contract,
)

logger = logging.getLogger(__name__)


class RegistrationBootstrapResources:
    """Perform the resource operations coordinated by the bootstrap handler."""

    @staticmethod
    async def create_pool(postgres_dsn: str) -> Pool:
        """Acquire the original 2..10 connection pool after handler validation."""
        return await asyncpg.create_pool(  # no-contract-check: effect-node pool seam
            postgres_dsn,
            min_size=2,
            max_size=10,
        )

    @staticmethod
    def read_declarations(path: Path) -> dict[str, object]:
        """Read routing declarations with the original best-effort fallback."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "Failed to load registration contract at %s: %s. Using empty topic router.",
                path,
                exc,
            )
            raw = {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def topic_router(path: Path) -> dict[str, str]:
        """Read the registration routing declarations at the effect boundary."""
        return build_topic_router_from_contract(
            RegistrationBootstrapResources.read_declarations(path)
        )

    @staticmethod
    def read_schema(path: Path) -> str:
        """Read the schema file; the plugin retains its failure policy."""
        return path.read_text()

    @staticmethod
    def is_duplicate_schema_error(error: Exception) -> bool:
        """Classify the driver errors bootstrap treats as idempotent."""
        return isinstance(
            error,
            (
                asyncpg.exceptions.DuplicateTableError,
                asyncpg.exceptions.DuplicateObjectError,
            ),
        )
