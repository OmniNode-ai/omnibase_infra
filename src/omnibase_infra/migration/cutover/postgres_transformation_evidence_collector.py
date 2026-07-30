# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL collector for transformation-aware evidence query contracts."""

from __future__ import annotations

import hashlib
import json

import asyncpg

from omnibase_infra.migration.cutover.models import (
    ModelPostgresEvidenceQuerySet,
    ModelTransformationEvidence,
)


class PostgresTransformationEvidenceCollector:
    """Execute a complete read-only query set against one PostgreSQL database."""

    def __init__(self, connection: asyncpg.Connection) -> None:
        self._connection = connection

    async def collect(
        self,
        queries: ModelPostgresEvidenceQuerySet,
    ) -> ModelTransformationEvidence:
        """Collect all receipt dimensions without defaults or omitted scans."""
        keys = await self._fetch_strings(queries.keys_sql)
        rows = await self._fetch_strings(queries.rows_sql)
        return ModelTransformationEvidence(
            label=queries.label,
            evidence_contract_hash=self._query_contract_hash(queries),
            keys=keys,
            row_count=len(rows),
            transformed_row_hashes=tuple(
                sorted(hashlib.sha256(row.encode("utf-8")).hexdigest() for row in rows)
            ),
            foreign_keys=await self._fetch_strings(queries.foreign_keys_sql),
            sequences=await self._fetch_strings(queries.sequences_sql),
            owners=await self._fetch_strings(queries.owners_sql),
            grants=await self._fetch_strings(queries.grants_sql),
            policies=await self._fetch_strings(queries.policies_sql),
            views_functions=await self._fetch_strings(queries.views_functions_sql),
            dependencies=await self._fetch_strings(queries.dependencies_sql),
            collision_keys=await self._fetch_strings(queries.collisions_sql),
        )

    async def collect_pair(
        self,
        source_queries: ModelPostgresEvidenceQuerySet,
        target_queries: ModelPostgresEvidenceQuerySet,
    ) -> tuple[ModelTransformationEvidence, ModelTransformationEvidence]:
        """Collect source and target in one read-only repeatable-read snapshot."""
        async with self._connection.transaction(
            isolation="repeatable_read",
            readonly=True,
        ):
            source = await self.collect(source_queries)
            target = await self.collect(target_queries)
        return source, target

    @staticmethod
    def _query_contract_hash(queries: ModelPostgresEvidenceQuerySet) -> str:
        encoded = json.dumps(
            queries.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    async def _fetch_strings(self, query: str) -> tuple[str, ...]:
        rows = await self._connection.fetch(query)
        values: list[str] = []
        for row in rows:
            if len(row) != 1:
                raise ValueError("evidence queries must return exactly one column")
            value = row[0]
            if value is None:
                raise ValueError("evidence queries must not return NULL signatures")
            if not isinstance(value, str):
                raise TypeError("evidence queries must return text signatures")
            values.append(value)
        return tuple(sorted(values))


__all__ = ["PostgresTransformationEvidenceCollector"]
