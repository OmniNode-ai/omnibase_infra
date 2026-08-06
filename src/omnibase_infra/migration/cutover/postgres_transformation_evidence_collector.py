# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL collector for transformation-aware evidence query contracts."""

from __future__ import annotations

import hashlib
import json
import re
from uuid import UUID

import asyncpg

from omnibase_infra.migration.cutover.models import (
    ModelApplicationPathWriteProof,
    ModelConnectionIdentity,
    ModelPostgresEvidenceQuerySet,
    ModelTransformationEvidence,
)

_READ_PREFIX = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_MUTATING_TOKEN = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|GRANT|REVOKE|TRUNCATE|COPY|CALL|DO)\b",
    re.IGNORECASE,
)


class PostgresTransformationEvidenceCollector:
    """Execute a complete read-only query set against one PostgreSQL database."""

    def __init__(self, connection: asyncpg.Connection) -> None:
        self._connection = connection

    async def verify_application_path_write(
        self,
        family_id: UUID,
        verification_sql: str,
        schema_ref: str,
        target_sequence: int,
    ) -> ModelApplicationPathWriteProof:
        """Independently prove a real application-path write occurred.

        ``database_ref``/``principal`` come from a live ``current_database()``/
        ``current_user`` readback on the same connection that performed the
        write -- never from caller-typed strings.  ``verification_sql`` must be
        a caller-declared read-only query whose result rows are hashed into
        ``write_result_hash``, proving the write's actual data landed rather
        than merely being asserted.

        The computed proof is durably registered in
        ``omninode_internal.application_path_write_proofs`` before it is
        returned to the caller. The journal's ``APPLICATION_PATH_WRITE_PROVEN``
        transition dereferences this row and refuses any submitted proof that
        does not match it field-for-field -- a proof that never passed through
        this method (however well-shaped) has no durable row and is rejected.
        """
        if not _READ_PREFIX.match(verification_sql):
            raise ValueError("write verification query must start with SELECT or WITH")
        if ";" in verification_sql:
            raise ValueError("write verification query must be exactly one statement")
        if _MUTATING_TOKEN.search(verification_sql):
            raise ValueError("write verification query must be read-only")

        schema_exists = await self._connection.fetchval(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name = $1",
            schema_ref,
        )
        if not schema_exists:
            raise ValueError(
                f"schema {schema_ref!r} does not exist on the verifying connection"
            )

        identity_row = await self._connection.fetchrow(
            "SELECT current_database() AS database, current_user AS principal, "
            "pg_backend_pid() AS backend_pid, clock_timestamp() AS collected_at"
        )
        if identity_row is None:
            raise RuntimeError("write-proof identity readback returned no row")

        rows = await self._connection.fetch(verification_sql)
        if not rows:
            raise ValueError(
                "write verification query returned no rows; write is not proven"
            )
        canonical_rows = sorted(
            json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            )
            for row in rows
        )
        write_result_hash = hashlib.sha256(
            json.dumps(canonical_rows, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()
        verification_query_hash = hashlib.sha256(
            verification_sql.encode("utf-8")
        ).hexdigest()

        proof = ModelApplicationPathWriteProof(
            family_id=family_id,
            database_ref=identity_row["database"],
            principal=identity_row["principal"],
            schema_ref=schema_ref,
            target_sequence=target_sequence,
            verification_query_hash=verification_query_hash,
            write_result_hash=write_result_hash,
            connection_identity=ModelConnectionIdentity(
                database=identity_row["database"],
                backend_pid=int(identity_row["backend_pid"]),
                collected_at=identity_row["collected_at"],
            ),
        )
        await self._register_write_proof(proof)
        return proof

    async def _register_write_proof(
        self,
        proof: ModelApplicationPathWriteProof,
    ) -> None:
        """Durably record a collector-verified write proof, once per sequence."""
        await self._connection.execute(
            """
INSERT INTO omninode_internal.application_path_write_proofs
  (family_id, target_sequence, database_ref, principal, schema_ref,
   verification_query_hash, write_result_hash, backend_pid, collected_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (family_id, target_sequence) DO NOTHING
""",
            proof.family_id,
            proof.target_sequence,
            proof.database_ref,
            proof.principal,
            proof.schema_ref,
            proof.verification_query_hash,
            proof.write_result_hash,
            proof.connection_identity.backend_pid,
            proof.connection_identity.collected_at,
        )
        stored = await self._connection.fetchrow(
            """
SELECT write_result_hash
FROM omninode_internal.application_path_write_proofs
WHERE family_id = $1 AND target_sequence = $2
""",
            proof.family_id,
            proof.target_sequence,
        )
        if stored is None or stored["write_result_hash"] != proof.write_result_hash:
            raise ValueError(
                f"target sequence {proof.target_sequence} is already bound to a "
                "different durably-verified write proof"
            )

    async def collect_pair(
        self,
        source_queries: ModelPostgresEvidenceQuerySet,
        source_binding_ref: str,
        target_queries: ModelPostgresEvidenceQuerySet,
        target_binding_ref: str,
    ) -> tuple[ModelTransformationEvidence, ModelTransformationEvidence]:
        """Collect source and target in one read-only repeatable-read snapshot.

        Both sides are stamped with the identical, server-verified connection
        identity (database, backend pid, collection instant) proving they were
        captured atomically on the same live backend -- never on two evidence
        objects assembled independently or by hand.
        """
        async with self._connection.transaction(
            isolation="repeatable_read",
            readonly=True,
        ):
            identity = await self._read_connection_identity()
            source = await self._collect_one(
                source_queries, source_binding_ref, identity
            )
            target = await self._collect_one(
                target_queries, target_binding_ref, identity
            )
        return source, target

    async def _read_connection_identity(self) -> ModelConnectionIdentity:
        row = await self._connection.fetchrow(
            "SELECT current_database() AS database, "
            "pg_backend_pid() AS backend_pid, "
            "clock_timestamp() AS collected_at"
        )
        if row is None:
            raise RuntimeError("connection identity readback returned no row")
        return ModelConnectionIdentity(
            database=row["database"],
            backend_pid=int(row["backend_pid"]),
            collected_at=row["collected_at"],
        )

    async def _collect_one(
        self,
        queries: ModelPostgresEvidenceQuerySet,
        binding_ref: str,
        identity: ModelConnectionIdentity,
    ) -> ModelTransformationEvidence:
        """Collect all receipt dimensions without defaults or omitted scans."""
        keys = await self._fetch_strings(queries.keys_sql)
        rows = await self._fetch_strings(queries.rows_sql)
        return ModelTransformationEvidence(
            label=queries.label,
            evidence_contract_hash=self._query_contract_hash(queries),
            binding_ref=binding_ref,
            connection_identity=identity,
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
