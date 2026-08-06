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


def _collect_schema_names(node: object, schemas: set[str]) -> None:
    """Recursively pull every ``"Schema"`` value out of an EXPLAIN JSON plan."""
    if isinstance(node, dict):
        schema = node.get("Schema")
        if isinstance(schema, str):
            schemas.add(schema)
        for value in node.values():
            _collect_schema_names(value, schemas)
    elif isinstance(node, list):
        for item in node:
            _collect_schema_names(item, schemas)


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

        A shape-valid, schema-existing query is not sufficient on its own --
        that oracle previously accepted a proof minted from an arbitrary
        caller-chosen query against an unrelated schema (e.g. the family's
        own SOURCE schema), because nothing tied ``schema_ref``/
        ``verification_sql`` back to what the family actually collected as
        its target.  This method now looks up the family's durable contract
        (``omninode_internal.cutover_family_contracts``) to find its
        ``target_binding_ref``, then requires ``schema_ref`` -- and every
        schema PostgreSQL's own ``EXPLAIN (FORMAT JSON, VERBOSE)`` plan says
        ``verification_sql`` actually reads -- to be a member of the schema
        set durably registered for that binding ref by ``collect_pair()``.
        A query that touches no relation at all (e.g. a bare ``SELECT 1``)
        or that reads from any other schema is refused before it can ever
        be durably registered as a proof.

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

        await self._require_schema_bound_to_target(
            family_id, schema_ref, verification_sql
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

    async def _require_schema_bound_to_target(
        self,
        family_id: UUID,
        schema_ref: str,
        verification_sql: str,
    ) -> None:
        """Refuse a write proof unless its schema is the family's real target.

        ``family_id`` resolves to a durably-registered contract (registration
        is a repository-level precondition; ``application_path_write_proofs``
        itself carries a foreign key to it), which carries the family's
        ``target_binding_ref``.  That binding ref must in turn have a
        durably-registered schema set from a prior ``collect_pair()`` call --
        derived from PostgreSQL's own query plan of the target evidence
        collection, never from a caller string.  Both ``schema_ref`` and
        every schema the planner says ``verification_sql`` actually reads
        must be members of that set.
        """
        target_binding_ref = await self._connection.fetchval(
            """
SELECT contract_json->>'target_binding_ref'
FROM omninode_internal.cutover_family_contracts
WHERE family_id = $1
""",
            family_id,
        )
        if not target_binding_ref:
            raise ValueError(
                f"family {family_id} has no durably registered contract; "
                "register_family() must run before proving an application-path write"
            )

        registered_row = await self._connection.fetchrow(
            """
SELECT schema_names
FROM omninode_internal.target_binding_schemas
WHERE target_binding_ref = $1
""",
            target_binding_ref,
        )
        if registered_row is None:
            raise ValueError(
                f"target binding {target_binding_ref!r} has no durably-collected "
                "evidence; collect_pair() must run before proving an "
                "application-path write"
            )
        legitimate_schemas = frozenset(registered_row["schema_names"])

        if schema_ref not in legitimate_schemas:
            raise ValueError(
                f"schema {schema_ref!r} is not the durably-collected target "
                f"schema for binding {target_binding_ref!r}"
            )

        referenced_schemas = await self._explain_schemas(verification_sql)
        if not referenced_schemas:
            raise ValueError(
                "write verification query does not read any relation; a write "
                "is not proven by a query that touches no target data"
            )
        foreign_schemas = referenced_schemas - legitimate_schemas
        if foreign_schemas:
            raise ValueError(
                "write verification query reads schema(s) "
                f"{sorted(foreign_schemas)} outside the durably-collected "
                f"target for binding {target_binding_ref!r}"
            )

    async def _explain_schemas(self, sql: str) -> frozenset[str]:
        """Return the schema names PostgreSQL's own planner says ``sql`` reads.

        Uses ``EXPLAIN (FORMAT JSON, VERBOSE)`` -- never a text-level parse of
        the caller's SQL -- so the answer is the server's own understanding
        of which relations the query touches, not something a caller can
        spoof by phrasing (comments, aliasing, literal-only queries).
        """
        plan_rows = await self._connection.fetch(
            f"EXPLAIN (FORMAT JSON, VERBOSE) {sql}"
        )
        if not plan_rows:
            raise RuntimeError("EXPLAIN returned no plan for the verification query")
        plan = json.loads(plan_rows[0][0])
        schemas: set[str] = set()
        _collect_schema_names(plan, schemas)
        return frozenset(schemas)

    async def _register_write_proof(
        self,
        proof: ModelApplicationPathWriteProof,
    ) -> None:
        """Durably record a collector-verified write proof, once per sequence."""
        await self._connection.execute(
            """
INSERT INTO omninode_internal.application_path_write_proofs
  (family_id, target_sequence, database_ref, principal, schema_ref,
   verification_query_hash, write_result_hash, connection_database,
   backend_pid, collected_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (family_id, target_sequence) DO NOTHING
""",
            proof.family_id,
            proof.target_sequence,
            proof.database_ref,
            proof.principal,
            proof.schema_ref,
            proof.verification_query_hash,
            proof.write_result_hash,
            proof.connection_identity.database,
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

        The target's schema(s) -- as PostgreSQL's own query plan reports them
        for the data-bearing ``keys_sql``/``rows_sql`` queries, never a caller
        string -- are durably registered against ``target_binding_ref`` so
        ``verify_application_path_write()`` can later refuse a write proof
        minted against any other schema (e.g. the source's).
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
        await self._register_target_binding_schemas(target_binding_ref, target_queries)
        return source, target

    async def _register_target_binding_schemas(
        self,
        target_binding_ref: str,
        queries: ModelPostgresEvidenceQuerySet,
    ) -> None:
        """Durably record which schemas a target binding's real data lives in.

        Derived from PostgreSQL's own ``EXPLAIN`` plan of the data-bearing
        ``keys_sql``/``rows_sql`` queries only (never the auxiliary
        integrity-check queries, which legitimately touch ``pg_catalog``/
        ``information_schema`` and would otherwise leak those in as a
        false-positive "legitimate" target schema).  Runs outside the
        read-only snapshot transaction because it durably writes.
        """
        schemas = (await self._explain_schemas(queries.keys_sql)) | (
            await self._explain_schemas(queries.rows_sql)
        )
        if not schemas:
            raise ValueError(
                f"target binding {target_binding_ref!r} evidence queries "
                "reference no relation; cannot durably establish its target "
                "schema"
            )
        await self._connection.execute(
            """
INSERT INTO omninode_internal.target_binding_schemas (target_binding_ref, schema_names)
VALUES ($1, $2)
ON CONFLICT (target_binding_ref)
DO UPDATE SET schema_names = EXCLUDED.schema_names, registered_at = clock_timestamp()
""",
            target_binding_ref,
            sorted(schemas),
        )

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
