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
# EXPLAIN VERBOSE always schema-qualifies a non-``pg_catalog`` function call
# in expression fields (``Output``, ``Filter``, ...) as ``schema.func(...)``,
# even when the function lives in the same schema as the scan it appears
# beside. A bare column reference is never rendered this way (it is
# ``relation.column`` -- alias-qualified, never schema-qualified). This is
# how a query that scans a legitimate relation but pulls its actual returned
# value out of a foreign-schema function call is caught: relation-scan
# harvesting alone cannot see inside an opaque (non-inlined) function call.
_QUALIFIED_CALL_PATTERN = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_]*\s*\("
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


def _collect_expression_schema_refs(node: object, schemas: set[str]) -> None:
    """Recursively harvest schema-qualified function-call refs from EXPLAIN text.

    PostgreSQL does not inline every function call into a child plan node --
    a ``STABLE``/``VOLATILE`` or non-inlinable ``SQL`` function invoked in a
    target list or filter shows up only as an opaque string such as
    ``"legacy_fixture.usage_count()"`` inside fields like ``"Output"``. A
    relation-scan-only walk (``_collect_schema_names``) is blind to this,
    which previously let a verification query "prove" a write by scanning a
    legitimate relation while pulling its actual returned value out of a
    function call that reads a different (e.g. source) schema entirely.
    """
    if isinstance(node, dict):
        for value in node.values():
            _collect_expression_schema_refs(value, schemas)
    elif isinstance(node, list):
        for item in node:
            _collect_expression_schema_refs(item, schemas)
    elif isinstance(node, str):
        for match in _QUALIFIED_CALL_PATTERN.finditer(node):
            schemas.add(match.group(1))


def _collect_relations(node: object, relations: set[tuple[str, str]]) -> None:
    """Recursively pull every ``(schema, relation)`` pair off EXPLAIN scan nodes.

    Schema membership alone is not write-level binding: two unrelated tables
    can share a schema, and a caller can mint a "proof" against any relation
    in that schema regardless of whether it was ever part of the target
    evidence the family actually collected. This harvests the exact relation
    identity PostgreSQL's own planner reports, never a caller string.
    """
    if isinstance(node, dict):
        schema = node.get("Schema")
        relation = node.get("Relation Name")
        if isinstance(schema, str) and isinstance(relation, str):
            relations.add((schema, relation))
        for value in node.values():
            _collect_relations(value, relations)
    elif isinstance(node, list):
        for item in node:
            _collect_relations(item, relations)


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
        schema membership alone is not write-level binding either, since any
        relation in the target schema (or a foreign-schema function call
        embedded in an otherwise-legitimate scan) would satisfy it. This
        method looks up the family's durable contract
        (``omninode_internal.cutover_family_contracts``) to find both its
        ``target_binding_ref`` and its immutable
        ``target_evidence_contract_hash``, then requires ``schema_ref``,
        every schema PostgreSQL's own ``EXPLAIN (FORMAT JSON, VERBOSE)`` plan
        says ``verification_sql`` actually reads (including through an
        opaque schema-qualified function call), and every relation it
        directly scans, to all be members of the *exact* relation set
        ``collect_pair()`` durably registered for that
        ``(target_binding_ref, evidence_contract_hash)`` pair -- never a
        schema string, and never merely ``target_binding_ref``, which a
        later ``collect_pair()`` call could otherwise silently re-point at
        different (attacker-supplied) query content. A query that touches
        no relation at all (e.g. a bare ``SELECT 1``), that reads from any
        other schema or relation, or that pulls its returned value through a
        foreign-schema function call is refused before it can ever be
        durably registered as a proof.

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
        """Refuse a write proof unless it is bound to the family's real target.

        ``family_id`` resolves to a durably-registered contract (registration
        is a repository-level precondition; ``application_path_write_proofs``
        itself carries a foreign key to it), which carries both the family's
        ``target_binding_ref`` *and* its immutable
        ``target_evidence_contract_hash``. The durably-registered relation
        set is looked up by the *pair* -- never ``target_binding_ref`` alone
        -- so a later ``collect_pair()`` call for the same binding ref with
        different (e.g. attacker-typed) query content is content-addressed
        into a distinct row instead of silently overwriting the row this
        family actually depends on. ``schema_ref``, every schema
        ``verification_sql``'s own EXPLAIN plan (including opaque
        schema-qualified function calls it invokes) says it reads, and every
        relation it directly scans must all be members of that pinned row.
        """
        contract_row = await self._connection.fetchrow(
            """
SELECT contract_json->>'target_binding_ref' AS target_binding_ref,
       contract_json->>'target_evidence_contract_hash' AS target_evidence_contract_hash
FROM omninode_internal.cutover_family_contracts
WHERE family_id = $1
""",
            family_id,
        )
        if contract_row is None or not contract_row["target_binding_ref"]:
            raise ValueError(
                f"family {family_id} has no durably registered contract; "
                "register_family() must run before proving an application-path write"
            )
        target_binding_ref = contract_row["target_binding_ref"]
        target_evidence_contract_hash = contract_row["target_evidence_contract_hash"]

        registered_row = await self._connection.fetchrow(
            """
SELECT relation_names
FROM omninode_internal.target_binding_schemas
WHERE target_binding_ref = $1 AND evidence_contract_hash = $2
""",
            target_binding_ref,
            target_evidence_contract_hash,
        )
        if registered_row is None:
            raise ValueError(
                f"target binding {target_binding_ref!r} has no durably-collected "
                "evidence matching this family's registered "
                "target_evidence_contract_hash; collect_pair() must run with "
                "the exact target query set the family contract declares "
                "before proving an application-path write"
            )
        legitimate_relations = frozenset(
            (schema, relation)
            for schema, relation in (
                label.split(".", 1) for label in registered_row["relation_names"]
            )
        )
        legitimate_schemas = frozenset(
            schema for schema, _relation in legitimate_relations
        )

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

        referenced_relations = await self._explain_relations(verification_sql)
        if not referenced_relations:
            raise ValueError(
                "write verification query does not scan any relation; a "
                "write is not proven by a query that reads no target-bound "
                "table"
            )
        foreign_relations = referenced_relations - legitimate_relations
        if foreign_relations:
            raise ValueError(
                "write verification query reads relation(s) "
                f"{sorted(foreign_relations)} outside the durably-collected "
                f"target for binding {target_binding_ref!r}"
            )

    async def _explain_plan(self, sql: str) -> object:
        """Return PostgreSQL's own ``EXPLAIN (FORMAT JSON, VERBOSE)`` plan for ``sql``.

        Never a text-level parse of the caller's SQL -- the answer is the
        server's own understanding of the query, not something a caller can
        spoof by phrasing (comments, aliasing, literal-only queries).
        """
        plan_rows = await self._connection.fetch(
            f"EXPLAIN (FORMAT JSON, VERBOSE) {sql}"
        )
        if not plan_rows:
            raise RuntimeError("EXPLAIN returned no plan for the verification query")
        plan: object = json.loads(plan_rows[0][0])
        return plan

    async def _explain_schemas(self, sql: str) -> frozenset[str]:
        """Return every schema PostgreSQL's own planner says ``sql`` reads.

        Includes both relation-scan schemas and schemas referenced only
        through an opaque, schema-qualified function call embedded in an
        expression (see ``_collect_expression_schema_refs``) -- a query can
        "read" data through either path.
        """
        plan = await self._explain_plan(sql)
        schemas: set[str] = set()
        _collect_schema_names(plan, schemas)
        _collect_expression_schema_refs(plan, schemas)
        return frozenset(schemas)

    async def _explain_relations(self, sql: str) -> frozenset[tuple[str, str]]:
        """Return every ``(schema, relation)`` PostgreSQL's planner directly scans in ``sql``."""
        plan = await self._explain_plan(sql)
        relations: set[tuple[str, str]] = set()
        _collect_relations(plan, relations)
        return frozenset(relations)

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

        The target's relation(s) -- as PostgreSQL's own query plan reports
        them for the data-bearing ``keys_sql``/``rows_sql`` queries, never a
        caller string -- are durably registered against the pair
        ``(target_binding_ref, evidence_contract_hash)`` so
        ``verify_application_path_write()`` can later refuse a write proof
        minted against any other relation (e.g. an unrelated table in the
        same schema, or the source's).
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
        """Durably record which relations a target binding's real data lives in.

        Derived from PostgreSQL's own ``EXPLAIN`` plan of the data-bearing
        ``keys_sql``/``rows_sql`` queries only (never the auxiliary
        integrity-check queries, which legitimately touch ``pg_catalog``/
        ``information_schema`` and would otherwise leak those in as a
        false-positive "legitimate" target relation). Keyed by
        ``(target_binding_ref, evidence_contract_hash)`` -- the same
        content-addressed hash of ``queries`` that a registered family
        pins as its immutable ``target_evidence_contract_hash`` -- so a
        later call for the same ``target_binding_ref`` with different query
        content (e.g. attacker-supplied) lands in a distinct row instead of
        overwriting the row an already-registered family depends on. Runs
        outside the read-only snapshot transaction because it durably
        writes.
        """
        relations = (await self._explain_relations(queries.keys_sql)) | (
            await self._explain_relations(queries.rows_sql)
        )
        if not relations:
            raise ValueError(
                f"target binding {target_binding_ref!r} evidence queries "
                "reference no relation; cannot durably establish its target "
                "relation"
            )
        evidence_contract_hash = self._query_contract_hash(queries)
        relation_names = sorted(
            f"{schema}.{relation}" for schema, relation in relations
        )
        await self._connection.execute(
            """
INSERT INTO omninode_internal.target_binding_schemas
  (target_binding_ref, evidence_contract_hash, relation_names)
VALUES ($1, $2, $3)
ON CONFLICT (target_binding_ref, evidence_contract_hash)
DO UPDATE SET relation_names = EXCLUDED.relation_names, registered_at = clock_timestamp()
""",
            target_binding_ref,
            evidence_contract_hash,
            relation_names,
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
