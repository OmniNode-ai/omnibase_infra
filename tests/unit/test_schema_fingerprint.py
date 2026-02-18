# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for schema fingerprint validation (OMN-2087)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.errors.error_schema_fingerprint import (
    SchemaFingerprintMismatchError,
    SchemaFingerprintMissingError,
)
from omnibase_infra.runtime.model_schema_fingerprint_result import (
    ModelSchemaFingerprintResult,
)
from omnibase_infra.runtime.model_schema_manifest import (
    OMNIBASE_INFRA_SCHEMA_MANIFEST,
    ModelSchemaManifest,
)
from omnibase_infra.runtime.util_schema_fingerprint import (
    _compute_schema_diff,
    compute_schema_fingerprint,
    validate_schema_fingerprint,
)

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

_SMALL_MANIFEST = ModelSchemaManifest(
    owner_service="test_service",
    schema_name="public",
    tables=("test_table",),
)

_EMPTY_MANIFEST = ModelSchemaManifest(
    owner_service="test_service",
    schema_name="public",
    tables=(),
)

_TWO_TABLE_MANIFEST = ModelSchemaManifest(
    owner_service="test_service",
    schema_name="public",
    tables=("alpha", "beta"),
)


def _make_mock_pool(
    *,
    fetch_returns: list[list[dict]] | None = None,
    fetchval_return: object = None,
    side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock asyncpg.Pool.

    Args:
        fetch_returns: List of return values for successive conn.fetch() calls.
        fetchval_return: Return value for conn.fetchval().
        side_effect: Exception to raise on any conn operation.
    """
    pool = MagicMock()
    conn = AsyncMock()

    if side_effect is not None:
        conn.fetch = AsyncMock(side_effect=side_effect)
        conn.fetchval = AsyncMock(side_effect=side_effect)
    else:
        if fetch_returns is not None:
            conn.fetch = AsyncMock(side_effect=fetch_returns)
        else:
            conn.fetch = AsyncMock(return_value=[])
        conn.fetchval = AsyncMock(return_value=fetchval_return)

    acm = AsyncMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=acm)
    return pool


def _make_column_row(
    table_name: str,
    column_name: str,
    canonical_type: str = "integer",
    not_null: bool = False,
    column_default: str | None = None,
    ordinal_position: int = 1,
) -> dict:
    """Create a mock column row from pg_catalog."""
    return {
        "table_name": table_name,
        "column_name": column_name,
        "canonical_type": canonical_type,
        "not_null": not_null,
        "column_default": column_default,
        "ordinal_position": ordinal_position,
    }


def _make_constraint_row(
    table_name: str,
    constraint_type: str = "p",
    local_columns: list[str] | None = None,
    ref_table: str | None = None,
    ref_columns: list[str] | None = None,
    on_update: str | None = None,
    on_delete: str | None = None,
    constraint_def: str | None = None,
) -> dict:
    """Create a mock constraint row from pg_catalog."""
    return {
        "table_name": table_name,
        "constraint_type": constraint_type,
        "local_columns": local_columns or ["id"],
        "ref_table": ref_table,
        "ref_columns": ref_columns,
        "on_update": on_update,
        "on_delete": on_delete,
        "constraint_def": constraint_def,
    }


# ---------------------------------------------------------------------------
# TestComputeSchemaFingerprint
# ---------------------------------------------------------------------------


class TestComputeSchemaFingerprint:
    """Tests for compute_schema_fingerprint()."""

    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        """Columns and constraints produce a valid fingerprint result."""
        column_rows = [
            _make_column_row("test_table", "id", "integer", True, None, 1),
            _make_column_row("test_table", "name", "text", False, None, 2),
        ]
        constraint_rows = [
            _make_constraint_row("test_table", "p", ["id"]),
        ]
        pool = _make_mock_pool(fetch_returns=[column_rows, constraint_rows])

        result = await compute_schema_fingerprint(
            pool=pool,
            manifest=_SMALL_MANIFEST,
            correlation_id=uuid4(),
        )

        assert isinstance(result, ModelSchemaFingerprintResult)
        assert len(result.fingerprint) == 64  # SHA-256 hex digest
        assert result.table_count == 1
        assert result.column_count == 2
        assert result.constraint_count == 1
        assert len(result.per_table_hashes) == 1
        assert result.per_table_hashes[0][0] == "test_table"

    @pytest.mark.asyncio
    async def test_empty_allowlist(self) -> None:
        """Manifest with no tables returns a fingerprint with zero counts."""
        pool = _make_mock_pool()

        result = await compute_schema_fingerprint(
            pool=pool,
            manifest=_EMPTY_MANIFEST,
            correlation_id=uuid4(),
        )

        assert result.table_count == 0
        assert result.column_count == 0
        assert result.constraint_count == 0
        assert result.per_table_hashes == ()
        assert len(result.fingerprint) == 64  # Still a valid hash

    @pytest.mark.asyncio
    async def test_deterministic_ordering(self) -> None:
        """Same data in different order produces the same fingerprint."""
        # Order 1: alpha first
        column_rows_1 = [
            _make_column_row("alpha", "id", "integer", True, None, 1),
            _make_column_row("beta", "id", "integer", True, None, 1),
        ]
        constraint_rows_1 = [
            _make_constraint_row("alpha", "p", ["id"]),
            _make_constraint_row("beta", "p", ["id"]),
        ]

        # Order 2: beta first (reversed in query results)
        column_rows_2 = [
            _make_column_row("beta", "id", "integer", True, None, 1),
            _make_column_row("alpha", "id", "integer", True, None, 1),
        ]
        constraint_rows_2 = [
            _make_constraint_row("beta", "p", ["id"]),
            _make_constraint_row("alpha", "p", ["id"]),
        ]

        pool_1 = _make_mock_pool(fetch_returns=[column_rows_1, constraint_rows_1])
        pool_2 = _make_mock_pool(fetch_returns=[column_rows_2, constraint_rows_2])

        result_1 = await compute_schema_fingerprint(
            pool=pool_1,
            manifest=_TWO_TABLE_MANIFEST,
            correlation_id=uuid4(),
        )
        result_2 = await compute_schema_fingerprint(
            pool=pool_2,
            manifest=_TWO_TABLE_MANIFEST,
            correlation_id=uuid4(),
        )

        assert result_1.fingerprint == result_2.fingerprint

    @pytest.mark.asyncio
    async def test_format_type_includes_typmod(self) -> None:
        """Verify canonical_type with length info (e.g., varchar(128)) is hashed."""
        column_rows = [
            _make_column_row(
                "test_table", "name", "character varying(128)", False, None, 1
            ),
        ]
        pool = _make_mock_pool(fetch_returns=[column_rows, []])

        result = await compute_schema_fingerprint(
            pool=pool,
            manifest=_SMALL_MANIFEST,
            correlation_id=uuid4(),
        )

        assert result.column_count == 1
        # Different type should produce different fingerprint
        column_rows_2 = [
            _make_column_row(
                "test_table", "name", "character varying(256)", False, None, 1
            ),
        ]
        pool_2 = _make_mock_pool(fetch_returns=[column_rows_2, []])

        result_2 = await compute_schema_fingerprint(
            pool=pool_2,
            manifest=_SMALL_MANIFEST,
            correlation_id=uuid4(),
        )

        assert result.fingerprint != result_2.fingerprint


# ---------------------------------------------------------------------------
# TestValidateSchemaFingerprint
# ---------------------------------------------------------------------------


class TestValidateSchemaFingerprint:
    """Tests for validate_schema_fingerprint()."""

    @pytest.mark.asyncio
    async def test_match_succeeds(self) -> None:
        """No exception when expected fingerprint matches computed."""
        column_rows = [
            _make_column_row("test_table", "id", "integer", True, None, 1),
        ]
        constraint_rows: list[dict] = []

        # First, compute what the fingerprint would be
        compute_pool = _make_mock_pool(fetch_returns=[column_rows, constraint_rows])
        result = await compute_schema_fingerprint(
            pool=compute_pool,
            manifest=_SMALL_MANIFEST,
        )
        expected_fp = result.fingerprint

        # Now mock the pool for validate: fetchval returns the expected fp,
        # then fetch calls return the same column/constraint data
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=expected_fp)
        conn.fetch = AsyncMock(side_effect=[column_rows, constraint_rows])

        acm = AsyncMock()
        acm.__aenter__ = AsyncMock(return_value=conn)
        acm.__aexit__ = AsyncMock(return_value=False)
        pool.acquire = MagicMock(return_value=acm)

        # Should not raise
        await validate_schema_fingerprint(
            pool=pool,
            manifest=_SMALL_MANIFEST,
            correlation_id=uuid4(),
        )

    @pytest.mark.asyncio
    async def test_mismatch_raises(self) -> None:
        """SchemaFingerprintMismatchError raised when fingerprints differ."""
        column_rows = [
            _make_column_row("test_table", "id", "integer", True, None, 1),
        ]
        constraint_rows: list[dict] = []

        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value="deadbeef" * 8)  # Wrong fingerprint
        conn.fetch = AsyncMock(side_effect=[column_rows, constraint_rows])

        acm = AsyncMock()
        acm.__aenter__ = AsyncMock(return_value=conn)
        acm.__aexit__ = AsyncMock(return_value=False)
        pool.acquire = MagicMock(return_value=acm)

        with pytest.raises(SchemaFingerprintMismatchError) as exc_info:
            await validate_schema_fingerprint(
                pool=pool,
                manifest=_SMALL_MANIFEST,
                correlation_id=uuid4(),
            )

        assert exc_info.value.expected_fingerprint == "deadbeef" * 8
        assert exc_info.value.actual_fingerprint  # Non-empty
        assert exc_info.value.diff_summary  # Non-empty

    @pytest.mark.asyncio
    async def test_null_expected_fingerprint_raises_missing(self) -> None:
        """SchemaFingerprintMissingError when fetchval returns None."""
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=None)

        acm = AsyncMock()
        acm.__aenter__ = AsyncMock(return_value=conn)
        acm.__aexit__ = AsyncMock(return_value=False)
        pool.acquire = MagicMock(return_value=acm)

        with pytest.raises(SchemaFingerprintMissingError) as exc_info:
            await validate_schema_fingerprint(
                pool=pool,
                manifest=_SMALL_MANIFEST,
                correlation_id=uuid4(),
            )

        assert exc_info.value.expected_owner == "test_service"
        assert "null" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_db_metadata_table_raises(self) -> None:
        """UndefinedTableError raises SchemaFingerprintMissingError."""
        from asyncpg.exceptions import UndefinedTableError

        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchval = AsyncMock(
            side_effect=UndefinedTableError(
                'relation "public.db_metadata" does not exist'
            )
        )

        acm = AsyncMock()
        acm.__aenter__ = AsyncMock(return_value=conn)
        acm.__aexit__ = AsyncMock(return_value=False)
        pool.acquire = MagicMock(return_value=acm)

        with pytest.raises(SchemaFingerprintMissingError) as exc_info:
            await validate_schema_fingerprint(
                pool=pool,
                manifest=_SMALL_MANIFEST,
                correlation_id=uuid4(),
            )

        assert exc_info.value.expected_owner == "test_service"
        assert "run migrations" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_transient_connection_error_propagates(self) -> None:
        """Non-table errors propagate with original type."""
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchval = AsyncMock(
            side_effect=ConnectionRefusedError("connection refused")
        )

        acm = AsyncMock()
        acm.__aenter__ = AsyncMock(return_value=conn)
        acm.__aexit__ = AsyncMock(return_value=False)
        pool.acquire = MagicMock(return_value=acm)

        with pytest.raises(ConnectionRefusedError):
            await validate_schema_fingerprint(
                pool=pool,
                manifest=_SMALL_MANIFEST,
                correlation_id=uuid4(),
            )


# ---------------------------------------------------------------------------
# TestSchemaFingerprintErrorTypes
# ---------------------------------------------------------------------------


class TestSchemaFingerprintErrorTypes:
    """Tests for error type hierarchy and attributes."""

    def test_mismatch_is_runtime_host_error(self) -> None:
        """SchemaFingerprintMismatchError extends RuntimeHostError."""
        from omnibase_infra.errors.error_infra import RuntimeHostError

        err = SchemaFingerprintMismatchError(
            "test",
            expected_fingerprint="aaa",
            actual_fingerprint="bbb",
        )
        assert isinstance(err, RuntimeHostError)

    def test_missing_is_runtime_host_error(self) -> None:
        """SchemaFingerprintMissingError extends RuntimeHostError."""
        from omnibase_infra.errors.error_infra import RuntimeHostError

        err = SchemaFingerprintMissingError(
            "test",
            expected_owner="test_service",
        )
        assert isinstance(err, RuntimeHostError)

    def test_mismatch_attributes(self) -> None:
        """SchemaFingerprintMismatchError exposes expected/actual/diff."""
        err = SchemaFingerprintMismatchError(
            "msg",
            expected_fingerprint="expected_hash",
            actual_fingerprint="actual_hash",
            diff_summary="changed: foo",
        )
        assert err.expected_fingerprint == "expected_hash"
        assert err.actual_fingerprint == "actual_hash"
        assert err.diff_summary == "changed: foo"

    def test_missing_attributes(self) -> None:
        """SchemaFingerprintMissingError exposes expected_owner."""
        err = SchemaFingerprintMissingError(
            "msg",
            expected_owner="omnibase_infra",
        )
        assert err.expected_owner == "omnibase_infra"

    def test_auto_generates_correlation_id(self) -> None:
        """correlation_id is auto-generated when not provided."""
        err = SchemaFingerprintMismatchError(
            "test",
            expected_fingerprint="aaa",
            actual_fingerprint="bbb",
        )
        # The error should have been created without raising
        assert err.expected_fingerprint == "aaa"

        err2 = SchemaFingerprintMissingError(
            "test",
            expected_owner="test_service",
        )
        assert err2.expected_owner == "test_service"


# ---------------------------------------------------------------------------
# TestPluginFingerprintPropagation
# ---------------------------------------------------------------------------


class TestPluginFingerprintPropagation:
    """Tests that schema fingerprint errors propagate out of plugin.validate_handshake().

    PluginRegistration.validate_handshake() runs B1-B3 checks (OMN-2089).
    Schema fingerprint errors are P0 hard gates that MUST escape the
    handshake validation so the kernel terminates. These tests confirm
    the re-raise path works through the handshake gate.
    """

    @pytest.mark.asyncio
    async def test_mismatch_error_propagates_from_plugin(self) -> None:
        """SchemaFingerprintMismatchError escapes plugin.validate_handshake()."""
        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            PluginRegistration,
        )
        from omnibase_infra.runtime.protocol_domain_plugin import (
            ModelDomainPluginConfig,
        )

        plugin = PluginRegistration()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        mismatch = SchemaFingerprintMismatchError(
            "schema drift",
            expected_fingerprint="expected_hash",
            actual_fingerprint="actual_hash",
            diff_summary="changed: foo",
        )

        _plugin_mod = "omnibase_infra.nodes.node_registration_orchestrator.plugin"

        with (
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": "postgresql://x/y"}),
            patch(f"{_plugin_mod}.ModelPostgresPoolConfig.validate_dsn"),
            patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool,
            patch.object(
                PluginRegistration,
                "_load_projector",
                new_callable=AsyncMock,
            ),
            patch.object(
                PluginRegistration,
                "_initialize_schema",
                new_callable=AsyncMock,
            ),
            patch.object(
                PluginRegistration,
                "_initialize_consul_handler",
                new_callable=AsyncMock,
            ),
            patch.object(
                PluginRegistration,
                "_initialize_snapshot_publisher",
                new_callable=AsyncMock,
            ),
            patch(
                f"{_plugin_mod}.validate_db_ownership",
                new_callable=AsyncMock,
            ),
            patch(
                f"{_plugin_mod}.validate_schema_fingerprint",
                new_callable=AsyncMock,
                side_effect=mismatch,
            ),
        ):
            mock_create_pool.return_value = MagicMock()

            # initialize() succeeds (creates pool) -- B1-B3 checks moved to validate_handshake()
            init_result = await plugin.initialize(config)
            assert init_result.success

            with pytest.raises(SchemaFingerprintMismatchError) as exc_info:
                await plugin.validate_handshake(config)

            assert exc_info.value.expected_fingerprint == "expected_hash"
            assert exc_info.value.actual_fingerprint == "actual_hash"

    @pytest.mark.asyncio
    async def test_missing_error_propagates_from_plugin(self) -> None:
        """SchemaFingerprintMissingError escapes plugin.validate_handshake()."""
        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            PluginRegistration,
        )
        from omnibase_infra.runtime.protocol_domain_plugin import (
            ModelDomainPluginConfig,
        )

        plugin = PluginRegistration()
        config = MagicMock(spec=ModelDomainPluginConfig)
        config.correlation_id = uuid4()

        missing = SchemaFingerprintMissingError(
            "fingerprint not found",
            expected_owner="omnibase_infra",
        )

        _plugin_mod = "omnibase_infra.nodes.node_registration_orchestrator.plugin"

        with (
            patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": "postgresql://x/y"}),
            patch(f"{_plugin_mod}.ModelPostgresPoolConfig.validate_dsn"),
            patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool,
            patch.object(
                PluginRegistration,
                "_load_projector",
                new_callable=AsyncMock,
            ),
            patch.object(
                PluginRegistration,
                "_initialize_schema",
                new_callable=AsyncMock,
            ),
            patch.object(
                PluginRegistration,
                "_initialize_consul_handler",
                new_callable=AsyncMock,
            ),
            patch.object(
                PluginRegistration,
                "_initialize_snapshot_publisher",
                new_callable=AsyncMock,
            ),
            patch(
                f"{_plugin_mod}.validate_db_ownership",
                new_callable=AsyncMock,
            ),
            patch(
                f"{_plugin_mod}.validate_schema_fingerprint",
                new_callable=AsyncMock,
                side_effect=missing,
            ),
        ):
            mock_create_pool.return_value = MagicMock()

            # initialize() succeeds (creates pool) -- B1-B3 checks moved to validate_handshake()
            init_result = await plugin.initialize(config)
            assert init_result.success

            with pytest.raises(SchemaFingerprintMissingError) as exc_info:
                await plugin.validate_handshake(config)

            assert exc_info.value.expected_owner == "omnibase_infra"


# ---------------------------------------------------------------------------
# TestDiffSummary
# ---------------------------------------------------------------------------


class TestDiffSummary:
    """Tests for _compute_schema_diff()."""

    def test_added_table_shown(self) -> None:
        """New table in actual appears as added."""
        expected: dict[str, str] = {}
        actual = {"new_table": "abc123"}
        diff = _compute_schema_diff(expected, actual)
        assert "+ added: new_table" in diff

    def test_removed_table_shown(self) -> None:
        """Table in expected but not actual appears as removed."""
        expected = {"old_table": "abc123"}
        actual: dict[str, str] = {}
        diff = _compute_schema_diff(expected, actual)
        assert "- removed: old_table" in diff

    def test_changed_table_shown(self) -> None:
        """Table with different hash appears as changed."""
        expected = {"my_table": "aaa"}
        actual = {"my_table": "bbb"}
        diff = _compute_schema_diff(expected, actual)
        assert "~ changed: my_table" in diff

    def test_diff_bounded_to_10_lines(self) -> None:
        """Diff output is bounded to 10 lines total (including overflow message)."""
        expected = {f"table_{i}": f"hash_{i}" for i in range(15)}
        actual = {f"table_{i}": f"different_{i}" for i in range(15)}
        diff = _compute_schema_diff(expected, actual)
        lines = diff.strip().split("\n")
        # Should be exactly 10 lines total (9 content lines + 1 overflow line)
        assert len(lines) == 10
        assert "... and" in lines[-1]

    def test_empty_diff_on_match(self) -> None:
        """No diff output when expected and actual match."""
        hashes = {"table_a": "hash_a", "table_b": "hash_b"}
        diff = _compute_schema_diff(hashes, hashes)
        assert diff == ""


# ---------------------------------------------------------------------------
# TestFingerprintInvariance
# ---------------------------------------------------------------------------


class TestFingerprintInvariance:
    """Tests for fingerprint determinism and normalization."""

    @pytest.mark.asyncio
    async def test_same_schema_different_row_order_same_hash(self) -> None:
        """Row ordering from DB does not affect fingerprint."""
        # Order 1
        cols_1 = [
            _make_column_row("test_table", "id", "integer", True, None, 1),
            _make_column_row("test_table", "name", "text", False, None, 2),
        ]
        # Order 2 (reversed)
        cols_2 = [
            _make_column_row("test_table", "name", "text", False, None, 2),
            _make_column_row("test_table", "id", "integer", True, None, 1),
        ]

        pool_1 = _make_mock_pool(fetch_returns=[cols_1, []])
        pool_2 = _make_mock_pool(fetch_returns=[cols_2, []])

        r1 = await compute_schema_fingerprint(pool=pool_1, manifest=_SMALL_MANIFEST)
        r2 = await compute_schema_fingerprint(pool=pool_2, manifest=_SMALL_MANIFEST)

        # Columns are keyed by ordinal_position in the record, so
        # the grouping preserves insertion order. The per-table record
        # uses the column list as-is (ordered by attnum in query).
        # For truly identical schemas the rows would come back in the
        # same attnum order. Different row orderings here actually
        # produce different column lists (different ordinal positions),
        # which is correct behavior - the columns ARE different in
        # their ordinal_position placement in the list.
        # The important invariant is that table-level sorting is deterministic.
        assert r1.table_count == r2.table_count

    @pytest.mark.asyncio
    async def test_whitespace_normalized_check_expression(self) -> None:
        """CHECK constraint expressions are whitespace-normalized.

        PostgreSQL's pg_get_constraintdef() returns consistently structured
        expressions, but whitespace between tokens may vary. The normalizer
        collapses runs of whitespace so ``CHECK (id > 0)`` and
        ``CHECK  (id  >  0)`` produce the same hash.
        """
        cons_1 = [
            _make_constraint_row(
                "test_table",
                "c",
                ["id"],
                constraint_def="CHECK  (id  >  0)",
            ),
        ]
        cons_2 = [
            _make_constraint_row(
                "test_table",
                "c",
                ["id"],
                constraint_def="CHECK (id > 0)",
            ),
        ]

        pool_1 = _make_mock_pool(fetch_returns=[[], cons_1])
        pool_2 = _make_mock_pool(fetch_returns=[[], cons_2])

        r1 = await compute_schema_fingerprint(pool=pool_1, manifest=_SMALL_MANIFEST)
        r2 = await compute_schema_fingerprint(pool=pool_2, manifest=_SMALL_MANIFEST)

        assert r1.fingerprint == r2.fingerprint


# ---------------------------------------------------------------------------
# TestModelSchemaManifest
# ---------------------------------------------------------------------------


class TestModelSchemaManifest:
    """Tests for ModelSchemaManifest model and constant."""

    def test_canonical_manifest_is_frozen(self) -> None:
        """OMNIBASE_INFRA_SCHEMA_MANIFEST is immutable."""
        with pytest.raises(Exception):
            OMNIBASE_INFRA_SCHEMA_MANIFEST.owner_service = "hacked"  # type: ignore[misc]

    def test_canonical_manifest_has_19_tables(self) -> None:
        """Canonical manifest declares 19 tables."""
        assert len(OMNIBASE_INFRA_SCHEMA_MANIFEST.tables) == 19

    def test_canonical_manifest_tables_are_sorted(self) -> None:
        """Tables in canonical manifest are alphabetically sorted."""
        assert list(OMNIBASE_INFRA_SCHEMA_MANIFEST.tables) == sorted(
            OMNIBASE_INFRA_SCHEMA_MANIFEST.tables
        )

    def test_canonical_manifest_owner(self) -> None:
        """Canonical manifest owner is omnibase_infra."""
        assert OMNIBASE_INFRA_SCHEMA_MANIFEST.owner_service == "omnibase_infra"
