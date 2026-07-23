# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""B12<->B6 seam regression: canary DDL columns must match the exported seam
fingerprint field-by-field (OMN-14779).

This is the omnibase_infra half of a two-repo cross-boundary regression test.
It parses the *real* migration SQL text (no live DB, no ORM reflection) and
asserts it matches ``src/omnibase_infra/contracts/canary/model_replay_projection_seam.json``
column-by-column: name, type, nullability, primary-key membership.

The seam JSON is packaged (``src/omnibase_infra/contracts/...``, not
``docker/...``) specifically so the omnimarket-side companion test can import
it as installed package data and assert the SAME fingerprint against the real
``ModelReplayProjection`` / ``ModelReplayCursor`` Pydantic ``model_fields`` --
closing the loop without a live DB and without omnibase_infra depending on
omnimarket (forbidden by repo layering).

If this test is green but the omnimarket-side companion is red (or vice
versa), the seam has drifted on one side only -- fix the seam JSON (if the
drift is intentional) or the drifted artifact (if it is not), never widen
this test to paper over a real mismatch.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

JSONDict = dict[str, Any]

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DDL_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "canary"
    / "forward"
    / "001_create_delivery_replay_canary_projection.sql"
)
TABLE_NAME = "delivery_replay_canary_projection"

_CONSTRAINT_KEYWORDS = {"PRIMARY", "UNIQUE", "CONSTRAINT", "CHECK", "FOREIGN"}


def _load_seam() -> JSONDict:
    text = (
        files("omnibase_infra.contracts.canary")
        .joinpath("model_replay_projection_seam.json")
        .read_text(encoding="utf-8")
    )
    parsed: JSONDict = json.loads(text)
    return parsed


def _extract_create_table_body(sql: str, table_name: str) -> str:
    """Return the raw column-list text between the outer CREATE TABLE parens."""
    pattern = re.compile(
        rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{re.escape(table_name)}\s*\(",
        re.IGNORECASE,
    )
    match = pattern.search(sql)
    assert match is not None, f"CREATE TABLE {table_name} not found in {DDL_FILE}"
    start = match.end()
    depth = 1
    pos = start
    while pos < len(sql) and depth > 0:
        if sql[pos] == "(":
            depth += 1
        elif sql[pos] == ")":
            depth -= 1
        pos += 1
    assert depth == 0, "Unbalanced parens while scanning CREATE TABLE body"
    return sql[start : pos - 1]


def _split_top_level_commas(text: str) -> list[str]:
    """Split on commas not nested inside parens or single-quoted strings."""
    segments: list[str] = []
    depth = 0
    in_quote = False
    current: list[str] = []
    for ch in text:
        if ch == "'":
            in_quote = not in_quote
            current.append(ch)
        elif in_quote:
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            segments.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        segments.append("".join(current))
    return segments


def _parse_ddl_columns(body: str) -> dict[str, dict[str, object]]:
    """Parse a CREATE TABLE column-list body into per-column facts.

    Returns a mapping ``column_name -> {"type": str, "nullable": bool,
    "primary_key": bool}``. Handles both inline ``PRIMARY KEY`` on a column
    definition and a table-level ``CONSTRAINT ... PRIMARY KEY (col, ...)``.
    """
    columns: dict[str, dict[str, object]] = {}
    pk_columns: set[str] = set()

    # Strip `--` line comments from the WHOLE body before splitting on commas.
    # Comment text frequently contains commas of its own (e.g. "per-(topic,
    # partition) terminal offsets"), which would otherwise corrupt top-level
    # comma splitting if stripped per-segment after the split.
    uncommented_body = re.sub(r"--[^\n]*", "", body)

    for raw_segment in _split_top_level_commas(uncommented_body):
        segment = raw_segment.strip()
        if not segment:
            continue
        first_word = segment.split()[0].upper()

        if first_word in _CONSTRAINT_KEYWORDS:
            pk_match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", segment, re.IGNORECASE)
            if pk_match:
                for col in pk_match.group(1).split(","):
                    pk_columns.add(col.strip().strip('"'))
            continue

        tokens = segment.split(None, 1)
        col_name = tokens[0].strip('"')
        rest = tokens[1] if len(tokens) > 1 else ""

        type_match = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*(?:\s*\([^)]*\))?)", rest)
        col_type = type_match.group(1).strip() if type_match else ""
        # Normalize whitespace inside e.g. "NUMERIC(10, 3)"
        col_type = re.sub(r"\s+", " ", col_type).strip().upper()

        nullable = re.search(r"\bNOT\s+NULL\b", rest, re.IGNORECASE) is None
        inline_pk = re.search(r"\bPRIMARY\s+KEY\b", rest, re.IGNORECASE) is not None

        columns[col_name] = {
            "type": col_type,
            "nullable": nullable,
            "primary_key": inline_pk,
        }

    for col in pk_columns:
        if col in columns:
            columns[col]["primary_key"] = True

    return columns


@pytest.fixture(scope="module")
def seam() -> JSONDict:
    return _load_seam()


@pytest.fixture(scope="module")
def ddl_columns() -> dict[str, dict[str, object]]:
    assert DDL_FILE.exists(), f"Canary DDL migration missing: {DDL_FILE}"
    sql = DDL_FILE.read_text(encoding="utf-8")
    body = _extract_create_table_body(sql, TABLE_NAME)
    return _parse_ddl_columns(body)


@pytest.mark.unit
class TestSeamArtifactShape:
    def test_seam_declares_the_canary_table(self, seam: JSONDict) -> None:
        assert seam["ddl_table"] == TABLE_NAME
        assert seam["ticket"] == "OMN-14779"

    def test_seam_has_at_least_one_field(self, seam: JSONDict) -> None:
        assert len(seam["fields"]) > 0


@pytest.mark.unit
class TestDdlMatchesSeamFieldByField:
    """Drives the real seam: parsed DDL vs the exported fingerprint."""

    def test_every_seam_column_exists_in_ddl(
        self, seam: JSONDict, ddl_columns: dict[str, dict[str, object]]
    ) -> None:
        for field in seam["fields"]:
            assert field["ddl_column"] in ddl_columns, (
                f"Seam declares column {field['ddl_column']!r} "
                f"(from {field['source_field_path']}) but it is absent from "
                f"the parsed DDL -- DDL has drifted from the seam."
            )

    def test_seam_type_matches_ddl_type(
        self, seam: JSONDict, ddl_columns: dict[str, dict[str, object]]
    ) -> None:
        for field in seam["fields"]:
            actual = ddl_columns[field["ddl_column"]]
            assert actual["type"] == field["ddl_type"].upper(), (
                f"{field['ddl_column']}: DDL type {actual['type']!r} != "
                f"seam-declared type {field['ddl_type']!r}"
            )

    def test_seam_nullability_matches_ddl_nullability(
        self, seam: JSONDict, ddl_columns: dict[str, dict[str, object]]
    ) -> None:
        for field in seam["fields"]:
            actual = ddl_columns[field["ddl_column"]]
            assert actual["nullable"] == field["ddl_nullable"], (
                f"{field['ddl_column']}: DDL nullable={actual['nullable']} != "
                f"seam-declared ddl_nullable={field['ddl_nullable']}"
            )

    def test_seam_primary_key_matches_ddl_primary_key(
        self, seam: JSONDict, ddl_columns: dict[str, dict[str, object]]
    ) -> None:
        for field in seam["fields"]:
            actual = ddl_columns[field["ddl_column"]]
            assert actual["primary_key"] == field["ddl_primary_key"], (
                f"{field['ddl_column']}: DDL primary_key={actual['primary_key']} != "
                f"seam-declared ddl_primary_key={field['ddl_primary_key']}"
            )

    def test_no_undeclared_model_mapped_columns_in_ddl(
        self, seam: JSONDict, ddl_columns: dict[str, dict[str, object]]
    ) -> None:
        """Every DDL column must be either seam-mapped or explicitly declared
        operational-only. Catches a new column added to the DDL without an
        accompanying seam-JSON update (drift in the DDL -> model direction)."""
        declared = {field["ddl_column"] for field in seam["fields"]}
        declared |= set(seam.get("operational_only_ddl_columns", ()))
        undeclared = set(ddl_columns) - declared
        assert not undeclared, (
            f"DDL columns {undeclared} are not represented in the seam JSON "
            f"(neither model-mapped nor declared operational-only) -- update "
            f"model_replay_projection_seam.json in the same PR that changes the DDL."
        )


@pytest.mark.unit
class TestCorrelationIdNarrowingIsIntentional:
    """OMN-14779 acceptance #2: correlation_id is UUID|None on the pure B6
    model but NOT NULL PK on the landing table. Assert this narrowing is
    flagged, justified, and mechanically enforced -- not a silent hole."""

    def test_correlation_id_seam_entry_flags_narrowing(self, seam: JSONDict) -> None:
        entries = [f for f in seam["fields"] if f["ddl_column"] == "correlation_id"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["source_field_path"] == "ModelReplayProjection.correlation_id"
        assert entry["source_nullable"] is True
        assert entry["narrowing"] is True
        assert entry["narrowing_reason"], "narrowing_reason must not be empty"

    def test_correlation_id_ddl_is_not_null_primary_key(
        self, ddl_columns: dict[str, dict[str, object]]
    ) -> None:
        actual = ddl_columns["correlation_id"]
        assert actual["type"] == "UUID"
        assert actual["nullable"] is False, (
            "correlation_id must be NOT NULL in the landing table -- a null "
            "correlation_id row must never be accepted by the canary seam."
        )
        assert actual["primary_key"] is True


@pytest.mark.unit
class TestAcceptanceCoveredFields:
    """OMN-14779 acceptance #2 explicit field list: cursor fields, checksum,
    compared/diverged/divergence_reasons must all be present in the seam."""

    @pytest.mark.parametrize(
        "ddl_column",
        [
            "cursor_token",
            "cursor_positions",
            "cursor_event_count",
            "projection_checksum",
            "compared",
            "diverged",
            "divergence_reasons",
        ],
    )
    def test_field_present_in_seam(self, seam: JSONDict, ddl_column: str) -> None:
        columns = {f["ddl_column"] for f in seam["fields"]}
        assert ddl_column in columns
