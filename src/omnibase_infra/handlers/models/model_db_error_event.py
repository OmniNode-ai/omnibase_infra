# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Model for PostgreSQL error events emitted to Kafka.

Produced by the postgres error emitter in scripts/monitor_logs.py when a
PostgreSQL ERROR log line is detected. Events are deduplicated via Valkey
using a SHA-256 fingerprint before publish.

Topic: ``onex.evt.omnibase-infra.db-error.v1`` (TOPIC_DB_ERROR_V1)

Related Tickets:
    - OMN-3407: PostgreSQL error event emitter
    - OMN-3408: Kafka consumer -> Linear ticket reporter
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ModelDbErrorEvent(BaseModel):
    """Structured event representing a single PostgreSQL error occurrence.

    Fields map directly to the parsed content of a PostgreSQL multi-line
    error block. All fields except ``error_message``, ``fingerprint``,
    ``first_seen_at``, and ``service`` are optional — PostgreSQL logs vary
    in which supplementary fields are present.
    """

    error_code: str | None = None
    """SQLSTATE error code if present in the log line (e.g. ``"42883"``).

    PostgreSQL prefixes some ERROR lines with the SQLSTATE in square brackets,
    e.g. ``ERROR:  [42883] operator does not exist``. ``None`` when the prefix
    is absent.
    """

    error_message: str
    """Primary error message text extracted from the ERROR: log line.

    Always present. Example: ``"operator does not exist: character varying = uuid"``
    """

    hint: str | None = None
    """Optional HINT field captured from the error block.

    PostgreSQL appends ``HINT:`` lines to some errors to suggest corrective
    action. ``None`` when not present in the captured block.
    """

    detail: str | None = None
    """Optional DETAIL field captured from the error block.

    PostgreSQL appends ``DETAIL:`` lines with additional context for some
    errors. ``None`` when not present.
    """

    sql_statement: str | None = None
    """Optional SQL statement captured from the STATEMENT field of the error block.

    PostgreSQL includes the originating SQL statement for some error types.
    ``None`` when not present.
    """

    table_name: str | None = None
    """Table name extracted via regex from the error message or SQL statement.

    Best-effort extraction — ``None`` when no table name can be identified.
    """

    fingerprint: str
    """SHA-256 fingerprint of the normalized error fields, truncated to 32 chars.

    Computed as::

        sha256(f"{error_code}:{normalize(error_message)}:{table_name or ''}:{normalize(sql_statement or '')}")[:32]

    Where ``normalize()`` strips leading/trailing whitespace, collapses internal
    whitespace, and strips SQL string literals.

    Used as the Valkey dedup key to prevent duplicate Kafka events for the same
    recurring error. The key is set in Valkey **only after** a successful Kafka
    publish.
    """

    first_seen_at: datetime
    """UTC timestamp of the log line that triggered this error block capture."""

    service: str
    """Docker container name or prefix that produced the error.

    Populated from the container name returned by Docker discovery (e.g.
    ``"omnibase-infra-postgres"``).
    """
