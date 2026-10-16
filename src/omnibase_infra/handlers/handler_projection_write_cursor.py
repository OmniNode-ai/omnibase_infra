# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed statement boundary around an already attested driver cursor."""

from collections.abc import Mapping, Sequence

from omnibase_infra.protocols.protocol_projection_driver_cursor import (
    ProtocolProjectionDriverCursor,
)


class HandlerProjectionWriteCursor:
    """Bind the driver's SQL methods without exposing untyped driver results."""

    def __init__(self, cursor: ProtocolProjectionDriverCursor) -> None:
        self._execute_statement = cursor.execute
        self._fetch_rows = cursor.fetchall

    def _returning_rows(
        self,
        statement: object,
        parameters: Mapping[str, object],
        returning: Sequence[str],
    ) -> list[dict[str, object]]:
        self._execute_statement(statement, parameters)
        result: list[dict[str, object]] = []
        for record in self._fetch_rows():
            if isinstance(record, Mapping):
                result.append(dict(record))
            elif isinstance(record, Sequence):
                result.append(dict(zip(returning, record, strict=True)))
            else:
                raise TypeError("projection statement returned an invalid row")
        return result

    def run_projection_upsert(
        self,
        statement: object,
        parameters: Mapping[str, object],
        *,
        returning: Sequence[str],
    ) -> list[dict[str, object]]:
        """Read the actual merged inputs from this transaction's UPSERT."""
        return self._returning_rows(statement, parameters, returning)

    def run_guarded_derived_update(
        self,
        statement: object,
        parameters: Mapping[str, object],
        *,
        returning: Sequence[str],
    ) -> list[dict[str, object]]:
        """Read final derived columns without leaving this cursor or transaction."""
        return self._returning_rows(statement, parameters, returning)
