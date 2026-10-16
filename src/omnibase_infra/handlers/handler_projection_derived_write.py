# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""SQL plumbing for a derived write inside its caller-owned transaction.

The domain callback owns derivation. This module only validates its declared
columns, binds values, and checks that the input row still matches. It owns no
pricing, baseline resolution, money arithmetic, connections or publication.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from psycopg2 import sql

from omnibase_infra.protocols.protocol_projection_write_cursor import (
    ProtocolProjectionWriteCursor,
)

_COLUMN = re.compile(r"^[a-z_][a-z0-9_]*$")
_IDENTITY_COLUMNS = frozenset({"tenant_id", "writer_identity", "written_at"})


class HandlerProjectionDerivedWrite:
    """Execute the conditional derived update on a caller-owned transaction.

    This is an in-process storage handler: it receives an already attested
    cursor, applies no monetary policy, and never commits or publishes. The
    topology operation owns access checks and transaction lifetime.
    """

    @staticmethod
    def returning_columns(
        *,
        row: Mapping[str, object],
        conflict_columns: Sequence[str],
        input_columns: Sequence[str],
        derived_columns: Sequence[str],
        insert_only_columns: frozenset[str],
        expression_columns: Mapping[str, str],
        returning: Sequence[str],
    ) -> tuple[str, ...]:
        """Validate the capability before any SQL, and name its internal readback."""
        for columns, required in (
            (input_columns, True),
            (derived_columns, True),
            (returning, False),
        ):
            if (required and not columns) or len(set(columns)) != len(columns):
                raise ValueError(
                    "derived write requires non-empty, distinct column lists"
                )
            if any(not _COLUMN.fullmatch(column) for column in columns):
                raise ValueError("Invalid derived write column name")
        forbidden = _IDENTITY_COLUMNS.union(
            conflict_columns,
            input_columns,
            insert_only_columns,
            expression_columns,
            row,
        )
        if set(derived_columns) & forbidden:
            raise ValueError(
                "derived columns must not change identity or supplied inputs"
            )
        return tuple(dict.fromkeys((*conflict_columns, *input_columns, *returning)))

    @staticmethod
    def handle(
        cursor: ProtocolProjectionWriteCursor,
        *,
        schema: str,
        table: str,
        stored_rows: list[dict[str, object]],
        conflict_columns: Sequence[str],
        input_columns: Sequence[str],
        derived_columns: Sequence[str],
        returning: Sequence[str],
        derive: Callable[[Mapping[str, object]], Mapping[str, object]],
        adapt: Callable[[dict[str, object]], dict[str, object]],
    ) -> list[dict[str, object]]:
        """Update only derived columns if RETURNING inputs still match, or raise.

        The UPSERT has already locked the row, on this cursor's connection. The
        caller keeps that transaction open through this update, then commits before
        returning anything. A callback exception or unexpected update count rolls
        back the UPSERT too; no intermediate row is returned for publication.
        """
        if len(stored_rows) != 1:
            raise RuntimeError("derived write UPSERT did not return exactly one row")
        stored = stored_rows[0]
        callback_inputs = {
            column: stored[column]
            for column in dict.fromkeys((*conflict_columns, *input_columns))
        }
        derived = derive(MappingProxyType(callback_inputs))
        if not isinstance(derived, Mapping) or set(derived) != set(derived_columns):
            raise ValueError(
                "callback must return exactly the declared derived columns"
            )
        values = adapt(dict(derived))
        parameters = {
            f"derived_{index}": values[column]
            for index, column in enumerate(derived_columns)
        }
        assignments = sql.SQL(", ").join(
            sql.SQL("{} = {}").format(
                sql.Identifier(column), sql.Placeholder(f"derived_{index}")
            )
            for index, column in enumerate(derived_columns)
        )
        predicates = []
        for index, column in enumerate(
            dict.fromkeys((*conflict_columns, *input_columns))
        ):
            parameters[f"input_{index}"] = stored[column]
            predicates.append(
                sql.SQL("{} IS NOT DISTINCT FROM {}").format(
                    sql.Identifier(column), sql.Placeholder(f"input_{index}")
                )
            )
        readback = tuple(returning) or tuple(conflict_columns)
        columns = sql.SQL(", ").join(sql.Identifier(column) for column in readback)
        rows = cursor.run_guarded_derived_update(
            sql.SQL("UPDATE {}.{} SET {} WHERE {} RETURNING {}").format(
                sql.Identifier(schema),
                sql.Identifier(table),
                assignments,
                sql.SQL(" AND ").join(predicates),
                columns,
            ),
            parameters,
            returning=readback,
        )
        if len(rows) != 1:
            raise RuntimeError(
                "derived write input guard did not update exactly one row"
            )
        return rows if returning else []
