# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed controls for a derived write, using the real adapter SQL seam.

The new entry point must preserve declared access and tenant authority, refuse
identifier injection and derived writes to identity or input columns, and roll
back a callback that attempts to write outside its declared derived columns.
Real transaction visibility and rollback live in the integration sibling.
"""

import pytest

from tests.unit.runtime.auto_wiring.test_projection_attested_write_omn18159 import (
    _CONNECTIONS,
    TABLE,
    _adapter,
    _patched_driver,
    _RecordingCursor,
)

__all__ = ["_patched_driver"]


def _request(adapter, **changes):
    arguments = {
        "derive": lambda row: {"summary": f"derived:{row['payload']}"},
        "input_columns": ("payload",),
        "derived_columns": ("summary",),
        "returning": ("correlation_id", "payload", "summary"),
        **changes,
    }
    return adapter.upsert_returning(
        TABLE,
        "correlation_id",
        {"correlation_id": "run", "payload": "input"},
        **arguments,
    )


def test_a_read_only_declaration_cannot_derive_a_write() -> None:
    adapter, _ = _adapter(_RecordingCursor(), access="read")
    with pytest.raises(PermissionError, match="write refused"):
        _request(adapter)


@pytest.mark.parametrize(
    ("input_columns", "derived_columns"),
    [(("payload",), ()), ((), ("summary",)), (("payload",), ("summary",))],
)
def test_declared_derivation_cannot_silently_omit_its_callback(
    input_columns: tuple[str, ...], derived_columns: tuple[str, ...]
) -> None:
    cursor = _RecordingCursor()
    adapter, _ = _adapter(cursor)
    with pytest.raises(ValueError, match="derive"):
        _request(
            adapter,
            derive=None,
            input_columns=input_columns,
            derived_columns=derived_columns,
        )
    assert not any(sql.lstrip().startswith("INSERT") for sql, _ in cursor.executed)


def test_a_caller_cannot_override_the_recorded_tenant() -> None:
    adapter, _ = _adapter(_RecordingCursor())
    with pytest.raises(ValueError, match="tenant"):
        _request(adapter, tenant="00000000-0000-4000-8000-000000021015")


@pytest.mark.parametrize("column", ["correlation_id", "tenant_id", "payload"])
def test_identity_and_input_columns_cannot_be_derived(column: str) -> None:
    cursor = _RecordingCursor()
    adapter, _ = _adapter(cursor)
    with pytest.raises(ValueError, match="derived"):
        _request(adapter, derived_columns=(column,))
    assert not any(sql.lstrip().startswith("INSERT") for sql, _ in cursor.executed)


def test_input_identifier_injection_is_refused_before_the_upsert() -> None:
    cursor = _RecordingCursor()
    adapter, _ = _adapter(cursor)
    with pytest.raises(ValueError, match="column"):
        _request(adapter, input_columns=("payload); DROP TABLE generation_events;--",))
    assert not any(sql.lstrip().startswith("INSERT") for sql, _ in cursor.executed)


def test_a_callback_cannot_write_an_undeclared_derived_column() -> None:
    cursor = _RecordingCursor(
        [{"correlation_id": "run", "payload": "stored", "summary": "old"}]
    )
    adapter, _ = _adapter(cursor)
    with pytest.raises(ValueError, match="derived"):
        _request(adapter, derive=lambda _: {"summary": "new", "tenant_id": "forged"})
    _CONNECTIONS[-1].rollback.assert_called_once()
    _CONNECTIONS[-1].commit.assert_not_called()
