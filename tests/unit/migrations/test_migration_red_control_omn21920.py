# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED controls remove reconciliation without depending on comment markers."""

import pytest

from tests.helpers.util_migration_shape import without_guarded_column_additions

pytestmark = pytest.mark.unit


def test_red_control_preserves_create_index_and_unguarded_alter() -> None:
    before = "CREATE TABLE IF NOT EXISTS public.events (id UUID);\n"
    after = (
        "\nCREATE INDEX events_correlation ON public.events (correlation_id);\n"
        "ALTER TABLE public.events ALTER COLUMN id SET NOT NULL;\n"
    )
    sql = (
        before
        + "ALTER TABLE public.events ADD COLUMN IF NOT EXISTS correlation_id UUID;"
        + after
    )
    stripped, count = without_guarded_column_additions(sql)
    assert count == 1
    assert stripped == before + after


def test_red_control_ignores_fake_additions_and_literal_semicolons() -> None:
    prefix = (
        "-- ALTER TABLE events ADD COLUMN IF NOT EXISTS fake TEXT;\n"
        "SELECT 'ALTER TABLE events ADD COLUMN IF NOT EXISTS fake TEXT;';\n"
        "DO $$ BEGIN RAISE NOTICE 'ALTER TABLE events ADD COLUMN IF NOT EXISTS fake TEXT;'; END $$;\n"
    )
    sql = (
        prefix
        + "ALTER TABLE public.events ADD COLUMN IF NOT EXISTS message TEXT DEFAULT 'x;y';\n"
        + 'ALTER TABLE public.events ADD COLUMN IF NOT EXISTS "state" VARCHAR(50);\n'
        + "CREATE INDEX events_state ON public.events (state);"
    )
    stripped, count = without_guarded_column_additions(sql)
    assert count == 2
    assert (
        stripped == prefix + "\n\nCREATE INDEX events_state ON public.events (state);"
    )


def test_red_control_reports_no_reconciliation() -> None:
    sql = "CREATE TABLE events (id UUID);"
    assert without_guarded_column_additions(sql) == (sql, 0)


def test_red_control_refuses_an_unterminated_addition() -> None:
    with pytest.raises(ValueError, match="no terminating semicolon"):
        without_guarded_column_additions(
            "ALTER TABLE events ADD COLUMN IF NOT EXISTS correlation_id UUID"
        )
