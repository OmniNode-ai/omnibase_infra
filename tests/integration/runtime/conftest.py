# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Re-export the ephemeral Postgres fixture for tests under this directory.

``tests/integration/migrations/conftest.py`` owns the real fixture
(``EphemeralPostgres`` / ``ephemeral_postgres``); pytest only auto-discovers
fixtures from a file's own directory and its ancestors, not sibling
directories, so a live-database test under ``tests/integration/runtime/``
needs this thin re-export to see it.
"""

from __future__ import annotations

from tests.integration.migrations.conftest import (
    EphemeralPostgres,
    ephemeral_postgres,
)
