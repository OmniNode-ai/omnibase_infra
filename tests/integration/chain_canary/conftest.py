# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for the OMN-18872 chain-canary projection integration proofs.

``test_application_migration_ledger_omn15413.py`` owns the PostgreSQL 16
harness. It is re-exported here rather than imported into the test module for
the reason that file's own siblings record: imported into a module, the name
shadows every ``pg16`` test parameter in it and ruff rejects the redefinition.
"""

from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    pg16,
)

__all__ = ["pg16"]
