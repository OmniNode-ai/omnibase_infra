# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the hosted activation grant probe against a disposable database."""

from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.ephemeral_pg]


def test_activation_grant_supports_the_runtime_upsert_and_read(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    proof = Path(__file__).parent / "proofs/compaction_activations_omn21115.sql"
    probe = "SELECT to_regclass('omninode_internal.compaction_activations')"
    # Start without the table, so the after-check cannot pass on a leftover.
    before = ephemeral_postgres.psql("-At", "-c", probe)
    assert before.returncode == 0, before.stderr
    assert not before.stdout.strip()
    result = ephemeral_postgres.psql("-X", "-v", "ON_ERROR_STOP=1", "-f", str(proof))
    assert result.returncode == 0, result.stderr
    # The probe must leave the service pristine even when it succeeds.
    after = ephemeral_postgres.psql("-At", "-c", probe)
    assert after.returncode == 0, after.stderr
    assert not after.stdout.strip()
