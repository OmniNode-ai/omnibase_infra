# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The topology's declared TABLE grants must be delivered by a migration.

OMN-17374. See ``scripts/validation/check_topology_grant_delivery.py`` for the
defect class and why the bound is a ratchet rather than a clean zero.

The incident this suite replays is the OMN-17374 half of it, and it is replayed
against the real checked-in topology and the real checked-in migration corpus --
no fixture topology, no synthetic corpus. A gate proven only against inputs that
cannot exhibit the failure is the OMN-15547 defect.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validation.check_topology_grant_delivery import (
    MAX_UNDELIVERED,
    GrantKey,
    declared_grants,
    delivered_grants,
    undelivered,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The relation whose absent grant refused BOTH the delegation writer's identity
# lookup and node_projection_tenant_registry's own INSERT on the .201 dev lane,
# 2026-09-01. Named explicitly because this pair is the incident, and a ratchet
# that merely counts would stay green if this exact grant were reverted while an
# unrelated one landed.
OMN_17374_INCIDENT = GrantKey("omninode_runtime", "public", "tenant_registry_mirror")


@pytest.mark.unit
def test_incident_grant_is_delivered_by_a_migration() -> None:
    """OMN-17374 replay: the registry-mirror grant reaches a real database.

    RED before ``0001_grant_omninode_runtime_tenant_registry_mirror.sql``:
    this pair was declared in all three topology instances and issued by nothing
    in the corpus, so the runtime -- which resolves this relation through the
    ``omninode_runtime_service`` binding because both contracts classify it
    ``omninode_internal`` -- got ``permission denied for table
    tenant_registry_mirror`` on every read and every write.
    """
    delivered = delivered_grants(REPO_ROOT / "docker/migrations/forward")
    assert OMN_17374_INCIDENT in delivered, (
        f"{OMN_17374_INCIDENT} is declared by the topology and issued by no "
        "migration. That is the OMN-17374 outage exactly: the delegation "
        "writer's tenant lookup and the registry projection's own INSERT both "
        "ride this grant."
    )


@pytest.mark.unit
def test_incident_grant_is_declared_by_the_topology() -> None:
    """The gate's subject is the generated topology, not a hand list.

    If the declaration ever disappears, this suite must fail rather than pass
    vacuously -- a delivered grant nobody declares is a different defect, and a
    green run for the reason "we stopped asking" is the worst outcome available.
    """
    declared = declared_grants(
        REPO_ROOT / "src/omnibase_infra/topology/instances/local.yaml"
    )
    assert OMN_17374_INCIDENT in declared


@pytest.mark.unit
def test_undelivered_count_is_exactly_the_ratchet_bound() -> None:
    """The bound bites in both directions.

    Above it: a new topology grant landed with no migration to issue it, which
    is the next OMN-16993/OMN-17374 waiting for its relation to take traffic.

    Below it: somebody closed part of the residual and left the bound stale, so
    the gate would silently tolerate a regression back up to the old number. The
    bound moves in the same change that closes the grant, or not at all.
    """
    missing = undelivered(REPO_ROOT)
    rendered = "\n".join(f"  {key}" for key in missing)
    assert len(missing) == MAX_UNDELIVERED, (
        f"{len(missing)} undelivered topology TABLE grants, bound is "
        f"{MAX_UNDELIVERED}. Update MAX_UNDELIVERED in the same change that "
        f"moves the count.\n{rendered}"
    )


@pytest.mark.unit
def test_checker_cli_exits_zero_on_the_checked_in_tree() -> None:
    """The wired entry point is the one CI and pre-commit actually run."""
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/validation/check_topology_grant_delivery.py"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.unit
def test_checker_fails_when_the_residual_grows() -> None:
    """A tightened bound must actually reject the current tree.

    This is the RED control: it proves the gate can fail, using the real corpus
    and the real topology, without editing either.
    """
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/validation/check_topology_grant_delivery.py"),
            "--max-undelivered",
            str(MAX_UNDELIVERED - 1),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "UNDELIVERED" in completed.stdout


@pytest.mark.unit
def test_grant_on_all_tables_is_not_read_as_a_named_delivery() -> None:
    """``GRANT ... ON ALL TABLES IN SCHEMA`` delivers no NAMED relation.

    The bootstrap issues blanket grants for ``role_omnidash``. Reading one as
    satisfying a per-relation declaration would make the gate green for exactly
    the principal whose authorization is NOT blanket -- ``omninode_runtime``,
    which the bootstrap's own LOGIN_ONLY_ROLE_MAP note deliberately excludes.
    """
    delivered = delivered_grants(REPO_ROOT / "docker/migrations/forward")
    for key in delivered:
        assert key.table.upper() != "ALL"
