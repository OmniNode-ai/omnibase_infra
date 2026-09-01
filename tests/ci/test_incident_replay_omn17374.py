# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17374 incident replay: a declared grant that no migration issues.

OMN-15547 default-deny. ``check_topology_grant_delivery.py`` is newly wired
enforcement, so it must be replayed against the bytes of the real failure rather
than exempted by a baseline.

PROVENANCE OF THE ARTIFACT
--------------------------
``tests/fixtures/omn17374/0000_create_tenant_registry_mirror.61676f5c5.sql.captured``
is ``git cat-file``'d from ``61676f5c545fdefd7b241781ef6b964f9ff213fa`` -- the
commit that landed the file (omnibase_infra#3037, OMN-16930). Its sha256 is

    5a5f36011b2d3075b49c94f577b2019e6fbcf28e05137cc8a9941d1d9f5dd1ba

and the ``.201`` dev lane's ``platform_catalog.schema_migrations`` records

    node:node_projection_tenant_registry:0000_create_tenant_registry_mirror.sql
    5a5f36011b2d3075b49c94f577b2019e6fbcf28e05137cc8a9941d1d9f5dd1ba
    applied 2026-08-31 03:04:20.727769+00

read live 2026-09-01. So these are provably the exact bytes a real database
executed, not a reconstruction -- and executing them is what produced the
relation the runtime could not read.

WHAT THE ARTIFACT GETS WRONG
----------------------------
It CREATEs ``tenant_registry_mirror`` and grants SELECT to exactly two roles,
``role_omnidash`` and ``app_dashboard``. Both contracts that touch the relation
classify it ``omninode_internal``, so the runtime resolves its read AND its
write through the ``omninode_runtime_service`` binding -- principal
``omninode_runtime``, which this file grants nothing. The topology has declared
that grant since the same change; nothing ever issued it.

The consequence, live on the dev lane 2026-09-01 inside ``omninode-runtime`` on
its own ``OMNINODE_INTERNAL_DB_URL``::

    current_user: omninode_runtime
    select count(*) from tenant_registry_mirror
      -> InsufficientPrivilegeError: permission denied for table
         tenant_registry_mirror

which refused BOTH ``node_projection_delegation``'s write-time tenant lookup and
``node_projection_tenant_registry``'s own INSERT -- the second being why the
mirror sat at zero rows while its consumer group reported Stable at lag 0.

WHY THERE IS AN ACCEPT CONTROL
------------------------------
A guard hard-wired to report every relation as undelivered would replay this
incident perfectly and condemn the whole corpus. The accept control is the fix
shipping in this same commit: with ``0001`` present the same pair resolves as
delivered. It is read from the tree rather than captured by git-object because
no such object exists yet -- and the tree copy is what CI actually runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.validation.check_topology_grant_delivery import (
    GrantKey,
    declared_grants,
    delivered_grants,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

CAPTURED = REPO_ROOT / (
    "tests/fixtures/omn17374/0000_create_tenant_registry_mirror.61676f5c5.sql.captured"
)
CAPTURED_SHA256 = "5a5f36011b2d3075b49c94f577b2019e6fbcf28e05137cc8a9941d1d9f5dd1ba"

FIX = REPO_ROOT / (
    "docker/migrations/forward/nodes/node_projection_tenant_registry/"
    "0001_grant_omninode_runtime_tenant_registry_mirror.sql"
)

INCIDENT = GrantKey("omninode_runtime", "public", "tenant_registry_mirror")


@pytest.mark.unit
def test_captured_artifact_is_the_bytes_the_dev_lane_applied() -> None:
    """R1: the artifact is committed, unmodified, and independently pinned.

    The pin is not decorative. This same digest is the ``checksum`` column value
    the ``.201`` dev lane recorded when it applied the migration, so an edit to
    the fixture would silently sever the replay from the incident it claims.
    """
    assert CAPTURED.exists(), f"{CAPTURED} is missing"
    actual = hashlib.sha256(CAPTURED.read_bytes()).hexdigest()
    assert actual == CAPTURED_SHA256, (
        f"{CAPTURED.name} has been modified since capture: declared "
        f"{CAPTURED_SHA256}, actual {actual}"
    )


@pytest.mark.unit
def test_replay_rejects_the_real_registry_mirror_bytes(tmp_path: Path) -> None:
    """REJECT: the real corpus-of-one leaves the declared grant undelivered.

    The guard is driven over the verbatim captured bytes -- no hand-typed SQL,
    no synthetic relation name -- and the declaration side comes from the real
    checked-in topology, not a fixture. A world with this guard could not have
    merged #3037 without noticing; the world without it shipped a relation the
    runtime could neither read nor write and found out eight days later.
    """
    corpus = tmp_path / "forward" / "nodes" / "node_projection_tenant_registry"
    corpus.mkdir(parents=True)
    (corpus / "0000_create_tenant_registry_mirror.sql").write_bytes(
        CAPTURED.read_bytes()
    )

    declared = declared_grants(
        REPO_ROOT / "src/omnibase_infra/topology/instances/local.yaml"
    )
    delivered = delivered_grants(tmp_path / "forward")

    assert INCIDENT in declared, (
        "the topology must still declare the incident grant, or this replay "
        "passes for the wrong reason"
    )
    assert INCIDENT not in delivered, (
        "the captured 0000 bytes grant only role_omnidash and app_dashboard; "
        "reading them as delivering the omninode_runtime grant would make the "
        "guard blind to the exact failure it exists to catch"
    )
    # The artifact is not grant-free -- it issues two. The guard must
    # distinguish WHICH principal was granted, not merely notice a GRANT.
    assert GrantKey("role_omnidash", "public", "tenant_registry_mirror") in delivered
    assert GrantKey("app_dashboard", "public", "tenant_registry_mirror") in delivered


@pytest.mark.unit
def test_replay_accepts_the_corpus_once_the_fix_is_present(tmp_path: Path) -> None:
    """ACCEPT control: the guard is not hard-wired to report a violation."""
    corpus = tmp_path / "forward" / "nodes" / "node_projection_tenant_registry"
    corpus.mkdir(parents=True)
    (corpus / "0000_create_tenant_registry_mirror.sql").write_bytes(
        CAPTURED.read_bytes()
    )
    (corpus / "0001_grant_omninode_runtime_tenant_registry_mirror.sql").write_bytes(
        FIX.read_bytes()
    )

    delivered = delivered_grants(tmp_path / "forward")
    assert INCIDENT in delivered
