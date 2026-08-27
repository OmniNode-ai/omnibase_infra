# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The three-way checksum conflict that bricked the .201 dev lane (OMN-16705).

``_ledger/bootstrap.sql`` fails an entire ``forward-migration`` run when a
migration recorded in ``platform_catalog.schema_migrations`` no longer hashes to
its recorded ``content_sha256``::

    IF EXISTS (
      SELECT 1 FROM platform_catalog.schema_migrations ledger
      JOIN onex_application_migration_manifest manifest USING (...)
      WHERE ledger.checksum_kind = 'content_sha256'
        AND ledger.checksum <> manifest.checksum
    ) THEN
      RAISE EXCEPTION 'conflicting migration checksum in canonical node history';

There are therefore THREE things that must agree for a declared migration:

1. the bytes a live database recorded when it applied the file,
2. the bytes checked in at ``docker/migrations/forward/<artifact_path>``,
3. the checksum column of ``_ledger/application-migrations.tsv``.

The manifest validator already proves (2) == (3). Nothing proved (1). Two
in-place rewrites of already-applied files broke (1) and cost the dev lane every
subsequent migration run:

* ``0ca6735fa`` 2026-08-22 18:39 -> ``fdf0cc2c...`` routing 0001, APPLIED 20:09:08
* ``88f4ac346`` 2026-08-22 20:44 -> ``9505c67f...`` rewrote it 35 minutes later
* ``7de798a4a`` 2026-08-24 11:44 -> ``4cdaf9f2...`` rewrote it again (OMN-16450)
* ``559ee461a`` 2026-08-22 00:19 -> ``c1691130...`` credentials 0000, APPLIED 20:09:18
* ``7de798a4a`` 2026-08-24 11:44 -> ``d113ac80...`` rewrote it (+46 lines)

``_AS_APPLIED`` below is not a guess and not a re-derivation from git: it is a
live read of the lane's own ledger, taken read-only on 2026-08-27 with

.. code-block:: console

    ssh jonah@192.168.86.201 docker exec omnibase-infra-postgres \\
      psql -U postgres -d omnidash_analytics -At -F '|' -c \\
      "SELECT version, checksum_kind, checksum, applied_at
         FROM platform_catalog.schema_migrations
        WHERE version IN (...)"

The RED half of this proof is ``test_the_rewritten_bytes_really_did_conflict``:
it pins the rewritten hashes and asserts they differ from the applied ones, so a
future change that quietly re-adopts the rewritten bytes cannot make the GREEN
assertions vacuously true by moving both sides at once.

Ticket: OMN-16705
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_FORWARD = Path(__file__).resolve().parents[2] / "docker" / "migrations" / "forward"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"

_ROUTING = "nodes/node_delegation_routing_reducer/0001_create_delegation_routing_tenant_overlay.sql"
_CREDENTIALS = "nodes/node_projection_tenant_credentials/0000_create_tenant_inference_credentials.sql"

# Live readback of platform_catalog.schema_migrations on the .201 dev lane.
_AS_APPLIED: dict[str, str] = {
    _ROUTING: "fdf0cc2cf9f4c9fec9c9ffc96807258654aeac9fced869bb5e98f92959fa4873",
    _CREDENTIALS: "c1691130f33e3e7ca1cbf64572d58324d1a3c5d1e4156f60cb4f1b7a612ea68c",
}

# The bytes that produced the conflict. Never re-adopt these for these paths.
_REWRITTEN: dict[str, tuple[str, ...]] = {
    _ROUTING: (
        "9505c67f3947750945d8555870d8d2b1913cc97d9a1d0054d75a1c9a42411f35",  # 88f4ac346
        "4cdaf9f269e258fcd09d3c43afe3dcbcd5b5346e32686e32f92b8ce1a792993d",  # 7de798a4a
    ),
    _CREDENTIALS: (
        "d113ac80e173e79ad90563ddc1b09be85280c751191aa34eb0bc6be7a6f82ec5",  # 7de798a4a
    ),
}

# The additive successors that carry each rewrite's delta forward.
_SUCCESSORS = (
    "nodes/node_delegation_routing_reducer/0002_overlay_positive_bound_constraints.sql",
    "nodes/node_projection_tenant_credentials/0002_credential_identity_not_null.sql",
)


def _content_sha256(artifact_path: str) -> str:
    return hashlib.sha256((_FORWARD / artifact_path).read_bytes()).hexdigest()


def _manifest_checksums() -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in _MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        checksums[fields[0]] = fields[5]
    return checksums


def test_the_rewritten_bytes_really_did_conflict() -> None:
    """RED, and the anti-vacuity guard for every assertion below."""
    for artifact_path, rewritten in _REWRITTEN.items():
        applied = _AS_APPLIED[artifact_path]
        for rewritten_sha in rewritten:
            assert rewritten_sha != applied, (
                f"{artifact_path}: the pinned rewritten hash {rewritten_sha} now "
                "equals the applied hash, so this proof no longer reproduces the "
                "conflict it exists to prevent"
            )


@pytest.mark.parametrize("artifact_path", sorted(_AS_APPLIED))
def test_checked_in_bytes_match_what_the_lane_applied(artifact_path: str) -> None:
    """(1) == (2): the file hashes to the checksum the live ledger recorded."""
    actual = _content_sha256(artifact_path)
    assert actual == _AS_APPLIED[artifact_path], (
        f"{artifact_path} hashes to {actual}, but the .201 dev lane recorded "
        f"{_AS_APPLIED[artifact_path]}. bootstrap.sql will raise 'conflicting "
        "migration checksum in canonical node history' and exit the whole "
        "forward-migration run. Do not edit an applied migration -- add the next "
        "ordinal in the same node directory."
    )


@pytest.mark.parametrize("artifact_path", sorted(_AS_APPLIED))
def test_manifest_declares_what_the_lane_applied(artifact_path: str) -> None:
    """(1) == (3): the checked-in declaration agrees with the live ledger."""
    declared = _manifest_checksums().get(artifact_path)
    assert declared == _AS_APPLIED[artifact_path], (
        f"{artifact_path}: manifest declares {declared}, lane recorded "
        f"{_AS_APPLIED[artifact_path]}"
    )


@pytest.mark.parametrize("artifact_path", _SUCCESSORS)
def test_the_delta_is_carried_by_an_additive_successor(artifact_path: str) -> None:
    """Restoring the bytes is only half a fix; the delta must still land."""
    assert (_FORWARD / artifact_path).is_file(), artifact_path
    declared = _manifest_checksums().get(artifact_path)
    assert declared == _content_sha256(artifact_path), (
        f"{artifact_path} is not declared in {_MANIFEST.name} at its own checksum"
    )
