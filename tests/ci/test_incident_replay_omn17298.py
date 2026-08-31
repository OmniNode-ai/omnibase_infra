# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for the OMN-17298 RLS-policy atomicity guard (OMN-15547).

The guard is newly wired enforcement, so default-deny demands a real
regression case rather than a debt-baseline exemption.

THE REJECT ARTIFACT is the verbatim blob of
``node_canary_score_reducer/0003_capability_scores_tenant_id_to_uuid.sql`` as
it landed in 211e81e5c, captured with a ``git-object:`` locator anyone can
re-fetch. Its sha256 is ``d1eeefff...3282b2c`` -- byte-for-byte the checksum
the .201 dev lane recorded in ``platform_catalog.schema_migrations`` when it
applied that file on 2026-08-17 02:30:59.157734+00. The fixture is therefore
provably the exact bytes a real database ran, not a reconstruction of them.

THE ACCEPT CONTROL is migration 0033's verbatim blob from 3f10ee5e4, the
remedy OMN-17288 landed for the identical shape one node over. Without it a
guard hard-wired to report a violation would replay the incident perfectly
while condemning the fix, and 0033 is the strongest possible control because
its own header QUOTES the defective statements in prose -- a byte-blind
matcher passes the reject case and fails this one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "omn17298"
_SCRIPT = (
    _REPO_ROOT / "scripts" / "validation" / "check_migration_rls_policy_atomicity.py"
)

_REJECT = _FIXTURES / "0003_capability_scores_tenant_id_to_uuid.211e81e5c.sql.captured"
_REJECT_SHA = "d1eeefff11fbec255216154ba26aa6f0f29602a0b6fd8fd89e8de5d8d3282b2c"

_ACCEPT = (
    _FIXTURES
    / "0033_delegation_events_uuid_via_registry_single_transaction.3f10ee5e4.sql.captured"
)
_ACCEPT_SHA = "c49ca35f9ce64ed586b9b087533a4dbf69e46ec45791f9ff2e70da815cddd448"


def _guard() -> object:
    spec = importlib.util.spec_from_file_location(
        "check_migration_rls_policy_atomicity_replay", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_captured_bytes_are_the_bytes_the_dev_lane_applied() -> None:
    """Pin both fixtures. A reformatted artifact is a different artifact."""
    assert _sha256(_REJECT) == _REJECT_SHA
    assert _sha256(_ACCEPT) == _ACCEPT_SHA


def test_omn17298_replay_rejects_the_real_canary_0003_bytes() -> None:
    """The real guard, the real bytes, the verdict review missed twice.

    0003 drops ``tenant_isolation`` inside its ``DO $$`` block and recreates it
    after ``END$$``. The forward runner is ``psql -v ON_ERROR_STOP=1 -f <file>``
    with no ``--single-transaction``, so ``END$$`` COMMITS and the relation is
    briefly enforcing row-level security with zero policies -- every row denied
    to every non-owner, non-``BYPASSRLS`` principal, surfacing as SQLSTATE
    42501, which is indistinguishable at the call site from a missing GRANT.

    This shape shipped three times (0032, this file, and the opening hypothesis
    of OMN-17298) and passed every existing migration gate each time: the
    manifest validator only asks whether a declared checksum matches its file,
    and append-only only asks whether a declared file was edited. Neither one
    reads the SQL. This is the first check that does.
    """
    guard = _guard()
    found = guard.violations_for(_REJECT, _REJECT.read_text(encoding="utf-8"))  # type: ignore[attr-defined]

    assert found, "the real defective bytes must be rejected, not passed"
    assert any("RULE B" in violation for violation in found), found
    assert any("capability_scores" in violation for violation in found), found


def test_omn17298_replay_accepts_the_real_remedy_bytes() -> None:
    """The accept control: the fix must not be condemned by its own gate.

    0033's header quotes the three statements that made 0032 defective,
    verbatim, including a ``DROP POLICY``/``CREATE POLICY`` pair. A guard that
    matched raw bytes would reject this file -- the one file in the tree that
    is provably correct -- which is why comment scrubbing is load-bearing here
    rather than tidy.
    """
    guard = _guard()
    raw = _ACCEPT.read_text(encoding="utf-8")

    assert "--   DROP POLICY IF EXISTS tenant_isolation ON delegation_events;" in raw
    assert guard.violations_for(_ACCEPT, raw) == []  # type: ignore[attr-defined]
