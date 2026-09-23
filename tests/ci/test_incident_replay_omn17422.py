# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for the OMN-17422 AC5 ratchet, over bytes a database ran.

OMN-15547: a guard that has never been replayed against the real incident it
exists to catch is decorative. This replays OMN-18774 -- ``generation_events``
carrying a ``tenant_isolation`` policy predicated on the ``app.tenant_id``
session GUC while its owning contract declares it ``omninode_internal``, so no
declared writer could ever satisfy it.

Both artifacts are the bytes onex-dev actually executed, not a reconstruction.
``public.node_schema_migrations`` on that lane, read read-only 2026-09-22,
records:

  node:node_projection_delegation:0027_generation_events_tenant_rls.sql
    applied 2026-07-28T21:15:06Z  checksum 00284bf6...5962
  node:node_projection_delegation:0043_generation_events_drop_tenant_posture.sql
    applied 2026-09-19T02:19:48Z  checksum b119079f...f05b

Each fixture is sha256-pinned below to exactly those values. The 0027 artifact
is captured with ``git cat-file`` from 6194dc4d9, the commit that landed it --
NOT from the current tree, which hashes ``9f5ffade...`` because the file was
amended after it ran. Pinning the tree copy would have replayed bytes no
database ever executed.

The accept control is not optional. A guard hard-wired to report a violation
would replay the incident perfectly and condemn its own remedy, so the same
module drives 0043 -- the real fix, byte-identical to what the lane applied --
through the identical path and requires a clean verdict.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "validation" / "check_tenant_guc_domain_parity.py"
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "omn17422"

_DEFECT = _FIXTURES / "0027_generation_events_tenant_rls.6194dc4d9.sql.captured"
_REMEDY = (
    _FIXTURES / "0043_generation_events_drop_tenant_posture.4139b1fad.sql.captured"
)

# The checksums onex-dev's own ledger recorded for these two migrations.
_DEFECT_SHA = "00284bf64e3b2a8935bf1737d78dea64c6801966d61edf2b0e6ff19c19d85962"
_REMEDY_SHA = "b119079fd139d1a271a0ad50001b1c120f115db98c8cbc401ff4f0251a0ff05b"


def _load() -> object:
    spec = importlib.util.spec_from_file_location(
        "check_tenant_guc_domain_parity", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> object:
    return _load()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root(tmp_path: Path, *files: Path) -> Path:
    """A throwaway root carrying the captured bytes in their own lineage."""
    lineage = (
        tmp_path
        / "docker"
        / "migrations"
        / "forward"
        / "nodes"
        / "node_projection_delegation"
    )
    lineage.mkdir(parents=True)
    for source in files:
        # Strip the capture suffix so the ordinal parses as the real migration.
        name = source.name.split(".")[0] + ".sql"
        (lineage / name).write_bytes(source.read_bytes())

    instances = tmp_path / "src" / "omnibase_infra" / "topology" / "instances"
    instances.mkdir(parents=True)
    (instances / "local.yaml").write_text(
        "databases:\n"
        "  application:\n"
        "    principals:\n"
        "      omninode_runtime:\n"
        "        grants:\n"
        "          - object_type: TABLE\n"
        "            schema: public\n"
        "            objects:\n"
        "              - generation_events\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "src" / "omnibase_infra" / "topology"
    # generation_events reads `schema: public` in the real topology; the
    # INTERNAL bridge is the only checked-in thing that classifies it.
    (mapping / "physical_schema_mapping.py").write_text(
        "TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359: frozenset[str] = frozenset(\n"
        "    []\n)\n"
        "INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359: frozenset[str] = frozenset(\n"
        "    ['generation_events']\n)\n",
        encoding="utf-8",
    )
    return tmp_path


def test_the_artifacts_are_the_bytes_the_lane_applied() -> None:
    """Pin first. A replay over bytes nothing executed proves nothing."""
    assert _sha256(_DEFECT) == _DEFECT_SHA
    assert _sha256(_REMEDY) == _REMEDY_SHA


def test_replay_rejects_the_real_0027_bytes(gate: object, tmp_path: Path) -> None:
    """The incident: RLS on, GUC-predicated policy, internal-declared relation."""
    root = _root(tmp_path, _DEFECT)
    assert gate.main(["--root", str(root)]) == 1  # type: ignore[attr-defined]
    found, _ = gate.violations(root)  # type: ignore[attr-defined]
    assert {v.relation for v in found} == {"generation_events"}
    assert all(v.domain == "omninode_internal" for v in found)


def test_replay_accepts_the_real_0043_remedy(gate: object, tmp_path: Path) -> None:
    """The accept control, without which the reject case proves nothing.

    0043 is the migration OMN-18774 landed: DROP POLICY, DISABLE ROW LEVEL
    SECURITY, DROP COLUMN tenant_id. Replayed after 0027 in the same lineage,
    the net end state must be clean -- which is also the append-only property,
    since 0027's CREATE POLICY is still present in the corpus verbatim.
    """
    root = _root(tmp_path, _DEFECT, _REMEDY)
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]
    found, _ = gate.violations(root)  # type: ignore[attr-defined]
    assert found == []


def test_the_remedy_alone_is_not_read_as_a_defect(gate: object, tmp_path: Path) -> None:
    """A byte-blind matcher would trip on 0043's own prose, which quotes the
    shape it removes."""
    root = _root(tmp_path, _REMEDY)
    assert gate.main(["--root", str(root)]) == 0  # type: ignore[attr-defined]
