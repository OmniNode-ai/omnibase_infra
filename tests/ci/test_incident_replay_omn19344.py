# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay for the migration-class guard (OMN-19344, rule RB).

THE ARTIFACT is migration 087's verbatim blob from a1ca993ca, the last commit
to touch the file, captured with ``git cat-file``. It is ``DROP TABLE IF EXISTS
public.delegation_events CASCADE``. The .201 dev lane's
``omnibase_infra.public.schema_migrations`` records
``docker/087_drop_stale_delegation_events_decoy.sql`` applied
2026-07-31 06:44:26.707462+00 (read read-only 2026-09-24), after the file's
last change on 2026-06-21, so these are the bytes a real database executed.

THE FALSE GREEN: every gate the tree had passed 087 as rollback-neutral. No
migration declared a class, ``check_migrations_applied`` accepts a lane AHEAD
of the tree by design (OMN-18388), and no down-script exists for 087. A
rollback selector reading PASS receipts would have treated a composition older
than 087 as eligible over a schema whose table it had dropped (the unified
plan's review round 3, finding 1). The guard's verdict on these bytes declared
expand-only must be REJECT.

THE DISCRIMINATOR: the same bytes declared forward-only are ACCEPTED, and a real
additive migration (020, one CREATE TABLE plus an index and comments) declared
expand-only is ACCEPTED. A guard hard-wired to refuse would replay the reject
case perfectly and condemn both controls.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "validation" / "check_migration_class.py"
_FIXTURE = (
    _REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn19344"
    / "087_drop_stale_delegation_events_decoy.a1ca993ca.sql.captured"
)
_FIXTURE_SHA = "f1c5da54c354dd249df058d0aac0432d6099bf46e3ce479a951aafb0b68c8015"
_KEY = "forward/087_drop_stale_delegation_events_decoy.sql"
_ADDITIVE = (
    _REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "020_create_agent_actions_table.sql"
)


def _guard() -> object:
    spec = importlib.util.spec_from_file_location(
        "check_migration_class_replay", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_captured_bytes_are_pinned() -> None:
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == _FIXTURE_SHA


def test_omn19344_replay_rejects_087_declared_expand_only() -> None:
    guard = _guard()
    text = _FIXTURE.read_text(encoding="utf-8")
    violations = guard.violations_for(_KEY, text, "expand-only")  # type: ignore[attr-defined]
    assert violations, "087 drops a table the older code reads; expand-only is a lie"
    assert any("DROP" in v for v in violations)


def test_omn19344_discriminator_accepts_087_forward_only_and_a_real_additive_file() -> (
    None
):
    guard = _guard()
    text = _FIXTURE.read_text(encoding="utf-8")
    assert guard.violations_for(_KEY, text, "forward-only") == []  # type: ignore[attr-defined]
    additive = _ADDITIVE.read_text(encoding="utf-8")
    assert (
        guard.violations_for(
            "forward/020_create_agent_actions_table.sql", additive, "expand-only"
        )  # type: ignore[attr-defined]
        == []
    )


@pytest.mark.parametrize("declared", ["forward-only", "contract", None])
def test_omn19344_087_is_a_rollback_barrier_without_a_recorded_down(
    declared: str | None, tmp_path: Path
) -> None:
    guard = _guard()
    root = tmp_path / "migrations"
    (root / "forward").mkdir(parents=True)
    (root / _KEY).write_bytes(_FIXTURE.read_bytes())
    barrier = guard.rb2_barrier(_KEY, declared, [], root)  # type: ignore[attr-defined]
    assert barrier is not None and _KEY in barrier
