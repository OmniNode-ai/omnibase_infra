# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pre-PR verify slot runs the intelligence migration, into its own database.

Measured on the fourth slot boot (2026-09-24, omnibase_infra#3944 at
116e920b7): the slot overlay fenced ``intelligence-migration`` out entirely, so
nothing ever migrated ``omniintelligence_prepr1`` and the slot's projection-api
and runtime-worker failed stamping on a missing ``public.db_metadata`` there
(OMN-19404).

The runner is executed under ``sh`` against a stub ``psql`` that records every
database it was pointed at, so these pin what the script DOES rather than what
it says.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-intelligence-migrations.sh"
INTEL_CORPUS = REPO_ROOT / "docker" / "migrations" / "intelligence"
OVERLAY = REPO_ROOT / "docker" / "docker-compose.prepr.yml"
ENTRYPOINT = REPO_ROOT / "scripts" / "runtime_build" / "prepr_verify_lane.sh"

pytestmark = pytest.mark.unit

STUB_PSQL = """#!/bin/sh
# Records the -d target of every call, answers the two existence probes.
db=""
prev=""
for a in "$@"; do
  [ "$prev" = "-d" ] && db="$a"
  prev="$a"
done
printf '%s\\n' "$db" >> "$STUB_LOG"
case "$*" in
  *"FROM pg_database WHERE datname"*) [ "$STUB_DB_EXISTS" = "1" ] && echo 1 ;;
  *"CREATE DATABASE"*) echo "CREATE DATABASE $db" >> "$STUB_LOG" ;;
esac
exit 0
"""


def _run(
    tmp_path: Path, *, slot: str | None, db_exists: bool
) -> tuple[int, list[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "psql"
    stub.write_text(STUB_PSQL, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "psql.log"
    log.write_text("", encoding="utf-8")
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "POSTGRES_PASSWORD": "stub",
        "MIGRATIONS_DIR": str(INTEL_CORPUS),
        "PG_WAIT_RETRIES": "1",
        "STUB_LOG": str(log),
        "STUB_DB_EXISTS": "1" if db_exists else "0",
    }
    if slot is not None:
        env["ONEX_DB_SLOT"] = slot
    proc = subprocess.run(
        ["sh", str(RUNNER)], capture_output=True, text=True, check=False, env=env
    )
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln]
    return proc.returncode, lines, proc.stdout + proc.stderr


def test_outside_a_slot_the_runner_migrates_omniintelligence(tmp_path: Path) -> None:
    """Positive control: the stub sees the real targets, so the slot pin can fail."""
    code, targets, out = _run(tmp_path, slot=None, db_exists=True)
    assert code == 0, out
    assert "omniintelligence" in targets
    assert set(targets) <= {"postgres", "omniintelligence"}


def test_under_a_slot_every_statement_targets_the_slot_database(tmp_path: Path) -> None:
    code, targets, out = _run(tmp_path, slot="prepr1", db_exists=True)
    assert code == 0, out
    assert "omniintelligence_prepr1" in targets
    assert set(targets) <= {"postgres", "omniintelligence_prepr1"}, targets


def test_under_a_slot_an_absent_database_is_refused_not_created(tmp_path: Path) -> None:
    code, targets, out = _run(tmp_path, slot="prepr1", db_exists=False)
    assert code == 4, out
    assert "slot_fence_refusal" in out
    assert not any(t.startswith("CREATE DATABASE") for t in targets), targets


def test_a_malformed_slot_token_is_refused(tmp_path: Path) -> None:
    code, targets, out = _run(tmp_path, slot="Prepr_1", db_exists=True)
    assert code == 3, out
    assert targets == []


class _TagTolerantLoader(yaml.SafeLoader):
    """Load the overlay without resolving compose's ``!override`` tags."""


def _passthrough(loader: Any, _suffix: str, node: Any) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_TagTolerantLoader.add_multi_constructor("!", _passthrough)
_TagTolerantLoader.add_multi_constructor("tag:yaml.org,2002:merge", _passthrough)


def test_the_overlay_runs_the_intelligence_migration_as_a_slot_one_shot() -> None:
    # Only pass-through constructors for compose's own tags are added to a
    # SafeLoader, so nothing arbitrary can be instantiated.
    overlay = yaml.load(OVERLAY.read_text(encoding="utf-8"), Loader=_TagTolerantLoader)  # noqa: S506
    svc = overlay["services"]["intelligence-migration"]
    assert svc["profiles"] == ["prepr-migrate"]
    assert "${ONEX_PREPR_SLOT" in svc["container_name"]
    assert svc["depends_on"] == []
    merged: dict[str, Any] = {}
    for block in svc["environment"].get("<<", []):
        merged.update(block)
    merged.update({k: v for k, v in svc["environment"].items() if k != "<<"})
    assert "ONEX_DB_SLOT must be set" in merged["ONEX_DB_SLOT"]


def test_the_entrypoint_runs_it_after_the_forward_migration_and_before_up() -> None:
    code = [
        ln.strip()
        for ln in ENTRYPOINT.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#")
    ]

    def index(fragment: str) -> int:
        hits = [i for i, ln in enumerate(code) if fragment in ln]
        assert len(hits) == 1, (fragment, hits)
        return hits[0]

    forward = index("run --rm --no-deps forward-migration")
    intel = index("run --rm --no-deps intelligence-migration")
    up = index("up -d --no-deps --no-build")
    assert forward < intel < up
