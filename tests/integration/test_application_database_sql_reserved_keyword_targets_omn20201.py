# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The application-database SQL lint over the real SQL corpus (OMN-20201, I21).

The unit regressions drive single statements. This module runs the shipped lint
over every tracked .sql file in the repository and over the committed frozen
baseline, because a key word read as a relation target shows up as a baseline
row long before anyone reads the statement that produced it.

Nothing here connects to a database.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    _POSTGRES_RESERVED_KEYWORDS,
    lint_application_database_sql,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).parents[2]
_BASELINE = _REPO_ROOT / "scripts" / "ci" / "application_database_sql_baseline.yaml"
_TARGET = re.compile(r"application relation target '([^'.]+)' must be schema-qualified")


def _tracked_sql_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "*.sql"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    ).stdout.split()
    return [_REPO_ROOT / name for name in listed]


def test_no_tracked_sql_file_yields_a_reserved_key_word_target() -> None:
    files = _tracked_sql_files()
    assert len(files) > 100, "positive control: the corpus must be found"
    topology = load_topology_profile("local")
    offenders = []
    for path in files:
        for violation in lint_application_database_sql(
            path.read_text(encoding="utf-8"), topology
        ):
            match = _TARGET.fullmatch(violation)
            if match and match.group(1) in _POSTGRES_RESERVED_KEYWORDS:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {violation}")
    assert not offenders, "\n".join(offenders)


def test_frozen_baseline_holds_no_reserved_key_word_target_row() -> None:
    rows = yaml.safe_load(_BASELINE.read_text(encoding="utf-8"))["violations"]
    assert rows, "positive control: the baseline must load"
    targets = [
        match.group(1)
        for row in rows
        if (match := _TARGET.fullmatch(row["violation"])) is not None
    ]
    assert targets, "positive control: the baseline must hold target rows"
    assert not [t for t in targets if t in _POSTGRES_RESERVED_KEYWORDS]


def test_real_corpus_statements_that_produced_the_rows_stay_clean() -> None:
    topology = load_topology_profile("local")
    for relative in (
        "docker/application-acl-proof/seed.sql",
        "docker/migrations/rollback/rollback_096_grant_role_omnidash_omnidash_analytics.sql",
        "docker/migrations/forward/nodes/node_projection_live_events/0001_reclassify_event_lifecycle_types.sql",
    ):
        joined = "\n".join(
            lint_application_database_sql(
                (_REPO_ROOT / relative).read_text(encoding="utf-8"), topology
            )
        )
        for word in ("all", "true", "false", "then", "or"):
            assert (
                f"application relation target {word!r} must be schema-qualified"
                not in joined
            ), relative
