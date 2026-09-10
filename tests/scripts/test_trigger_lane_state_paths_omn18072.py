# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The rebuild trigger must fire on lane-STATE changes too (OMN-18072).

MEASURED: omnibase_infra#3352 (``0e9106da``) changed
``scripts/run-forward-migrations.sh`` -- the forward-migration runner every
compose up executes against the lane database -- and deleted files under
``docker/migrations/_blocked/``. The Runtime Rebuild Trigger ran on that merge
(run 34330947198) and reported "No rebuild trigger: no runtime_change label or
runtime path changes detected", so rule 24(a)'s automatic lab pass never fired
and the seam has still never executed on the dev lane.

The canonical deploy-gate classifier is right to miss it: its question is "does
this PR need deploy EVIDENCE", and it already covers ``docker/Dockerfile*`` and
``docker/docker-compose*.yml``. The rebuild trigger asks a different and wider
question -- "does this merge change what the lane RUNS" -- and a migration
runner, a migration corpus and the runtime policy env are all lane state a
rebuild applies. This module supplements the canonical list for the trigger
only; it does not widen the required deploy gate, which lives in another repo
and gates every PR in four.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "trigger_rebuild_on_merge.py"
)


def _load_module() -> object:
    spec = importlib.util.spec_from_file_location("_trigger_rebuild", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_trigger_rebuild"] = module
    spec.loader.exec_module(module)
    return module


# The exact file list of omnibase_infra#3352, the merge that produced no trigger.
PR_3352_FILES = [
    "docker/migrations/_blocked/chain_canary_reader_role.sql.blocked",
    "docker/migrations/_blocked/rollback_chain_canary_reader_role.sql.blocked",
    "scripts/run-forward-migrations.sh",
    "tests/scripts/test_login_only_role_grants_live_omn18060.py",
    "tests/unit/infra/test_login_only_role_grants_omn18060.py",
]

# Positive control for the zero: a docs-only merge must still yield nothing.
DOCS_ONLY_FILES = [
    "docs/runbooks/cold-lane-full-bringup.md",
    "README.md",
    "docs/architecture/ONEX_CANONICAL_ARCHITECTURE.md",
]


@pytest.mark.unit
class TestLaneStatePaths:
    def test_pr_3352_file_list_now_yields_a_trigger(self) -> None:
        """RED before OMN-18072: this exact list produced no rebuild."""
        module = _load_module()
        hits = module.find_lane_state_paths(PR_3352_FILES)  # type: ignore[attr-defined]
        assert "scripts/run-forward-migrations.sh" in hits
        assert any(path.startswith("docker/migrations/") for path in hits)
        assert module.should_trigger(hits, [])  # type: ignore[attr-defined]

    def test_docs_only_merge_still_yields_no_trigger(self) -> None:
        """Positive control for the zero above: an empty result must be earned."""
        module = _load_module()
        assert module.find_lane_state_paths(DOCS_ONLY_FILES) == []  # type: ignore[attr-defined]
        assert not module.should_trigger([], [])  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "path",
        [
            "scripts/run-forward-migrations.sh",
            "docker/migrations/0042_add_chain_canary_reader.sql",
            "docker/migrations/_blocked/chain_canary_reader_role.sql.blocked",
            "docker/docker-compose.infra.yml",
            "docker/docker-compose.dev-lane.yml",
            "docker/runtime-policy.env",
        ],
    )
    def test_each_declared_lane_state_path_matches(self, path: str) -> None:
        module = _load_module()
        assert module.find_lane_state_paths([path]) == [path]  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "path",
        [
            "docs/plans/2026-09-09-something.md",
            "scripts/generate_deep_dive.py",
            "tests/unit/test_thing.py",
            "docker/migrations-notes.txt",
            ".github/workflows/ci.yml",
        ],
    )
    def test_unrelated_paths_do_not_match(self, path: str) -> None:
        """The widening is a named list, not a directory opening."""
        module = _load_module()
        assert module.find_lane_state_paths([path]) == []  # type: ignore[attr-defined]

    def test_supplement_is_unioned_with_the_canonical_classifier(self) -> None:
        """The canonical result is never replaced or narrowed, only added to."""
        module = _load_module()

        def _canonical(files: list[str]) -> list[str]:
            return [f for f in files if f.startswith("src/omnibase_infra/runtime/")]

        files = ["src/omnibase_infra/runtime/kernel.py", "docker/runtime-policy.env"]
        hits = module.classify_runtime_paths(files, _canonical)  # type: ignore[attr-defined]
        assert "src/omnibase_infra/runtime/kernel.py" in hits
        assert "docker/runtime-policy.env" in hits

    def test_canonical_classifier_failure_still_fails_closed(self) -> None:
        """The supplement must not turn a broken canonical classifier into a pass."""
        module = _load_module()

        def _bad(files: list[str]) -> list[str]:
            return ["", "  "]

        with pytest.raises(ValueError, match="invalid path list"):
            module.classify_runtime_paths(["docker/runtime-policy.env"], _bad)  # type: ignore[attr-defined]
