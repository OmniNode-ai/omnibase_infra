# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19322 (plan rows R1/R2, rule RB): the one-command rollback's selection.

The command reads PASS compose-dev receipts newest first as rollback candidates,
keeps the newest one rule RB makes eligible against the live migration ledger of
every database the lane binds, and redeploys it through the sanctioned path.
These tests pin the selection and every refusal, and prove that a refusal
redeploys nothing. The known-bad migration is the real ``087`` (``DROP TABLE``)
and the real ``031`` whose down-migration has a recorded lab execution.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "rollback_lane_to_known_good.py"
_MIGRATIONS = _REPO_ROOT / "docker" / "migrations"

_FWD_ONLY = "forward/087_drop_stale_delegation_events_decoy.sql"
_EXPAND = "forward/020_create_agent_actions_table.sql"
_LIFTED = "forward/031_create_llm_call_metrics_and_cost_aggregates.sql"
_BASE = frozenset({"forward/001_registration_projection.sql"})


@pytest.fixture(scope="module")
def tool() -> object:
    spec = importlib.util.spec_from_file_location(
        "rollback_lane_to_known_good", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cand(tool: object, sha: str, declared: frozenset[str]) -> object:
    return tool.ModelCandidate(  # type: ignore[attr-defined]
        sha=sha * 40 if len(sha) == 1 else sha,
        receipt_artifact_id=1,
        finished_at="2026-09-23T00:00:00Z",
        declared=declared,
    )


def _ledger(tool: object, applied: frozenset[str] | None, error: str = "") -> object:
    return tool.ModelLedgerRead(  # type: ignore[attr-defined]
        database="omnibase_infra", applied=applied, error=error
    )


def _select(
    tool: object, candidates: Sequence[object], ledgers: Sequence[object]
) -> object:
    return tool.select_candidate(  # type: ignore[attr-defined]
        candidates,
        ledgers,
        tool.load_classes(),  # type: ignore[attr-defined]
        tool.load_down_executions(),  # type: ignore[attr-defined]
        _MIGRATIONS,
    )


# ---- AC4 (RB-4): the executed pair ---------------------------------------


def test_forward_only_applied_since_candidate_refuses_it_by_name(tool: object) -> None:
    """Known-bad: 087 applied after C, no recorded down -> C refused, named."""
    c = _cand(tool, "a", _BASE)
    sel = _select(tool, [c], [_ledger(tool, _BASE | {_FWD_ONLY})])
    assert sel.verdict == "REFUSED"  # type: ignore[attr-defined]
    text = " ".join(sel.refusals)  # type: ignore[attr-defined]
    assert _FWD_ONLY in text and "omnibase_infra" in text and "a" * 12 in text
    assert sel.candidate is None  # type: ignore[attr-defined]


def test_only_expand_only_since_candidate_selects_it(tool: object) -> None:
    """Valid: only 020 (expand-only) applied after C -> C selected."""
    c = _cand(tool, "b", _BASE)
    sel = _select(tool, [c], [_ledger(tool, _BASE | {_EXPAND})])
    assert sel.verdict == "SELECTED"  # type: ignore[attr-defined]
    assert sel.candidate.sha == "b" * 40  # type: ignore[attr-defined]
    assert sel.requires_down == ()  # type: ignore[attr-defined]


def test_candidate_declaring_every_applied_head_is_eligible(tool: object) -> None:
    """RB-1(a): C declares 087 itself, so nothing was applied since C."""
    c = _cand(tool, "c", _BASE | {_FWD_ONLY})
    sel = _select(tool, [c], [_ledger(tool, _BASE | {_FWD_ONLY})])
    assert sel.verdict == "SELECTED"  # type: ignore[attr-defined]


def test_newest_eligible_candidate_wins_over_an_older_one(tool: object) -> None:
    newer = _cand(tool, "d", _BASE | {_FWD_ONLY})
    older = _cand(tool, "e", _BASE)
    sel = _select(tool, [newer, older], [_ledger(tool, _BASE | {_FWD_ONLY})])
    assert sel.candidate.sha == "d" * 40  # type: ignore[attr-defined]


def test_refused_newer_candidate_falls_through_to_an_eligible_older_one(
    tool: object,
) -> None:
    """Order is newest first; a refusal is recorded and the next one tried."""
    newer = _cand(tool, "f", _BASE)  # 087 applied since -> refused
    older = _cand(tool, "0", _BASE | {_FWD_ONLY})  # declares 087 -> eligible
    sel = _select(tool, [newer, older], [_ledger(tool, _BASE | {_FWD_ONLY})])
    assert sel.verdict == "SELECTED"  # type: ignore[attr-defined]
    assert sel.candidate.sha == "0" * 40  # type: ignore[attr-defined]
    assert any(_FWD_ONLY in r for r in sel.refusals)  # type: ignore[attr-defined]


def test_recorded_down_lifts_the_barrier_but_requires_the_down_first(
    tool: object,
) -> None:
    """RB-2's exception: 031 has a PASS lab execution recorded (OMN-19344)."""
    c = _cand(tool, "1", _BASE)
    sel = _select(tool, [c], [_ledger(tool, _BASE | {_LIFTED})])
    assert sel.verdict == "SELECTED"  # type: ignore[attr-defined]
    assert sel.requires_down == (_LIFTED,)  # type: ignore[attr-defined]


def test_undeclared_applied_migration_is_a_barrier(tool: object) -> None:
    c = _cand(tool, "2", _BASE)
    sel = _select(tool, [c], [_ledger(tool, _BASE | {"forward/999_not_in_tree.sql"})])
    assert sel.verdict == "REFUSED"  # type: ignore[attr-defined]
    assert "forward/999_not_in_tree.sql" in " ".join(sel.refusals)  # type: ignore[attr-defined]


# ---- AC2: every candidate refused is recorded, never silently picked ------


def test_all_candidates_refused_is_a_named_refusal(tool: object) -> None:
    cands = [_cand(tool, "3", _BASE), _cand(tool, "4", _BASE)]
    sel = _select(tool, cands, [_ledger(tool, _BASE | {_FWD_ONLY})])
    assert sel.verdict == "REFUSED" and sel.candidate is None  # type: ignore[attr-defined]
    assert len(sel.refusals) == 2  # type: ignore[attr-defined]
    assert all(_FWD_ONLY in r for r in sel.refusals)  # type: ignore[attr-defined]


# ---- AC5 (RB-3): fail closed ------------------------------------------------


def test_unreadable_ledger_is_indeterminate(tool: object) -> None:
    c = _cand(tool, "5", _BASE)
    sel = _select(tool, [c], [_ledger(tool, None, "psql exit 2: connection refused")])
    assert sel.verdict == "INDETERMINATE"  # type: ignore[attr-defined]
    assert "omnibase_infra" in sel.reason and "connection refused" in sel.reason  # type: ignore[attr-defined]
    assert sel.candidate is None  # type: ignore[attr-defined]


def test_no_pass_receipt_is_a_named_gap(tool: object) -> None:
    sel = _select(tool, [], [_ledger(tool, _BASE)])
    assert sel.verdict == "NO_CANDIDATE"  # type: ignore[attr-defined]
    assert "no PASS compose-dev receipt" in sel.reason  # type: ignore[attr-defined]


def test_no_ledger_read_at_all_is_indeterminate(tool: object) -> None:
    sel = _select(tool, [_cand(tool, "6", _BASE)], [])
    assert sel.verdict == "INDETERMINATE"  # type: ignore[attr-defined]


@pytest.mark.parametrize("verdict_case", ["refused", "indeterminate", "gap"])
def test_a_refusal_redeploys_nothing(tool: object, verdict_case: str) -> None:
    calls: list[str] = []
    if verdict_case == "refused":
        sel = _select(
            tool, [_cand(tool, "7", _BASE)], [_ledger(tool, _BASE | {_FWD_ONLY})]
        )
    elif verdict_case == "indeterminate":
        sel = _select(tool, [_cand(tool, "8", _BASE)], [_ledger(tool, None, "x")])
    else:
        sel = _select(tool, [], [_ledger(tool, _BASE)])
    rc = tool.act_on_selection(sel, execute=True, redeploy=calls.append)  # type: ignore[attr-defined]
    assert rc != 0 and calls == []


def test_selection_requiring_a_down_redeploys_nothing(tool: object) -> None:
    calls: list[str] = []
    sel = _select(tool, [_cand(tool, "9", _BASE)], [_ledger(tool, _BASE | {_LIFTED})])
    rc = tool.act_on_selection(sel, execute=True, redeploy=calls.append)  # type: ignore[attr-defined]
    assert rc != 0 and calls == []


def test_selected_candidate_is_redeployed_only_with_execute(tool: object) -> None:
    sel = _select(tool, [_cand(tool, "b", _BASE)], [_ledger(tool, _BASE | {_EXPAND})])
    calls: list[str] = []
    assert tool.act_on_selection(sel, execute=False, redeploy=calls.append) == 0  # type: ignore[attr-defined]
    assert calls == []
    assert tool.act_on_selection(sel, execute=True, redeploy=calls.append) == 0  # type: ignore[attr-defined]
    assert calls == ["b" * 40]


# ---- No path to production or a governed lane ------------------------------


@pytest.mark.parametrize(
    "lane", ["onex-prod", "prod", "stability-test", "judge", "lakshman", "staging"]
)
def test_only_compose_dev_is_accepted(tool: object, lane: str) -> None:
    with pytest.raises(SystemExit) as exc:
        tool.main(["--lane", lane, "--lane-host", "nowhere"])  # type: ignore[attr-defined]
    assert exc.value.code == 2


# ---- Ledger ids map to the migration keys the class manifest uses ----------


@pytest.mark.parametrize(
    ("database", "raw", "key"),
    [
        (
            "omnibase_infra",
            "docker/087_drop_stale_delegation_events_decoy.sql",
            _FWD_ONLY,
        ),
        (
            "omnidash_analytics",
            "node:node_canary_score_reducer:0003_capability_scores_tenant_id_to_uuid.sql",
            "forward/nodes/node_canary_score_reducer/0003_capability_scores_tenant_id_to_uuid.sql",
        ),
        (
            "omniintelligence",
            "026_create_code_entities",
            "intelligence/026_create_code_entities.sql",
        ),
    ],
)
def test_ledger_id_maps_to_manifest_key(
    tool: object, database: str, raw: str, key: str
) -> None:
    assert tool.ledger_id_to_key(database, raw) == key  # type: ignore[attr-defined]


def test_foreign_stream_id_maps_to_none(tool: object) -> None:
    # Cloud-stream rows in the application database belong to another image.
    assert (
        tool.ledger_id_to_key("omnidash_analytics", "20251207_tenants_uuid_pk.sql")
        is None
    )  # type: ignore[attr-defined]


def test_declared_keys_from_a_tree_listing(tool: object) -> None:
    listing = [
        "docker/migrations/forward/020_create_agent_actions_table.sql",
        "docker/migrations/forward/nodes/node_x/0001_x.sql",
        "docker/migrations/forward/_ledger/bootstrap.sql",
        "docker/migrations/rollback/rollback_031_x.sql",
        "docker/migrations/intelligence/000_extensions.sql",
        "docker/migrations/forward/000_create_multiple_databases.sh",
    ]
    assert tool.declared_keys_from_listing(listing) == frozenset(  # type: ignore[attr-defined]
        {
            "forward/020_create_agent_actions_table.sql",
            "forward/nodes/node_x/0001_x.sql",
            "intelligence/000_extensions.sql",
        }
    )


# ---- The candidate is the composition that RAN, not the receipt's key -------


def test_probed_revision_is_read_from_the_bound_probe(tool: object) -> None:
    # Shape of the real receipt artifact 10781058712, keyed 22c030ac6 but
    # proven on 842f4c122c2c (read 2026-09-24).
    receipt = {
        "sha": "22c030ac6db4db7c682f42bd4130294f8a5aaf41",
        "result": "PASS",
        "checks": [
            {
                "name": "deployed_revision",
                "ok": True,
                "evidence": "lane at 842f4c122c2c",
            },
            {
                "name": "probe_generation_bound",
                "ok": True,
                "evidence": "probe bound to the generation convergence read: "
                "omninode-runtime id=f9629bd2e668 image=sha256:e8313a7f2a57b84f4 "
                "revision=842f4c122c2c",
            },
        ],
    }
    assert tool.probed_revision(receipt) == "842f4c122c2c"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "checks",
    [
        [],
        [
            {
                "name": "probe_generation_bound",
                "ok": False,
                "evidence": "revision=842f4c122c2c",
            }
        ],
        [
            {
                "name": "probe_generation_bound",
                "ok": True,
                "evidence": "no revision read",
            }
        ],
    ],
)
def test_receipt_without_a_bound_probe_revision_is_no_candidate(
    tool: object, checks: list[dict[str, object]]
) -> None:
    assert tool.probed_revision({"result": "PASS", "checks": checks}) == ""  # type: ignore[attr-defined]


# ---- The redeploy path: the candidate merge's own rebuild-trigger run --------
#
# ``--execute`` publishes nothing itself. It reruns the Runtime Rebuild Trigger
# run of the merge whose commit IS the candidate, so the redeploy-start command
# is published by the same CI job, identity and bus credentials a dev merge
# uses, pinned to that merge sha. It needs no local bus credential, which is
# what lets someone who did not write the change run it from the invocation.

_SHA = "c" * 40
_HEAD = "d" * 40
_MERGED = "2026-09-23T22:38:46Z"


def _pr(**over: object) -> dict[str, object]:
    pr: dict[str, object] = {
        "number": 4016,
        "merge_commit_sha": _SHA,
        "merged_at": _MERGED,
        "head": {"sha": _HEAD},
        "base": {"ref": "dev"},
    }
    pr.update(over)
    return pr


def _run_rec(
    run_id: int, *, verify: str = "success", **over: object
) -> dict[str, object]:
    run: dict[str, object] = {
        "id": run_id,
        "event": "pull_request",
        "head_sha": _HEAD,
        "created_at": "2026-09-23T22:38:49Z",
        "status": "completed",
        "jobs": [
            {"name": "Trigger node_redeploy Start", "conclusion": "success"},
            {"name": "Verify dev lane applied the redeploy", "conclusion": verify},
        ],
    }
    run.update(over)
    return run


def test_trigger_run_of_the_candidate_merge_is_selected(tool: object) -> None:
    run_id, reason = tool.select_trigger_run(_SHA, [_pr()], [_run_rec(7)])  # type: ignore[attr-defined]
    assert run_id == 7, reason


def test_no_merged_dev_pr_for_the_candidate_is_refused_by_name(tool: object) -> None:
    run_id, reason = tool.select_trigger_run(  # type: ignore[attr-defined]
        _SHA, [_pr(merge_commit_sha="e" * 40)], [_run_rec(7)]
    )
    assert run_id is None and _SHA[:12] in reason


def test_a_main_merge_is_not_a_dev_lane_redeploy(tool: object) -> None:
    run_id, _ = tool.select_trigger_run(  # type: ignore[attr-defined]
        _SHA, [_pr(base={"ref": "main"})], [_run_rec(7)]
    )
    assert run_id is None


def test_a_trigger_run_that_published_nothing_is_refused(tool: object) -> None:
    run_id, reason = tool.select_trigger_run(  # type: ignore[attr-defined]
        _SHA, [_pr()], [_run_rec(7, verify="skipped")]
    )
    assert run_id is None and "published" in reason


def test_a_run_before_the_merge_or_of_another_head_is_not_the_merge_run(
    tool: object,
) -> None:
    early = _run_rec(7, created_at="2026-09-23T22:00:00Z")
    other = _run_rec(8, head_sha="f" * 40)
    run_id, _ = tool.select_trigger_run(_SHA, [_pr()], [early, other])  # type: ignore[attr-defined]
    assert run_id is None


def test_newest_publishing_run_wins(tool: object) -> None:
    older = _run_rec(7)
    newer = _run_rec(9, created_at="2026-09-23T23:10:00Z")
    run_id, _ = tool.select_trigger_run(_SHA, [_pr()], [older, newer])  # type: ignore[attr-defined]
    assert run_id == 9


def test_an_in_flight_trigger_run_blocks_the_redeploy(tool: object) -> None:
    runs = [
        {"id": 1, "status": "completed"},
        {"id": 2, "status": "in_progress"},
        {"id": 3, "status": "queued"},
    ]
    assert tool.inflight_trigger_runs(runs) == [2, 3]  # type: ignore[attr-defined]
    assert tool.inflight_trigger_runs([{"id": 1, "status": "completed"}]) == []  # type: ignore[attr-defined]
