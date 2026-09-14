# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/audit_orphan_required_contexts.py (OMN-18346).

Covers the four properties the audit is only useful if it has: it reads the
PR-time evidence source at all, it mirrors GitHub's leaf matching rather than
raw string equality, it still names a genuinely orphaned context, and it refuses
to judge on a zero-row evidence read.

The incident replay over real captured bytes lives in
tests/ci/test_incident_replay_omn18346.py. These tests are the synthetic
complement: they exercise shapes (paginated, `checks[]`-form protection, empty
evidence) that the single live capture does not happen to contain.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "audit_orphan_required_contexts.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_orphan_required_contexts", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    return _load_module()


def _protection(
    contexts: list[str] | None = None, checks: list[str] | None = None
) -> str:
    rsc: dict[str, Any] = {"strict": False}
    if contexts is not None:
        rsc["contexts"] = contexts
    if checks is not None:
        rsc["checks"] = [{"context": c, "app_id": 1} for c in checks]
    return json.dumps(
        {"required_status_checks": rsc, "enforce_admins": {"enabled": True}}
    )


def _check_runs(names: list[str]) -> str:
    return json.dumps(
        {"total_count": len(names), "check_runs": [{"name": n} for n in names]}
    )


def _gh(
    *,
    protection: str,
    push_shas: list[str],
    pr_shas: list[str],
    runs_by_sha: dict[str, list[str]],
    protection_rc: int = 0,
) -> Any:
    empty = json.dumps({"total_count": 0, "check_runs": []})

    def gh(argv: list[str]) -> tuple[int, str]:
        path = argv[1]
        if path.endswith("/branches/main/protection"):
            return protection_rc, protection
        if "/commits?per_page=" in path:
            return 0, json.dumps([{"sha": s} for s in push_shas])
        if "/pulls?" in path:
            return 0, json.dumps(
                [
                    {"merged_at": "2026-09-13T00:00:00Z", "head": {"sha": s}}
                    for s in pr_shas
                ]
            )
        if "/check-runs" in path:
            for sha, names in runs_by_sha.items():
                if f"/commits/{sha}/check-runs" in path:
                    return (
                        (0, _check_runs(names))
                        if path.endswith("&page=1")
                        else (0, empty)
                    )
            return 0, empty
        return 1, ""

    return gh


# ---------------------------------------------------------------------------
# The defect: PR-time contexts absent from push check-runs must still PASS.
# ---------------------------------------------------------------------------
def test_contexts_seen_only_on_pr_head_shas_are_not_orphans(audit: ModuleType) -> None:
    result = audit.audit_repo_main(
        "OmniNode-ai",
        "somerepo",
        _gh(
            protection=_protection(["CI Summary", "Tests Gate", "verify / verify"]),
            push_shas=["a" * 40],
            pr_shas=["b" * 40],
            # The push commit carries only the push-triggered job. Every required
            # context reports on the PR head SHA and nowhere else.
            runs_by_sha={
                "a" * 40: ["Runtime Rebuild Trigger"],
                "b" * 40: ["CI Summary", "Tests Gate", "verify"],
            },
        ),
    )
    assert result["status"] == "ok"
    assert result["orphan_contexts"] == []


def test_push_only_evidence_is_still_counted(audit: ModuleType) -> None:
    """A context that only ever reports on push is observed, not orphaned."""
    result = audit.audit_repo_main(
        "OmniNode-ai",
        "somerepo",
        _gh(
            protection=_protection(["Runtime Rebuild Trigger"]),
            push_shas=["a" * 40],
            pr_shas=["b" * 40],
            runs_by_sha={
                "a" * 40: ["Runtime Rebuild Trigger"],
                "b" * 40: ["CI Summary"],
            },
        ),
    )
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# The falsifier: a genuinely orphaned context must still be reported by name.
# ---------------------------------------------------------------------------
def test_a_genuinely_orphaned_context_is_reported_and_named(audit: ModuleType) -> None:
    result = audit.audit_repo_main(
        "OmniNode-ai",
        "somerepo",
        _gh(
            protection=_protection(["CI Summary", "Renamed Job That No Longer Exists"]),
            push_shas=["a" * 40],
            pr_shas=["b" * 40],
            runs_by_sha={"a" * 40: ["CI Summary"], "b" * 40: ["CI Summary"]},
        ),
    )
    assert result["status"] == "violation"
    assert result["orphan_contexts"] == ["Renamed Job That No Longer Exists"]
    assert "Renamed Job That No Longer Exists" in result["message"]


def test_orphan_detection_survives_a_large_pr_evidence_set(audit: ModuleType) -> None:
    """A wide evidence set must not drown a single real orphan."""
    noise = [f"Job {i}" for i in range(120)]
    result = audit.audit_repo_main(
        "OmniNode-ai",
        "somerepo",
        _gh(
            protection=_protection(["Job 7", "Deleted Gate"]),
            push_shas=["a" * 40],
            pr_shas=["b" * 40],
            runs_by_sha={"a" * 40: [], "b" * 40: noise},
        ),
    )
    assert result["orphan_contexts"] == ["Deleted Gate"]


# ---------------------------------------------------------------------------
# Matching mirrors GitHub, not string equality.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("required", "reported"),
    [
        ("deploy-gate / deploy-gate", "deploy-gate"),
        (
            "call-reject-skip-token / scan / reject-skip-gate-token",
            "reject-skip-gate-token",
        ),
        ("verify / verify", "verify"),
    ],
)
def test_a_reusable_workflow_leaf_name_satisfies_its_required_context(
    audit: ModuleType, required: str, reported: str
) -> None:
    assert audit.find_orphan_contexts([required], {reported}) == []


def test_checks_form_protection_is_read_as_well_as_contexts_form(
    audit: ModuleType,
) -> None:
    required = audit.parse_required_contexts(
        _protection(contexts=["CI Summary"], checks=["CI Summary", "Tests Gate"])
    )
    assert required == ["CI Summary", "Tests Gate"]


# ---------------------------------------------------------------------------
# An empty evidence read is not evidence of absence (rule 16).
# ---------------------------------------------------------------------------
def test_zero_collected_check_runs_is_indeterminate_not_a_mass_orphan(
    audit: ModuleType,
) -> None:
    result = audit.audit_repo_main(
        "OmniNode-ai",
        "somerepo",
        _gh(
            protection=_protection(["CI Summary", "Tests Gate"]),
            push_shas=["a" * 40],
            pr_shas=["b" * 40],
            runs_by_sha={},
        ),
    )
    assert result["status"] == "indeterminate"
    assert result["orphan_contexts"] == []
    assert "evidence source unresolved" in result["message"]


def test_indeterminate_does_not_fail_the_audit(audit: ModuleType, capsys: Any) -> None:
    """An unresolvable read reports loudly and judges nothing."""
    result = audit.audit_repo_main(
        "OmniNode-ai",
        "somerepo",
        _gh(
            protection=_protection(["CI Summary"]),
            push_shas=[],
            pr_shas=[],
            runs_by_sha={},
        ),
    )
    assert result["status"] == "indeterminate"


def test_unreachable_protection_skips_rather_than_reporting_clean(
    audit: ModuleType,
) -> None:
    result = audit.audit_repo_main(
        "OmniNode-ai",
        "somerepo",
        _gh(
            protection="",
            push_shas=[],
            pr_shas=[],
            runs_by_sha={},
            protection_rc=1,
        ),
    )
    assert result["status"] == "skip"


def test_empty_required_contexts_is_ok(audit: ModuleType) -> None:
    """Release-synced mains carry no required contexts by design (OMN-16289)."""
    result = audit.audit_repo_main(
        "OmniNode-ai",
        "omnibase_core",
        _gh(protection=_protection([]), push_shas=[], pr_shas=[], runs_by_sha={}),
    )
    assert result["status"] == "ok"
    assert result["required_contexts"] == []


# ---------------------------------------------------------------------------
# Pagination: the PR head SHA carried 154 check-runs live.
# ---------------------------------------------------------------------------
def test_check_run_collection_paginates_past_the_first_page(audit: ModuleType) -> None:
    page1 = [f"Job {i}" for i in range(audit.PAGE_SIZE)]
    page2 = ["Job on page two"]

    def gh(argv: list[str]) -> tuple[int, str]:
        if argv[1].endswith("&page=1"):
            return 0, _check_runs(page1)
        if argv[1].endswith("&page=2"):
            return 0, _check_runs(page2)
        return 0, json.dumps({"total_count": 0, "check_runs": []})

    seen = audit.collect_check_run_names("OmniNode-ai", "somerepo", ["a" * 40], gh)
    assert "Job on page two" in seen
    assert len(seen) == audit.PAGE_SIZE + 1


def test_only_merged_pull_requests_contribute_evidence(audit: ModuleType) -> None:
    def gh(argv: list[str]) -> tuple[int, str]:
        return 0, json.dumps(
            [
                {"merged_at": None, "head": {"sha": "c" * 40}},
                {"merged_at": "2026-09-13T00:00:00Z", "head": {"sha": "d" * 40}},
            ]
        )

    shas = audit.collect_merged_pr_head_shas("OmniNode-ai", "somerepo", 30, gh)
    assert shas == ["d" * 40]


# ---------------------------------------------------------------------------
# No allowlist survives (OMN-18346 removed PR_ONLY_CONTEXTS).
# ---------------------------------------------------------------------------
def test_the_module_carries_no_context_allowlist(audit: ModuleType) -> None:
    """An allowlisted context is one whose disappearance can never be reported."""
    assert not hasattr(audit, "PR_ONLY_CONTEXTS")
    result = audit.audit_repo_main(
        "OmniNode-ai",
        "somerepo",
        _gh(
            protection=_protection(["main-target-guard"]),
            push_shas=["a" * 40],
            pr_shas=["b" * 40],
            runs_by_sha={"a" * 40: ["CI Summary"], "b" * 40: ["CI Summary"]},
        ),
    )
    assert result["orphan_contexts"] == ["main-target-guard"]


def test_the_module_has_no_mutation_path(audit: ModuleType) -> None:
    """Read the argument parser's own options, not the prose around them.

    A substring scan would trip over the docstring sentence explaining that there
    is no fix path — and would equally be satisfied by a fix flag spelled any
    other way. The parser is the thing that decides what this script can be asked
    to do.
    """
    import contextlib
    import io

    help_text = io.StringIO()
    with contextlib.redirect_stdout(help_text), contextlib.suppress(SystemExit):
        audit.main(["--help"])
    options = {
        token.strip(" ,")
        for line in help_text.getvalue().splitlines()
        for token in line.split()
        if token.startswith("--")
    }
    assert "--fix" not in options
    assert "--force" not in options
    # No mutating verb reaches `gh` from here.
    source = _SCRIPT.read_text()
    assert "--method PUT" not in source
    assert "--method DELETE" not in source
    assert "--method PATCH" not in source
