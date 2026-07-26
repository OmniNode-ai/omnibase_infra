# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the required-context parity ratchet (OMN-14288).

Exercises the PURE assertion logic in scripts/audit_branch_protection_lib.py
(normalize / direct-required / needs-closure / gate + manifest evaluation) with
synthetic inputs. No `gh`/network/subprocess — mirrors the extraction pattern of
test_branch_protection_audit.py.

The `TestReproducesConfirmedFindings` class pins the exact MISSING findings the
enforcement-parity audit confirmed live (omnimarket + omniclaude deploy-gate and
reject-skip on `dev`), so a future regression in the normalization or evaluator
that silently drops those findings fails CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_LIB_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "audit_branch_protection_lib.py"
)
_spec = importlib.util.spec_from_file_location("audit_branch_protection_lib", _LIB_PATH)
assert _spec is not None and _spec.loader is not None, f"cannot load {_LIB_PATH}"
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)


# ---------------------------------------------------------------------------
# normalize_context_forms
# ---------------------------------------------------------------------------


class TestNormalizeContextForms:
    def test_single_segment_is_itself(self) -> None:
        assert lib.normalize_context_forms("CI Summary") == {"CI Summary"}

    def test_reusable_context_adds_leaf(self) -> None:
        forms = lib.normalize_context_forms("deploy-gate / deploy-gate")
        assert forms == {"deploy-gate / deploy-gate", "deploy-gate"}

    def test_three_segment_reusable_adds_final_leaf(self) -> None:
        forms = lib.normalize_context_forms(
            "call-reject-skip-token / scan / reject-skip-gate-token"
        )
        assert "reject-skip-gate-token" in forms
        assert "call-reject-skip-token / scan / reject-skip-gate-token" in forms

    def test_strips_whitespace(self) -> None:
        assert lib.normalize_context_forms("  CI Summary  ") == {"CI Summary"}


# ---------------------------------------------------------------------------
# is_gate_directly_required
# ---------------------------------------------------------------------------


class TestIsGateDirectlyRequired:
    def test_exact_match(self) -> None:
        assert lib.is_gate_directly_required(
            "deploy-gate / deploy-gate", ["deploy-gate / deploy-gate", "CI Summary"]
        )

    def test_fuzzy_leaf_match(self) -> None:
        # required carries only the reported leaf; declared carries full path.
        assert lib.is_gate_directly_required(
            "deploy-gate / deploy-gate", ["deploy-gate", "CI Summary"]
        )

    def test_absent_is_false(self) -> None:
        assert not lib.is_gate_directly_required(
            "deploy-gate / deploy-gate", ["CI Summary"]
        )

    def test_ci_summary_does_not_falsely_cover_deploy_gate(self) -> None:
        # Guards the exact omnimarket/omniclaude case: contexts == ["CI Summary"]
        # must NOT be read as covering deploy-gate via normalization.
        assert not lib.is_gate_directly_required("deploy-gate", ["CI Summary"])


# ---------------------------------------------------------------------------
# compute_needs_closure
# ---------------------------------------------------------------------------


class TestComputeNeedsClosure:
    def test_transitive_closure(self) -> None:
        jobs = {
            "summary": {"needs": ["a", "b"]},
            "a": {"needs": ["c"]},
            "b": {},
            "c": {},
        }
        assert lib.compute_needs_closure(jobs, "summary") == {"a", "b", "c"}

    def test_string_needs_is_accepted(self) -> None:
        jobs = {"summary": {"needs": "only"}, "only": {}}
        assert lib.compute_needs_closure(jobs, "summary") == {"only"}

    def test_missing_root_yields_empty(self) -> None:
        assert lib.compute_needs_closure({"a": {}}, "nonexistent") == set()

    def test_cycle_terminates(self) -> None:
        jobs = {"a": {"needs": ["b"]}, "b": {"needs": ["a"]}}
        assert lib.compute_needs_closure(jobs, "a") == {"a", "b"}

    def test_root_excluded_from_closure(self) -> None:
        jobs = {"summary": {"needs": ["a"]}, "a": {}}
        assert "summary" not in lib.compute_needs_closure(jobs, "summary")


# ---------------------------------------------------------------------------
# evaluate_gate_parity
# ---------------------------------------------------------------------------


class TestEvaluateGateParity:
    def test_direct_covered_returns_none(self) -> None:
        gate = {"context": "deploy-gate / deploy-gate", "coverage": "direct"}
        assert (
            lib.evaluate_gate_parity("r", "dev", gate, ["deploy-gate / deploy-gate"])
            is None
        )

    def test_direct_missing_flagged(self) -> None:
        gate = {
            "context": "deploy-gate / deploy-gate",
            "coverage": "direct",
            "rule": "R",
        }
        finding = lib.evaluate_gate_parity("r", "dev", gate, ["CI Summary"])
        assert finding is not None
        assert finding["class"] == lib.PARITY_MISSING
        assert finding["gate"] == "deploy-gate / deploy-gate"
        assert finding["rule"] == "R"

    def test_unprotected_branch_flags_every_gate(self) -> None:
        gate = {"context": "CI Summary", "coverage": "direct"}
        finding = lib.evaluate_gate_parity("r", "dev", gate, None)
        assert finding is not None
        assert finding["class"] == lib.PARITY_UNPROTECTED

    def test_needs_child_covered_returns_none(self) -> None:
        gate = {
            "context": "occ-preflight",
            "coverage": "needs_child",
            "aggregator": "CI Summary",
            "aggregator_job_id": "ci-summary",
            "gate_job_id": "occ-preflight",
        }
        jobs = {"ci-summary": {"needs": ["occ-preflight", "lint"]}, "occ-preflight": {}}
        assert lib.evaluate_gate_parity("r", "dev", gate, ["CI Summary"], jobs) is None

    def test_needs_child_aggregator_not_required(self) -> None:
        gate = {
            "context": "occ-preflight",
            "coverage": "needs_child",
            "aggregator": "CI Summary",
            "aggregator_job_id": "ci-summary",
            "gate_job_id": "occ-preflight",
        }
        jobs = {"ci-summary": {"needs": ["occ-preflight"]}, "occ-preflight": {}}
        # CI Summary not in required -> aggregator itself unenforced.
        finding = lib.evaluate_gate_parity("r", "dev", gate, ["something-else"], jobs)
        assert finding is not None
        assert finding["class"] == lib.PARITY_NEEDS_CLOSURE

    def test_needs_child_gate_not_in_closure(self) -> None:
        gate = {
            "context": "deploy-gate",
            "coverage": "needs_child",
            "aggregator": "CI Summary",
            "aggregator_job_id": "ci-summary",
            "gate_job_id": "deploy-gate",  # lives in a separate workflow -> absent
        }
        jobs = {"ci-summary": {"needs": ["lint", "test"]}, "lint": {}, "test": {}}
        finding = lib.evaluate_gate_parity("r", "dev", gate, ["CI Summary"], jobs)
        assert finding is not None
        assert finding["class"] == lib.PARITY_NEEDS_CLOSURE
        assert "not in the transitive needs-closure" in finding["detail"]

    def test_needs_child_indeterminate_when_jobs_unresolved(self) -> None:
        gate = {
            "context": "occ-preflight",
            "coverage": "needs_child",
            "aggregator": "CI Summary",
            "aggregator_job_id": "ci-summary",
            "gate_job_id": "occ-preflight",
        }
        finding = lib.evaluate_gate_parity("r", "dev", gate, ["CI Summary"], None)
        assert finding is not None
        assert finding["class"] == lib.PARITY_INDETERMINATE

    def test_unknown_coverage_flagged(self) -> None:
        gate = {"context": "x", "coverage": "sideways"}
        finding = lib.evaluate_gate_parity("r", "dev", gate, ["x"])
        assert finding is not None
        assert finding["class"] == lib.PARITY_INVALID_MANIFEST


# ---------------------------------------------------------------------------
# evaluate_manifest_parity + confirmed-findings regression
# ---------------------------------------------------------------------------


class TestReproducesConfirmedFindings:
    """Pin the exact MISSING findings the enforcement-parity audit confirmed
    live on 2026-07-10 (omnimarket + omniclaude deploy-gate + reject-skip)."""

    def _manifest(self) -> dict:
        return {
            "repos": {
                "omnibase_core": {
                    "dev": {
                        "load_bearing_gates": [
                            {
                                "context": "deploy-gate / deploy-gate",
                                "coverage": "direct",
                            },
                            {
                                "context": "call-reject-skip-token / scan / reject-skip-gate-token",
                                "coverage": "direct",
                            },
                            {"context": "CI Summary", "coverage": "direct"},
                        ]
                    }
                },
                "omnimarket": {
                    "dev": {
                        "load_bearing_gates": [
                            {
                                "context": "deploy-gate / deploy-gate",
                                "coverage": "direct",
                            },
                            {
                                "context": "call-reject-skip-token / scan / reject-skip-gate-token",
                                "coverage": "direct",
                            },
                            {"context": "CI Summary", "coverage": "direct"},
                            {
                                "context": "occ-preflight",
                                "coverage": "needs_child",
                                "aggregator": "CI Summary",
                                "aggregator_job_id": "ci-summary",
                                "gate_job_id": "occ-preflight",
                            },
                        ]
                    }
                },
                "omniclaude": {
                    "dev": {
                        "load_bearing_gates": [
                            {"context": "deploy-gate", "coverage": "direct"},
                            {
                                "context": "scan / reject-skip-gate-token",
                                "coverage": "direct",
                            },
                            {"context": "CI Summary", "coverage": "direct"},
                        ]
                    }
                },
            }
        }

    def _live(self) -> dict:
        # Live required contexts verified 2026-07-10 via gh api.
        return {
            "omnibase_core:dev": {
                "required": [
                    "call-reject-skip-token / scan / reject-skip-gate-token",
                    "CI Summary",
                    "deploy-gate / deploy-gate",
                ],
                "aggregator_jobs": {},
            },
            "omnimarket:dev": {
                "required": ["CI Summary"],
                "aggregator_jobs": {
                    "CI Summary": {
                        "ci-summary": {"needs": ["occ-preflight", "lint", "test"]},
                        "occ-preflight": {},
                    }
                },
            },
            "omniclaude:dev": {
                "required": ["CI Summary"],
                "aggregator_jobs": {},
            },
        }

    def test_reproduces_m1_through_m4(self) -> None:
        findings = lib.evaluate_manifest_parity(self._manifest(), self._live())
        missing = {
            (f["repo"], f["gate"]) for f in findings if f["class"] == lib.PARITY_MISSING
        }
        assert missing == {
            ("omnimarket", "deploy-gate / deploy-gate"),  # M1
            (
                "omnimarket",
                "call-reject-skip-token / scan / reject-skip-gate-token",
            ),  # M2
            ("omniclaude", "deploy-gate"),  # M3
            ("omniclaude", "scan / reject-skip-gate-token"),  # M4
        }

    def test_control_repo_and_covered_gates_produce_no_findings(self) -> None:
        findings = lib.evaluate_manifest_parity(self._manifest(), self._live())
        # omnibase_core wired all three gates correctly -> no findings.
        assert not [f for f in findings if f["repo"] == "omnibase_core"]
        # omnimarket CI Summary (direct) + occ-preflight (needs_child, in closure)
        # are covered -> the ONLY omnimarket findings are the two MISSING gates.
        market = [f for f in findings if f["repo"] == "omnimarket"]
        assert {f["class"] for f in market} == {lib.PARITY_MISSING}
        assert len(market) == 2

    def test_total_finding_count_is_exactly_four(self) -> None:
        findings = lib.evaluate_manifest_parity(self._manifest(), self._live())
        assert len(findings) == 4
        assert all(f["class"] == lib.PARITY_MISSING for f in findings)


# ---------------------------------------------------------------------------
# evaluate_merge_policy_parity — merge-policy dimension (OMN-14288 extension)
# ---------------------------------------------------------------------------


class TestEvaluateMergePolicyParity:
    def test_matching_policy_no_findings(self) -> None:
        policy = {"queue": "disabled", "strict": False}
        assert lib.evaluate_merge_policy_parity("r", "dev", policy, False, False) == []

    def test_queue_drift_flagged(self) -> None:
        # declared disabled but live queue present
        policy = {"queue": "disabled", "strict": False}
        findings = lib.evaluate_merge_policy_parity("r", "dev", policy, True, False)
        assert len(findings) == 1
        assert findings[0]["class"] == lib.PARITY_QUEUE_DRIFT
        assert findings[0]["observed"] == "enabled"
        assert findings[0]["declared"] == "disabled"

    def test_strict_drift_flagged(self) -> None:
        policy = {"queue": "disabled", "strict": True}
        findings = lib.evaluate_merge_policy_parity("r", "dev", policy, False, False)
        assert len(findings) == 1
        assert findings[0]["class"] == lib.PARITY_STRICT_DRIFT
        assert findings[0]["declared"] is True
        assert findings[0]["observed"] is False

    def test_both_dimensions_drift(self) -> None:
        policy = {"queue": "disabled", "strict": True}
        findings = lib.evaluate_merge_policy_parity("r", "dev", policy, True, False)
        classes = {f["class"] for f in findings}
        assert classes == {lib.PARITY_QUEUE_DRIFT, lib.PARITY_STRICT_DRIFT}

    def test_indeterminate_when_live_unresolved(self) -> None:
        policy = {"queue": "disabled", "strict": False}
        findings = lib.evaluate_merge_policy_parity("r", "dev", policy, None, None)
        assert {f["class"] for f in findings} == {lib.PARITY_INDETERMINATE}
        assert len(findings) == 2

    def test_partial_policy_only_asserts_declared_keys(self) -> None:
        # only queue declared -> strict is not asserted even if live strict is True
        policy = {"queue": "disabled"}
        assert lib.evaluate_merge_policy_parity("r", "dev", policy, False, True) == []

    def test_invalid_queue_value_flagged(self) -> None:
        policy = {"queue": "sometimes"}
        findings = lib.evaluate_merge_policy_parity("r", "dev", policy, False, False)
        assert findings[0]["class"] == lib.PARITY_INVALID_MANIFEST


class TestManifestMergePolicyIntegration:
    """Pin the live-verified spi QUEUE_DRIFT finding (spi dev still has a queue
    against the decided queue=disabled policy) alongside a clean control repo."""

    def _manifest(self) -> dict:
        return {
            "repos": {
                "omnibase_core": {
                    "dev": {"merge_policy": {"queue": "disabled", "strict": False}}
                },
                "omnidash": {
                    "dev": {"merge_policy": {"queue": "disabled", "strict": True}}
                },
                "omnibase_spi": {
                    "dev": {"merge_policy": {"queue": "disabled", "strict": False}}
                },
            }
        }

    def _live(self) -> dict:
        # Verified 2026-07-10: spi dev still has a live merge queue.
        return {
            "omnibase_core:dev": {"queue_enabled": False, "strict": False},
            "omnidash:dev": {"queue_enabled": False, "strict": True},
            "omnibase_spi:dev": {"queue_enabled": True, "strict": False},
        }

    def test_reproduces_spi_queue_drift_only(self) -> None:
        findings = lib.evaluate_manifest_parity(self._manifest(), self._live())
        assert len(findings) == 1
        f = findings[0]
        assert (f["repo"], f["class"]) == ("omnibase_spi", lib.PARITY_QUEUE_DRIFT)
        # core (queue off, strict off) + omnidash (queue off, strict on) match -> clean.
        assert not [x for x in findings if x["repo"] in {"omnibase_core", "omnidash"}]
