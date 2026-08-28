# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16878: receipt-honesty and contract-validation must stay load-bearing.

OMN-16876's enforcement census found both checks running on every omnibase_infra
PR while being mechanically unable to block one. Per Operating Rule 5, detection
not wired as a pre-merge gate is advisory and gets ignored.

The stakes are higher in this repo than elsewhere. `dev` requires exactly ONE
status check — ``CI Summary`` — by the OMN-4497 single-umbrella design, so
:data:`EXPECTED_EXTERNAL_CONTEXTS` *is* the entire external enforcement surface.
A context missing from that tuple has no second surface to fall back on and no
branch-protection signal that it is missing; it simply stops mattering, silently.
That is precisely how OMN-13326 and OMN-13328 came to be marked Done while
neither context was enforced here.

These tests pin the tuple membership and the producer-side properties that make
membership meaningful.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXPECTED_EXTERNAL_CONTEXTS,
    EXTERNAL_GOOD_CONCLUSIONS,
    MEASURED_NOT_ENFORCED_CONTEXTS,
)

REPO_ROOT = Path(__file__).parent.parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# context name -> producing workflow file
NEWLY_ENFORCED: dict[str, str] = {
    "receipt-honesty": "receipt-honesty.yml",
    "contract-validation": "contract-validation.yml",
}


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def _triggers(workflow: dict) -> set[str]:
    # PyYAML parses a bare `on:` key as the boolean True.
    raw = workflow.get(True, workflow.get("on"))
    if isinstance(raw, dict):
        return set(raw)
    if isinstance(raw, list):
        return set(raw)
    return {str(raw)}


@pytest.mark.unit
@pytest.mark.parametrize("context", sorted(NEWLY_ENFORCED))
class TestEnforcementWiring:
    def test_context_is_asserted_by_ci_summary(self, context: str) -> None:
        """Membership in the tuple is the ONLY thing making this context block."""
        assert context in EXPECTED_EXTERNAL_CONTEXTS, (
            f"{context!r} left EXPECTED_EXTERNAL_CONTEXTS. On this repo that is "
            "not a downgrade to a second surface — dev requires only 'CI "
            "Summary', so the context would stop gating merges entirely, with "
            "nothing in branch protection to reveal it. This is the OMN-13326 / "
            "OMN-13328 false-Done failure mode."
        )

    def test_context_is_not_also_declared_unenforced(self, context: str) -> None:
        """A name cannot be both asserted and recorded as deliberately-not-enforced."""
        assert context not in MEASURED_NOT_ENFORCED_CONTEXTS, (
            f"{context!r} appears in both EXPECTED_EXTERNAL_CONTEXTS and "
            "MEASURED_NOT_ENFORCED_CONTEXTS. Those are contradictory claims "
            "about the same context."
        )

    def test_producer_reports_on_pr_and_merge_group(self, context: str) -> None:
        """A required context that cannot report on a queue SHA wedges the queue."""
        triggers = _triggers(_workflow(NEWLY_ENFORCED[context]))
        assert "pull_request" in triggers, (
            f"{context!r}: producer has no pull_request trigger, so branch "
            "protection could never see it satisfied."
        )
        assert "merge_group" in triggers, (
            f"{context!r}: producer has no merge_group trigger. Should a queue "
            "ever be re-enabled on this repo, an asserted context that never "
            "reports on the queue SHA wedges every merge."
        )

    def test_producer_job_has_no_skip_path(self, context: str) -> None:
        """No `needs:` and no job-level `if:` means nothing upstream can skip it.

        A skipped producer is the classic silent-pass shape (OMN-15057 vector 5).
        It is defused twice over here — see
        ``test_skipped_is_not_a_good_external_conclusion`` for the second layer —
        but the cheapest place to keep it defused is the producer itself.
        """
        workflow_file = NEWLY_ENFORCED[context]
        jobs = _workflow(workflow_file)["jobs"]
        # The context name is the bare job name, so exactly one job renders it.
        job = jobs.get(context)
        assert job is not None, (
            f"{workflow_file} has no job id {context!r}; the check-run name that "
            f"branch protection and CI Summary key on would change."
        )
        if job.get("needs"):
            assert str(job.get("if", "")).strip() == "always()", (
                f"{workflow_file}:{context} gained `needs:` without "
                "`if: always()`. GitHub's implicit job-level `if:` is success() "
                "over needs, so an upstream failure would SKIP this job."
            )


@pytest.mark.unit
def test_skipped_is_not_a_good_external_conclusion() -> None:
    """The assertion layer must treat a skipped external context as a failure."""
    assert frozenset({"success"}) == EXTERNAL_GOOD_CONCLUSIONS, (
        "Admitting 'skipped' here would let a skipped producer satisfy the one "
        "required context this repo has."
    )
