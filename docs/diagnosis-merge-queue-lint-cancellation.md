# Diagnosis: omnibase_infra dev merge-queue stall — Lint job cancellation

**PRs:** #1789, #1781, #1782, #1792
**Date:** 2026-05-30
**Status:** Blocked on infrastructure — requires runner-pool decision, not code.

## What is known

- All four PRs target `dev`, which is protected by a **merge queue** (`required_merge_queue`).
- The 5 required status contexts are: `CI Summary`, `Handler Contract Compliance`,
  `gate / CodeRabbit Thread Check`, `verify / verify`,
  `call-reject-skip-token / scan / reject-skip-gate-token`.
- A PR cannot enter the queue (`enqueuePullRequest` GraphQL mutation) until **all 5
  required contexts have POSTED a success conclusion**. Arming auto-merge does NOT enqueue.
- `CI Summary` is a job inside the `CI` workflow that aggregates the other CI jobs and
  exits non-zero if any sibling failed. It only posts a final conclusion when the CI
  workflow run completes.

## What was tried and why it failed

1. **Genuine code fixes (succeeded):**
   - #1789: real `ruff format` violation in `tests/ci/test_ci_workflow_resilience.py` → fixed.
     1 CodeRabbit token-hardening thread → resolved with evidence.
   - #1781: real `ruff format` violation in
     `src/omnibase_infra/runtime/message_dispatch_engine.py` → fixed.
     All 5 CodeRabbit threads resolved (2 substantive ones verified already-implemented
     in source: dict-envelope guard at ~line 1227; HANDLER_ERROR-scoped partial-success
     apply in `service_dispatch_result_applier.py`).
2. **Re-firing CI via empty commits (failed / counterproductive):** each push created a
   fresh `CI` workflow run. Runs sit `pending` in a backlog (18 queued observed), and
   when a runner finally picks one up, the **`Lint` job is CANCELLED** (conclusion=cancelled,
   empty/absent ruff output). A cancelled Lint → CI workflow run completes=failure →
   `CI Summary` never posts success → queue refuses enqueue. Re-pushing only deepened
   the backlog.
3. **enqueue/dequeue churn (partial):** #1782 (all contexts green) reached the queue but
   sits `UNMERGEABLE` despite every one of its merge_group workflows passing — a stale
   queue entry; dequeue+re-enqueue is the documented remedy but it re-lands at the back.

## Root cause hypothesis

The self-hosted runner pool intermittently **cancels the `Lint` job mid-run** (runner
recycle / egress fault — matches the runner-9 issue noted in task #24). This is NOT a
real format violation:

**Definitive proof:** on #1789's exact failing head `11929b2c2`,
`uv run ruff format --check src/ tests/` reports `3955 files already formatted` (exit 0)
locally. The code is clean; the CI Lint job dies for infrastructure reasons.

Because `CI Summary` depends on Lint, every Lint cancellation cascades to a missing/failed
`CI Summary`, which is a hard required context for the merge queue. The four PRs cannot
merge until a CI run completes with Lint succeeding.

## Proposed fix with rationale

This is an **infrastructure decision for the maintainer**, not a code change:

1. **Stabilize the runner pool** — recycle the runner that cancels jobs (the runner-9
   recycle deferred in task #24), or add capacity so jobs stop sitting `pending` for hours.
   Once Lint can run to completion, `CI Summary` posts success and the PRs enqueue normally.
2. **Then, in queue order:** #1782 is cleanest (all green, no code changes) — dequeue +
   re-enqueue to clear its stale `UNMERGEABLE` entry first. #1789 (keystone) next so its
   auth fix lands and stops the unauthenticated-uv-sync git-fetch flake for everyone.
   Then #1781, then #1792.
3. **Do NOT** push more empty commits — it adds runs to the backlog and makes the stall
   worse. The correct lever is the runner pool, then a single clean CI run per head.

## Mechanism reference (for whoever finishes this)

- Enqueue: `gh api graphql -f query='mutation($id:ID!){enqueuePullRequest(input:{pullRequestId:$id}){mergeQueueEntry{position state}}}' -f id=<PR_node_id>`
- Required-context check before enqueue: all 5 contexts must show `completed/success` on
  the PR head SHA via `gh api repos/.../commits/<sha>/check-runs`.
- A "missing" (not failed) required context = the workflow that owns it is still
  pending/queued or was cancelled — fix the runner, do not retry enqueue blindly.
