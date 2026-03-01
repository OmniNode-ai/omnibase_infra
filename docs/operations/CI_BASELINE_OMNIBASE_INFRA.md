# CI Baseline — omnibase_infra (GitHub-hosted runners)

Recorded: 2026-03-01
Ticket: OMN-3274
Purpose: Pre-rollout baseline for self-hosted runner migration (OMN-3273)

---

## Runner Group

- **Group name**: `omnibase-ci` (created 2026-03-01, id=3)
- **Visibility**: selected
- **Allowed repos**: omnibase_infra, omniclaude, omnibase_core, omniintelligence
- **Public repo access**: false
- **Restricted to workflows**: false

Verified via:

```bash
gh api /orgs/OmniNode-ai/actions/runner-groups --jq '.runner_groups[] | select(.name == "omnibase-ci")'
gh api /orgs/OmniNode-ai/actions/runner-groups/3/repositories --jq '.repositories[] | {id: .id, name: .name}'
```

---

## CI Timing Baseline (GitHub-hosted runners)

Sample of 10 most recent runs captured on 2026-03-01.
All runs used GitHub-hosted `ubuntu-latest` runners.
Queue time was 0s for all samples (runners available immediately).

| Run ID | Workflow | Branch | Conclusion | Queue (s) | Exec (s) | Total (s) |
|--------|----------|--------|------------|-----------|----------|-----------|
| 22551223626 | Test Suite | main | success | 0 | 589 | 589 |
| 22551031178 | Test Suite | gh-readonly-queue/main/pr-516 | success | 0 | 600 | 600 |
| 22550948777 | Test Suite | main | success | 0 | 596 | 596 |
| 22550858352 | Test Suite | epic/OMN-3266/OMN-3261 | success | 0 | 548 | 548 |
| 22550759942 | Test Suite | gh-readonly-queue/main/pr-515 | success | 0 | 606 | 606 |
| 22550554658 | Test Suite | jonah/prevention-kafka-boundary-tests | success | 0 | 669 | 669 |
| 22550550828 | Test Suite | jonah/prevention-kafka-boundary-tests | failure | 0 | 89 | 89 |
| 22550508776 | Test Suite | jonah/prevention-topic-lint-python | success | 0 | 592 | 592 |

> Note: Failed run (89s) reflects early exit — not representative of full suite duration.

### Summary Statistics (successful Test Suite runs)

| Metric | Value |
|--------|-------|
| Sample count | 7 successful runs |
| Min exec time | 548s (~9m 8s) |
| Max exec time | 669s (~11m 9s) |
| Mean exec time | 600s (~10m 0s) |
| Median exec time | 596s (~9m 56s) |
| Queue time | 0s (all samples) |

### Key Observations

- Test suite runs consistently in the **9–11 minute** range on GitHub-hosted runners
- No queue time observed — GitHub-hosted runners are available immediately
- Self-hosted runners may introduce queue time if all 5 runners are busy
- Self-hosted runners expected to be faster on execution (local network, no cold start)
- Post-rollout: compare `exec_secs` against this baseline to quantify improvement

---

## How to Reproduce

```bash
gh run list --repo OmniNode-ai/omnibase_infra --limit 10 \
  --json databaseId,createdAt,updatedAt,startedAt,workflowName,conclusion,headBranch
```

---

## Post-Rollout Comparison

After deploying self-hosted runners (OMN-3275, OMN-3276, OMN-3277) and enabling routing
(OMN-3278), re-run the same query and compare:

- `exec_secs`: should decrease (local deps, no cold start)
- `queue_secs`: watch for queuing if all 5 runners are saturated
- `conclusion`: failure rate should remain the same or improve
