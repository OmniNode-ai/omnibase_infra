# Aug-5 managed-staging readiness / rollback — GO-NO-GO PACKET (template)

**Ticket:** OMN-15125 · **Plan row:** rolling plan §3 B7 · **Parent epic:** OMN-14724
**Field manifest (the seam):** [`docs/runbooks/managed-staging-proof-kit/fields.yaml`](managed-staging-proof-kit/fields.yaml)
**Seam test:** `tests/ci/test_managed_staging_proof_kit_seam.py`

> **This is a template. It executes nothing and authorizes nothing.** Assembling it
> does not authorize the Aug-5 window. It is the artifact the operator reads *in order
> to decide*.

## The standing rule this packet exists to satisfy

Rolling plan §3 B7's proof column, verbatim:

> August 5 (soak Aug 6) is a **target, not a forecast**, until the reconciled blocker
> graph, dated chain with slack, and T20 handoff are attached.

So three rows below — `reconciled_blocker_graph`, `dated_chain_with_slack`,
`t20_handoff` — are not decoration. **Until all three carry real content, the correct
`go_no_go_decision` is NO-GO by construction**, regardless of how green everything
else looks.

Section-level proof-class ceiling: every §3 cloud fact is `proof_class: receipt-bound`
(contractor self-report), never `live-readback`. Do not upgrade a row's proof class in
this packet beyond what its evidence source can actually support.

## How to use it

1. Copy to `docs/evidence/OMN-15125/<UTC-date>-aug5-readiness-rollback-packet.md`.
2. Fill every row from its evidence source. `a6_thresholds_with_live_samples` needs a
   **sub-table**: one row per signal — signal, numeric threshold, live sampled value,
   sample timestamp. A threshold with no live sample counts as unloaded.
3. `executable_rollback` requires a **dry-run output**, not an assertion that a
   rollback exists (ticket criterion: "proven by a named dry-run or readback").
4. `source_digest` must equal the OMN-15123 freeze packet's `source_digest`, and
   `previous_digest` must be captured **before** any promotion — after promotion the
   rollback target is unrecoverable from the live surface.
5. Record the decision, the decider, and the UTC timestamp in `go_no_go_decision`, and
   mirror the row into `omni_home:docs/tracking/ROLLING_WORK_LEDGER.md`.


## Fields

Every row is required. `Value` is filled at run time from `Evidence source`;
an empty or prose-only value cell means the packet is not complete.

| Field | What it is | Value (paste verbatim readback) | Evidence source (command / path) |
|---|---|---|---|
| `source_digest` | Source digest being promoted | _(unfilled)_ | `git -C <candidate repo> rev-parse HEAD  # must equal one_tenant_contract_freeze.source_digest` |
| `previous_digest` | Previous digest (the rollback target) | _(unfilled)_ | `kubectl -n <ns> rollout history deploy/<name> && kubectl -n <ns> get deploy/<name> -o jsonpath='{.spec.template.spec.containers[0].image}'  # record the currently-serving digest BEFORE promotion` |
| `amd64_manifest` | linux/amd64 manifest present for the promoted digest | _(unfilled)_ | `docker manifest inspect <image>@<digest> \| jq '.manifests[].platform'  # must include {"architecture":"amd64","os":"linux"}` |
| `config_hash` | Rendered config hash | _(unfilled)_ | `kubectl -n <ns> get cm <configmap> -o json \| jq -S '.data' \| shasum -a 256  # must equal one_tenant_contract_freeze.config_digest` |
| `policy_hash` | IAM/broker policy hash | _(unfilled)_ | `aws iam get-role-policy --role-name <node-role> --policy-name <policy> --query PolicyDocument \| jq -S . \| shasum -a 256` |
| `vulnerability_result` | Vulnerability scan result for the promoted digest | _(unfilled)_ | `aws ecr describe-image-scan-findings --repository-name <repo> --image-id imageDigest=<digest> --query 'imageScanFindingsSummary.findingSeverityCounts'` |
| `a6_thresholds_with_live_samples` | A6 numeric thresholds loaded, each with a live sample value | _(unfilled)_ | `B10 wiring readback (OMN-14735/OMN-14948) -- one row per threshold: signal, numeric threshold, live sampled value, timestamp. A threshold with no live sample is unloaded.` |
| `monitoring_owner_actions` | Staffed monitoring owner + the action each breach triggers | _(unfilled)_ | `docs/runbooks/managed-staging-canary-teardown-rollback.md §3 (abort path, owner + sequence); name the on-call human and the window they are staffed for` |
| `b12_psql_readback` | B12 psql readback proving the landing table exists | _(unfilled)_ | `docs/runbooks/managed-staging-canary-postgres-provisioning.md §3.4 -- paste the psql \d+ delivery_replay_canary_projection output` |
| `teardown_readback` | OMN-14772 teardown readback | _(unfilled)_ | `docs/runbooks/managed-staging-canary-teardown-rollback.md §2 T-4..T-8 -- paste the post-teardown steady-state assertion output (OMN-14772)` |
| `executable_rollback` | Executable rollback procedure, proven by dry-run (not asserted) | _(unfilled)_ | `docs/runbooks/managed-staging-canary-teardown-rollback.md §4.1 rollback tuple; dry-run: kubectl -n <ns> rollout undo deploy/<name> --to-revision=<n> --dry-run=server` |
| `reconciled_blocker_graph` | Reconciled blocker graph (every open blocker, with its owner) | _(unfilled)_ | `Linear: children of OMN-14724 with statusType != completed, plus their blockedBy edges -- attach the query output, not a paraphrase` |
| `dated_chain_with_slack` | Dated critical-path chain with explicit slack | _(unfilled)_ | `one row per chained item: item, owner, start date, end date, slack days; slack computed against the Aug-5 window (soak Aug 6)` |
| `t20_handoff` | T20 handoff (final linux/amd64 build + digest handoff from the contractor lane) | _(unfilled)_ | `the contractor handoff artifact under docs/plans/ or docs/handoff/ naming the T20/B1 final build; cite path + commit` |
| `go_no_go_decision` | Go / no-go decision, decider, UTC timestamp | _(unfilled)_ | `operator decision recorded in this packet + the corresponding row in docs/tracking/ROLLING_WORK_LEDGER.md` |
| `plan_row_binding` | Rolling plan §3 B7 bound to OMN-15125 | _(unfilled)_ | `git -C $OMNI_HOME/omni_home log -p -- docs/plans/ROLLING_SEVEN_DAY_PLAN.md \| grep -n 'OMN-15125'` |

## Related

- `docs/runbooks/managed-staging-canary-postgres-provisioning.md` — B12 landing table + §3.4 readback
- `docs/runbooks/managed-staging-canary-teardown-rollback.md` — teardown / abort / rollback ownership (B13)
- `src/omnibase_infra/topics/managed_staging_canary_catalog.py` + `..._namespace.yaml` — B7 `onex.mstg1.` catalog, epoch, zero-collision readback (OMN-14727)
- `scripts/proof/e2e_cloud_workflow_harness.py` — OMN-10858 end-to-end proof harness (`--live` defaults OFF)
- `omni_home:docs/plans/2026-07-17-managed-staging-verified-state-and-task-split.md` — lane B task split
