# Aug-5 managed-staging readiness / rollback — GO-NO-GO PACKET (dated instance, NO-GO)

**Ticket:** OMN-15125 · **Parent epic:** OMN-14724
**Field manifest (the seam):** [`docs/runbooks/managed-staging-proof-kit/fields.yaml`](../../runbooks/managed-staging-proof-kit/fields.yaml)
**Template this was copied from:** [`docs/runbooks/managed-staging-aug5-readiness-rollback-packet.md`](../../runbooks/managed-staging-aug5-readiness-rollback-packet.md)
**Seam test:** `tests/ci/test_managed_staging_proof_kit_seam.py`
**Instance date:** 2026-08-05 (UTC) · **Author lane:** fable-epsilon-0805-eve (build subagent, Sonnet 5)

> **This is a template instance, not an authorization.** Assembling this packet does
> not authorize the Aug-5 window. Per the packet's own standing rule (quoted from the
> unfilled template): *"Until `reconciled_blocker_graph`, `dated_chain_with_slack`,
> and `t20_handoff` carry real content, the correct `go_no_go_decision` is NO-GO by
> construction."* All three are filled below with real (if partial/gapped) content —
> and the honest reading of that content is **NO-GO for tonight's Aug-5 window**. See
> `go_no_go_decision` at the bottom; read it before anything else in this packet.

## Fields

| Field | What it is | Value | Evidence source | Status |
|---|---|---|---|---|
| `source_digest` | Source digest being promoted | **`a853500ab3c74620acdba34b418f6bd087081153`** | ECR tag on the round-4 candidate; must equal `one_tenant_contract_freeze.source_digest` — cross-checked against `docs/evidence/OMN-15123/2026-08-05-one-tenant-contract-freeze-round4.md`, **matches exactly** | **FILLED — matches the OMN-15123 round-4 packet.** |
| `previous_digest` | Previous digest (the rollback target) | **runtime `sha256:d2a0cac015bb86fb3dd35939d5b9f064f8b69deb17f68f6bdd019986132eda46` (tag `candidate-d53e3cf-20260805004651`) · migrate `sha256:268061a07e069c89c1fefd3b824a28ddf45ddd24a1dc8a8df8b5e28d114aa4da` @ commit `d53e3cfa9038fa1f066ed2269486b0df523d38a7`** | Manifest evidence source calls for `kubectl -n <ns> rollout history` — **not available, no k8s access.** Substituted with the immediately-prior pin in the build lineage (`omninode_infra#814`, OMN-15655 round 3, per `ROLLING_WORK_LEDGER.md:12496`), independently re-read from ECR (`probes/round_history_runtime.json`), not trusted from the ledger citation alone. | **FILLED, WITH A CAVEAT.** This is the round-3 build lineage predecessor, **not a live "currently-serving" readback** — nothing is confirmed actually deployed/serving anywhere this session can observe, so there is no live rollback target to read back from a running deployment. Treat this as the best-available candidate rollback target, not a proven one. |
| `amd64_manifest` | linux/amd64 manifest present for the promoted digest | **PRESENT — `architecture: amd64, os: linux`** (round-4 runtime) | `docker manifest inspect --verbose 272493677981.dkr.ecr.us-east-1.amazonaws.com/omninode-runtime@sha256:b829c56f...` — `probes/manifest_round4_runtime.json` | **FILLED — live, this instance.** Cross-check: the `previous_digest` (round-3) runtime image is also confirmed `amd64`/`linux` (`probes/manifest_round3_runtime.json`), so a rollback would not hit an architecture mismatch. |
| `config_hash` | Rendered config hash | _BLOCKED_ | `kubectl -n <ns> get cm ... | shasum -a 256` | **BLOCKED — no k8s access from this host** (same finding as OMN-15123/15124 round-4: direct `:6443` times out). Must equal `one_tenant_contract_freeze.config_digest`, which is itself BLOCKED for the same reason. |
| `policy_hash` | IAM/broker policy hash | _BLOCKED_ | `aws iam get-role-policy --role-name <node-role> ...` | **BLOCKED — the candidate's actual node/pod IAM role name is not known from this host.** `aws iam list-roles` in this account surfaces only cluster/infra-management roles (`omninode-k3s-*-node-role`, ASG lifecycle, GitHub Actions) — none is unambiguously the runtime pod's scoped IRSA role; resolving it needs a `kubectl` lookup of the service-account annotation, which is blocked. |
| `vulnerability_result` | Vulnerability scan result for the promoted digest | **CRITICAL: 4 · HIGH: 12 · MEDIUM: 6 · LOW: 1** (scan status: `COMPLETE`) | `aws ecr describe-image-scan-findings --repository-name omninode-runtime --image-id imageDigest=sha256:b829c56f...` — `probes/vuln_scan_round4.json` | **FILLED — live, this instance.** **Discrepancy flagged:** `aws ecr describe-images ... --query imageDetails[0].imageScanStatus` returns `null` for this same digest — the two ECR read APIs disagree; `describe-image-scan-findings` (used here) returns an explicit `{"status":"COMPLETE",...}` block and real findings, so it is treated as authoritative. **4 CRITICAL findings is a real signal against promotion** — not evaluated further here (severity triage is outside this packet's scope), but it is a fact the go/no-go decision must weigh, not a clean bill of health. |
| `a6_thresholds_with_live_samples` | A6 numeric thresholds loaded, each with a live sample value | **Loaded: 5/5. Live-sampled: 0/5.** | `omnimarket/src/omnimarket/nodes/node_canary_monitoring_gate_compute/thresholds.yaml` (OMN-14732/OMN-14948) + Linear readback of OMN-14736 | **PARTIALLY FILLED — see sub-table below.** All 5 thresholds are loaded and wired (OMN-14735/OMN-14948 both Done); **zero have a live sampled value**, because no evidence surface reachable this session shows an actual canary soak run's observed numbers. |
| `monitoring_owner_actions` | Staffed monitoring owner + the action each breach triggers | **Owner named; staffed window NOT named.** | `docs/runbooks/managed-staging-canary-teardown-rollback.md` §0/§3 | **PARTIALLY FILLED.** §0 ownership table: abort-call owner and live-execution owner = **Jonah** (staffed operator), on the **agent's** B10-breach detection signal (agent owns detection + presents the halt, per §3). Abort sequence (§3): stop producer → halt consumers → snapshot breach evidence → run teardown from T-4. **No specific on-call staffed *window* (hours/timezone coverage) is named anywhere in the runbook** — it names the person (Jonah), not a shift. Flagged as a real gap, not filled with an invented window. |
| `b12_psql_readback` | B12 psql readback proving the landing table exists | _BLOCKED — genuine tension flagged_ | `docs/runbooks/managed-staging-canary-postgres-provisioning.md` §3.4 | **BLOCKED, AND FLAGGED.** §3.4 ("Readback — prove the landing table exists (ACCEPTANCE GATE)") is marked `HELD` in the live runbook — no filled `psql \d+` output is committed there. **Yet Linear OMN-14737 ("B12: Provision the canary Postgres landing table") shows `status: Done`, `completedAt: 2026-07-27`.** This is an unresolved discrepancy between the ticket status and the artifact its own field manifest cites as evidence — not resolved here (this session has no RDS access to independently re-verify either way; `nc -zv` to the RDS endpoint times out from this host). **Flagged for the next session/operator to reconcile: either the runbook needs its §3.4 filled with the actual readback that justified Done, or the Done flip needs re-examination.** |
| `teardown_readback` | OMN-14772 teardown readback | _BLOCKED_ | `docs/runbooks/managed-staging-canary-teardown-rollback.md` §2 T-4..T-8 | **BLOCKED, consistent with ticket state.** OMN-14772 (IMDS hop-limit revert, the T-7-adjacent residual) is live-checked this instance: `status: In Progress` (re-entered In Progress 2026-08-05T17:32:26Z after an In Review round), not Done — so a post-teardown steady-state assertion cannot exist yet. No tension here; ticket state and artifact absence agree. |
| `executable_rollback` | Executable rollback procedure, proven by dry-run (not asserted) | **Tuple NAMED, dry-run NOT run** | `docs/runbooks/managed-staging-canary-teardown-rollback.md` §4.1 + dry-run `kubectl rollout undo --dry-run=server` | **PARTIALLY FILLED.** The rollback tuple is named per §4.1's own required shape: previous image digest = round-3 (`previous_digest` row above), rollback owner = Jonah (live), rollback commands = standard `kubectl -n <ns> rollout undo deploy/<name> --to-revision=<n>` against the round-3-pinned deployment once one exists, reconciliation query = the same `b12_psql_readback`/topic-catalog checks this packet already cites, abort thresholds = the A6 table above. **The dry-run itself is BLOCKED — no k8s access.** Per the ticket's own AC wording ("proven by dry-run, not asserted"), naming the tuple does not satisfy this field; only the dry-run output would. |
| `reconciled_blocker_graph` | Reconciled blocker graph (every open blocker, with its owner) | **See table below — live `list_issues(parentId=OMN-14724)` readback, 11 children, this instant** | Linear MCP `list_issues` + `get_issue`, this session | **FILLED — live, this instance.** |
| `dated_chain_with_slack` | Dated critical-path chain with explicit slack | **See table below** | Derived from the reconciled blocker graph + this session's own live findings | **FILLED, with an honest verdict: negative slack.** |
| `t20_handoff` | T20 handoff (final linux/amd64 build + digest handoff from the contractor lane) | _NOT FOUND_ | `docs/plans/` / `docs/handoff/` search for a T20/B1 contractor handoff artifact | **BLOCKED — genuinely not found, not just unreached.** `docs/plans/2026-07-17-managed-staging-verified-state-and-task-split.md:135` maps `T20 (main-lineage candidate image) → B1 (Lane B)`, but no discrete "T20 handoff" document exists under `docs/plans/` or `docs/handoff/` naming a final linux/amd64 build handoff from a contractor lane. The closest analog is the round-4 build provenance itself — GitHub Actions runs `31048750499` (runtime) / `31048074889` (migrate), both ECR-verified this session — but that is a CI build log, not the contractor handoff artifact the manifest specifically asks for. Not force-fit; recorded as a real gap. |
| `go_no_go_decision` | Go / no-go decision, decider, UTC timestamp | **NO-GO — see below** | this packet + `docs/tracking/ROLLING_WORK_LEDGER.md` | **FILLED — see "Go/no-go decision" section below.** |
| `plan_row_binding` | Rolling plan §3 B7 bound to OMN-15125 | **RESOLVED — same fix as OMN-15123/OMN-15124** | live grep, `docs/plans/ROLLING_SEVEN_DAY_PLAN.md` | **FILLED.** `OMN-15125` present at `§0-CHAIN` link 6 (line 38) and `§2` "Fastest readiness order" item 6 (line 93), both citing it by ID directly. Resolved by the 2026-08-04T03:58:31Z plan-governor reconcile (`ROLLING_WORK_LEDGER.md:12253`) alongside OMN-15123/OMN-15124; re-confirmed live here, not trusted from the ledger claim alone. |

## `a6_thresholds_with_live_samples` — sub-table (required by the template)

| Signal | Numeric threshold | Comparison | Live sampled value | Sample timestamp |
|---|---|---|---|---|
| auth | 2 (count) | gte | **NONE — unloaded, no live sample exists** | — |
| tls | 1 (count) | gte | **NONE — unloaded, no live sample exists** | — |
| broker | 3 (count) | gte | **NONE — unloaded, no live sample exists** | — |
| lag | 5 (messages) | gte | **NONE — unloaded, no live sample exists** | — |
| rds | 2 (count) | gte | **NONE — unloaded, no live sample exists** | — |

Per the manifest's own field definition ("a threshold with no live sample counts as unloaded"): **all 5 signals count as unloaded**, despite all 5 being correctly configured and wired in code (OMN-14735/OMN-14948, both Done, thresholds sourced verbatim from the A6 contractor deliverable OMN-14732). This is a code-readiness fact, not a proof-readiness fact — the gate is wired, it has simply never fired against a real soak.

## `reconciled_blocker_graph` — live children of OMN-14724, this instant

| Ticket | Title (short) | Status | Owner | Note |
|---|---|---|---|---|
| OMN-15123 | Freeze the immutable canary tuple | In Progress | Jonah Gray | This session's own round-4 PR (#2667) advances it; not closed. |
| OMN-15124 | Candidate-isolation compatibility proof | In Progress | Jonah Gray | This session's own round-4 PR (#2668) advances it; not closed. |
| OMN-15125 | THIS ticket | In Progress | Jonah Gray | This packet is evidence toward, not closure of, this ticket. |
| OMN-14733 | B2: scale worker capacity off desired=0 | **Canceled** (2026-08-05T18:17:07Z) | Daniyal Abbas | **Canceled today**, was `axis:awaiting-operator-decision`. If worker capacity is genuinely needed to run the canary and this stays canceled, that is itself a live-blocking gap for B11-class work — not re-litigated here, flagged for the operator. |
| OMN-15253 | Typed staging-readiness contract (slice 1) | Done | Jonah Gray | Closed. |
| OMN-14734 | B3: reset Valkey credential | Done | Daniyal Abbas | Closed. |
| OMN-14737 | B12: provision canary Postgres landing table | Done (completedAt 2026-07-27) | Jonah Gray | **Tension flagged in the `b12_psql_readback` row above** — the runbook artifact this field cites as evidence shows §3.4 still `HELD`. |
| OMN-14736 | B11: run the one-tenant MSK backend canary | Done (completedAt 2026-07-27) | Daniyal Abbas | No independent evidence of a completed run was found reachable from this host this session (RDS/k8s both unreachable here); Daniyal likely has in-VPC access this host does not, so this is **not** asserted as a false-Done — recorded as unverified-from-this-vantage-point, consistent with the `a6_thresholds_with_live_samples` finding of zero live samples anywhere this session could reach. |
| OMN-14735 | B10: wire monitoring to numeric thresholds | Done | Jonah Gray | Closed, code-verified (thresholds.yaml exists, cited above). |
| OMN-14779 | B12↔B6 seam CI test | Done | Jonah Gray | Closed. |
| OMN-14642 | Dashboard reads via effect-node adapter (bridge) | Done | Jonah Gray | Closed. |
| OMN-14738 | B13: define teardown/rollback | Done | Jonah Gray | Closed — this is the runbook itself, which is why its still-`HELD` execution markers are meaningful (the *spec* is Done; the *execution* is not, by the spec's own design). |

**Net:** of 12 non-terminal-or-just-closed children, 3 are the OMN-15123/15124/15125 triad this lane is actively advancing, 1 (B2) went Canceled today with an unresolved capacity question, and 2 "Done" tickets (B11, B12) have artifacts that either can't be independently verified from this host (B11) or show a directly contradicting `HELD` marker in their own cited evidence surface (B12). **A clean blocker graph would show zero such tensions; this one shows two, both worth an operator pass**, not silently inherited as clean.

## `dated_chain_with_slack`

| Item | Owner | Start | Target end | Slack (vs. tonight's Aug-5 window) |
|---|---|---|---|---|
| Round-4 candidate build (runtime+migrate) | build lane | 2026-08-05T21:22Z (migrate push) / 21:33Z (runtime push) | complete | 0 — already built |
| Pin PRs `omninode_infra#818`/`#819` merge | Codex (queue) | opened 2026-08-05T21:35Z / later | **unmerged as of this packet** | **negative** — both still OPEN, one (`#818`) has a genuine unresolved test conflict (`test_migration_image_digest_matches_the_pinned_job` asserts the round-3 digest, not round-4's — per `ROLLING_WORK_LEDGER.md:12715`), not a flake |
| OMN-15123 freeze (`freeze_signature`) | this ticket chain | blocked on the pin PRs above | **not started** | **negative** — cannot start until the pins land |
| k8s/DB access (SSM tunnel) for the 5+ BLOCKED fields across OMN-15123/15124/15125 | unowned this session | not started | **unknown** | **unknown — no owner named, no ETA found** |
| Canary Postgres §3.4 readback (b12) | unresolved (see tension above) | unknown | unknown | **unknown** |
| Aug-5 window itself (soak Aug 6) | — | — | **tonight, 2026-08-05** | **the window is now; every upstream item above still shows 0 or negative slack against it** |

**Verdict: negative slack across the chain.** The pin PRs that gate everything downstream are open with a live, non-flake test conflict; the k8s/DB access needed to clear the remaining BLOCKED fields has no named owner or ETA in this session's reach; and two "Done" tickets in the blocker graph show artifacts that don't independently confirm from this vantage point. This is not a chain with a tight-but-real Aug-5 landing — it is a chain that has not yet resolved its own gating conflict.

## Go/no-go decision

**Decision: NO-GO for the 2026-08-05 window.**
**Decider:** this build lane (fable-epsilon-0805-eve), stating the mechanical conclusion the template's own rule requires — **not** an operator override. An operator can still say GO on different information; this packet's job is to make sure that decision is made with the real state in front of it, not a clean-looking but incomplete packet.
**UTC timestamp:** 2026-08-05 (packet assembly time, this instance).

**Why NO-GO, not "packet incomplete, defer decision":**
1. The candidate tuple is not yet pinned — `omninode_infra#818`/`#819` open, `#818` has a live test-assertion conflict against the round-3 digest (not resolved by this packet).
2. `freeze_signature` (OMN-15123) is correctly unfilled — there is no frozen tuple to promote.
3. 4 CRITICAL vulnerability findings on the round-4 image are unaddressed (not triaged in this packet — flagged, not waived).
4. Two blocker-graph tensions (B11/B12 "Done" vs. their own cited evidence) are unresolved.
5. Zero live A6 threshold samples exist anywhere this session could reach.
6. `t20_handoff` genuinely does not exist as a named artifact.

This mirrors the corresponding conclusion in `docs/tracking/2026-08-05-beta-blocking-axis-register.md` (EOD-refreshed the same day, `docs/tracking/ROLLING_WORK_LEDGER.md:12681`), which independently tracks OMN-15123/OMN-15124/OMN-15125 as still not-closed under the broader beta-blocking axis register — no contradiction found between that surface and this one.

## Probes committed alongside this packet

- `probes/vuln_scan_round4.json`
- `probes/manifest_round4_runtime.json`
- `probes/manifest_round3_runtime.json`
- `probes/round_history_runtime.json`

## Operator / Daniyal handoff

1. Resolve the `omninode_infra#818` digest-assertion conflict (round-3 vs round-4) before expecting either pin PR to merge cleanly.
2. Reconcile the B11/B12 "Done" vs. cited-evidence tension — either fill the runbook's §3.4 with the readback that justified B12's Done, or re-open and re-verify.
3. Name an owner + ETA for the SSM-tunnel k8s access this and the sibling OMN-15123/15124 packets both need to clear their remaining BLOCKED fields.
4. Decide the B2 (worker capacity) cancellation's downstream effect on B11-class execution capacity — recorded as canceled today with no visible successor ticket.
5. Locate or explicitly declare-absent the T20 contractor handoff artifact the manifest expects.

## Related

- `docs/evidence/OMN-15123/2026-08-05-one-tenant-contract-freeze-round4.md` — sibling round-4 freeze packet (source of `source_digest` cross-check)
- `docs/evidence/OMN-15124/2026-08-05-candidate-isolation-round4-evidence.md` — sibling round-4 isolation-proof evidence
- `docs/tracking/2026-08-05-beta-blocking-axis-register.md` — independent, same-day reconciliation of the broader beta blocking axes
- `docs/runbooks/managed-staging-canary-teardown-rollback.md` — rollback tuple §4.1, abort §3, ownership §0
- `docs/runbooks/managed-staging-canary-postgres-provisioning.md` — B12 §3.4 (still HELD)
- `omninode_infra#818` / `#819` — the open round-4 pin PRs this packet's `source_digest`/`previous_digest` depend on
