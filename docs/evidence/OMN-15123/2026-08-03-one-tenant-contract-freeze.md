# Managed-staging one-tenant contract — FREEZE PACKET (partial, dated instance)

**Ticket:** OMN-15123 · **Plan row:** rolling plan §3 B3 (structural note below — this
citation no longer resolves against the live plan) · **Parent epic:** OMN-14724
**Field manifest (the seam):** [`docs/runbooks/managed-staging-proof-kit/fields.yaml`](../../runbooks/managed-staging-proof-kit/fields.yaml)
**Template this was copied from:** [`docs/runbooks/managed-staging-one-tenant-contract-freeze.md`](../../runbooks/managed-staging-one-tenant-contract-freeze.md)
**Seam test:** `tests/ci/test_managed_staging_proof_kit_seam.py`
**Instance date:** 2026-08-03 (UTC) · **Author lane:** fable-beta-0803/G1 (build agent, Sonnet 5)

> ## THIS IS NOT A FREEZE. Read this before reading the table.
>
> A freeze requires every row below to carry a real, current readback (AC #1/#2).
> **9 of 19 rows are BLOCKED and carry no value** — this host has no live AWS
> session (SSO expired, human login pending) and no DB/deploy access, per this
> session's operating constraints. **A 10th row (`plan_row_binding`) is blocked
> for a structural reason unrelated to AWS** (see §ac3 below). Only the rows
> answerable from already-committed, offline repo state are filled.
>
> **Do not treat the commit that lands this file as the OMN-15123 freeze event.**
> `freeze_signature` below records only that this partial snapshot was committed;
> the real freeze happens when a later, complete instance supersedes this one
> with every BLOCKED row cleared by a live readback. Until then, OMN-15123's
> acceptance criteria remain unmet and the ticket must not be moved to Done from
> this artifact alone.

## AC-by-AC disposition

| # | Acceptance criterion | Status | Why |
|---|---|---|---|
| 1 | Committed, immutable contract artifact enumerating all 19 fields | **NOT MET** | This artifact exists and is committed, but only 8 of 19 fields carry a real value; 11 are BLOCKED (see table). Not immutable in the AC's sense until complete. |
| 2 | Digest fields read back equal to the live candidate at freeze time, command output attached | **NOT MET** | `source_digest`/`image_digest` are BLOCKED. The nearest known digest (`sha256:414944a4…`, OMN-14974, 2026-07-27T02:12Z per `ROLLING_WORK_LEDGER.md:8439`) is **7 days stale** relative to this instance date and is explicitly not usable as "at freeze time" — cited for context only, not as a filled value. |
| 3 | Rolling plan §3 B3's "unverifiable by construction" tag bound to this ticket id (plan diff cited) | **STRUCTURALLY BLOCKED, not AWS-blocked** | See §ac3. |
| 4 | OMN-14736 (B11 canary) references this frozen tuple as its input | **NOT MET** | The tuple isn't frozen (AC #1 unmet), so linking OMN-14736 to it now would misrepresent an incomplete artifact as authoritative input. Deferred to the complete instance. |

### §ac3 — the plan-row-binding field is blocked for a structural reason

The manifest's `plan_row_binding` evidence source reads:

```
git -C $OMNI_HOME/omni_home log -p -- docs/plans/ROLLING_SEVEN_DAY_PLAN.md | grep -n 'OMN-15123'  # cite the plan diff that replaced the 'unverifiable by construction' tag
```

Live grep against the current `docs/plans/ROLLING_SEVEN_DAY_PLAN.md` (2026-08-03),
run from this instance:

- No `§3` section heading exists anywhere in the file. The plan's decisions section
  is now `## 3. Decisions` with subsections `### 3a.` / `### 3b.` — a different
  structure than the "§3 B3" row the ticket and this manifest cite.
- The literal string `unverifiable by construction` does not appear anywhere in the
  current plan.
- `OMN-15123` **is** present in the plan (three hits, `§2` "Ranked seven-day work
  queue" chain-ordering prose, e.g. line 38: `(OMN-10858, OMN-15123, OMN-15124,
  OMN-15125)`), but none of those hits replace a "§3 B3 unverifiable-by-construction"
  tag — that row no longer exists in that form. The plan was restructured by the
  2026-08-01 §0-AIM rewrite (`docs/plans/ROLLING_SEVEN_DAY_PLAN.md` revision log,
  2026-08-01 entry) after the ticket was filed 2026-07-25 against the pre-rewrite
  section numbering.

This is a **plan-governor reconciliation gap**, not something a build lane can
close by editing the ticket's own citation: closing AC #3 correctly requires a
plan-governor pass over the live plan to either (a) locate the successor row the
old "§3 B3" content moved to and bind `OMN-15123` there explicitly, or (b) rule
that the row was subsumed by the §2 chain-ordering prose and record that ruling.
Neither is a build-lane decision. **Flagged here, not fixed.**

## Fields

| Field | What it is | Value | Evidence source | Status |
|---|---|---|---|---|
| `aws_account` | AWS account | _BLOCKED_ | `aws sts get-caller-identity --query Account --output text` | **BLOCKED — AWS SSO expired on this host, human login pending.** (Committed docs, e.g. `docs/runbooks/managed-staging-canary-teardown-rollback.md` §"Plane scope", name account `272493677981` — cited for context only; that is a doc reference, not the live readback this field requires, and is not treated as satisfying the field.) |
| `aws_region` | AWS region | _BLOCKED_ | `aws configure get region` | **BLOCKED — same AWS SSO gap.** (Same runbook names `us-east-1` for context only.) |
| `k8s_namespace` | Kubernetes namespace (single canary namespace) | _BLOCKED_ | `kubectl get ns <ns> -o jsonpath=...` | **BLOCKED — no live k8s access from this host/session.** |
| `msk_cluster_arn` | MSK cluster ARN | _BLOCKED_ | `aws kafka list-clusters-v2 ...` | **BLOCKED — AWS SSO gap.** |
| `rds_instance_identifier` | RDS instance identifier | _BLOCKED_ | `aws rds describe-db-instances ...` | **BLOCKED — AWS SSO gap.** |
| `gateway_endpoint` | The one gateway | _BLOCKED_ | `kubectl -n <ns> get svc,ingress ...` | **BLOCKED — no live k8s access.** |
| `synthetic_tenant_id` | The one synthetic tenant (UUID) | _BLOCKED_ | `psql "$CANARY_DSN" ...` | **BLOCKED — no DB access; the canary DB's provisioning status is itself unconfirmed from this host.** |
| `source_digest` | Candidate source digest | _BLOCKED (stale reference only)_ | `git -C <candidate repo> rev-parse HEAD` | **BLOCKED as "at freeze time."** Nearest known value: commit `710197a6` (omnibase_infra dev, per `ROLLING_WORK_LEDGER.md:8438`, 2026-07-27T02:05Z) — 7 days stale, not usable as current. |
| `image_digest` | Candidate image digest | _BLOCKED (stale reference only)_ | `aws ecr describe-images ...` | **BLOCKED as "at freeze time."** Nearest known value: `sha256:414944a4f1d543bd63119aa4172e0e90edfe2717e8d68dae6bdc5323fd81788a` (per `ROLLING_WORK_LEDGER.md:8439`, 2026-07-27T02:12Z) — 7 days stale, not usable as current, and not confirmed to be the same candidate tuple this freeze targets. |
| `config_digest` | Rendered runtime config digest | _BLOCKED_ | `kubectl -n <ns> get cm <configmap> ... \| shasum -a 256` | **BLOCKED — no live k8s access.** |
| `topic_catalog` | Approved `onex.mstg1.` topic/group catalog | **164 topics / 56 groups — see `probes/topic_catalog.txt`** | `uv run python -c '...build_canary_catalog_from_candidate...'` | **FILLED — offline, in-repo.** Ran on this host 2026-08-03; full sorted list committed alongside this packet (`probes/topic_catalog.txt`, 228 lines incl. headers). Generator: `src/omnibase_infra/topics/managed_staging_canary_catalog.py` (OMN-14727). |
| `zero_collision_readback` | Zero-collision readback | **`is_clean: True` against an EMPTY snapshot — see caveat** | `uv run python -c '...verify_zero_collision...'` | **FILLED, but OFFLINE-ONLY — not the live check the field label implies.** The module's own docstring is explicit: "The catalog + namespace live entirely in-repo, so the readback runs offline. The live readback against the actual 1089 topics + live consumer groups needs a broker connection from inside the VPC and is therefore deferred to Phase 3 (apply-to-cluster)." This run passed `existing_topics=[]`/`existing_groups=[]` (no live snapshot available), so `is_clean: True` proves only that the freshly generated 164-topic/56-group catalog has no *internal* collisions and the check code runs cleanly — it does **not** prove disjointness from the live cluster's 1089 topics. Full output: `probes/zero_collision_readback_offline.txt`. Live Phase-3 readback remains BLOCKED (no VPC/broker access from this host). |
| `msk_epoch` | Unique MSK epoch | **`mstg1`** | `src/omnibase_infra/topics/managed_staging_canary_catalog_namespace.yaml -> epoch` | **FILLED.** Read directly from the committed namespace file (line 22 area, `epoch: "mstg1"`). |
| `group_start_reset_policy` | Signed consumer-group start/reset policy | **`earliest` (policy value only — NOT signed)** | `..._namespace.yaml -> group_start_policy` + operator signature line | **PARTIALLY FILLED.** The policy value (`group_start_policy: "earliest"`) is a committed fact, read from the same namespace file. The manifest also requires an **operator signature line** in the filled packet — that signature does not exist yet; this row is not complete until an operator signs it. Not counted as BLOCKED (the value is known and committed) but not counted as satisfying the AC's "signed" requirement either. |
| `rollback_authority` | Named rollback authority | **See ownership table citation** | `docs/runbooks/managed-staging-canary-teardown-rollback.md` §0 | **FILLED.** §0 ownership table: trigger owner Jonah (planned teardown) / staffed operator Jonah (abort, on agent's B10 breach signal) / Jonah (rollback); live-execution owner is Jonah in all three paths; the agent prepares/validates/captures evidence. Supporting identity: the A3 create/delete-capable operator IAM identity (contractor-team-owned) for topic/group deletion — never the runtime identity. |
| `zero_prod_diff` | Zero-prod-diff assertion | **PASS (no prod resource named), with one documented self-match** | `grep -nE 'omnibase-infra-prod\|:28085\|:28086' <this packet>` | **FILLED.** Run against this file after it was written (`probes/zero_prod_diff_grep.txt`): exactly **one** match, this row's own `Evidence source` cell text (the grep pattern itself, quoting the strings it checks for) — that is the check's own documentation, not a prod resource used by the tuple. Excluding that self-referential row, zero matches; no actual prod resource is named anywhere in the tuple's data rows. |
| `omnidash_exclusion` | Omnidash exclusion | _BLOCKED_ | `kubectl -n <ns> get deploy -o jsonpath=...` | **BLOCKED — no live k8s access.** |
| `freeze_signature` | Freeze signature | **See note — not a valid freeze signature yet** | `git log -1 --format='%H %aI %an' -- <path>` | **DELIBERATELY NOT FILLED AS A FREEZE.** This row is reserved for the commit that lands the *complete* packet. This commit is not that commit (11 of 19 rows are BLOCKED or partial). Recording the landing commit here would misstate an incomplete artifact as the freeze event, which AC #1 explicitly guards against ("immutable" implies complete). Left unfilled by design; fill it only on the instance that clears every BLOCKED row above. |
| `plan_row_binding` | Rolling plan §3 B3 bound to OMN-15123 | _BLOCKED (structural)_ | `git log -p -- docs/plans/ROLLING_SEVEN_DAY_PLAN.md \| grep -n 'OMN-15123'` | **BLOCKED — see §ac3 above.** Not an AWS gap; a plan-governor reconciliation gap. |

## Probes committed alongside this packet

- `probes/topic_catalog.txt` — full offline-generated topic/group catalog (164 topics, 56 groups), `build_canary_catalog_from_candidate()` run 2026-08-03 on this host.
- `probes/zero_collision_readback_offline.txt` — offline `verify_zero_collision()` run against an empty snapshot; explicitly captioned as not the live check.
- `probes/zero_prod_diff_grep.txt` — raw `grep -nE` output against this file, with the two self-referential matches called out.

## Known drift flagged (do not inherit silently)

`src/omnibase_infra/topics/managed_staging_canary_catalog_namespace.yaml` line
~47's comment still reads *"The 2x kafka.t3.small managed cluster is already
~2.7x over AWS partition guidance"*. **This is stale.** OMN-15253's 2026-07-27
live readback (`ROLLING_WORK_LEDGER.md:8745`) corrected the live broker sizing to
**2x kafka.m5.large**. This packet does not carry a sizing field itself (sizing
lives in `default_partitions`/`default_replication_factor`, not in the frozen
tuple's field list), so there is nothing to correct in the table above — but any
future artifact that cites broker sizing from that yaml comment must use the
corrected `kafka.m5.large` value and note the comment itself is stale. Filing a
doc-fix for that comment is outside this ticket's scope (it lives in
`src/`, not `docs/evidence/`) and is not attempted here to keep this PR
evidence-only.

## What would close this ticket for real

1. A fresh AWS SSO session on a host with live access (human login, per this
   session's hard constraint that SSO is dead here) to fill the 9 AWS/k8s/DB
   rows with real command output.
2. A source/image/config digest pinned **at the moment of freeze**, not a
   7-day-old reference — meaning the freeze must happen at the same time as (or
   immediately after) a fresh candidate build, not asynchronously from it.
3. A plan-governor pass resolving §ac3 (bind `OMN-15123` to whatever row the old
   "§3 B3 unverifiable by construction" content became, or rule it subsumed).
4. Only once 1–3 land: a complete instance superseding this one, with
   `freeze_signature` filled on *that* commit, followed by a comment/link on
   OMN-14736 (AC #4).

## Related

- `docs/runbooks/managed-staging-canary-postgres-provisioning.md` — B12 landing table + §3.4 readback
- `docs/runbooks/managed-staging-canary-teardown-rollback.md` — teardown / abort / rollback ownership (B13)
- `src/omnibase_infra/topics/managed_staging_canary_catalog.py` + `..._namespace.yaml` — B7 `onex.mstg1.` catalog, epoch, zero-collision readback (OMN-14727)
- `docs/runbooks/managed-staging-proof-kit/fields.yaml` — the field manifest (seam) this packet fills
