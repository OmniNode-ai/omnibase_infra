# Managed-staging one-tenant contract — FREEZE PACKET (round-4 candidate, still partial)

**Ticket:** OMN-15123 · **Parent epic:** OMN-14724
**Field manifest (the seam):** [`docs/runbooks/managed-staging-proof-kit/fields.yaml`](../../runbooks/managed-staging-proof-kit/fields.yaml)
**Template this was copied from:** [`docs/runbooks/managed-staging-one-tenant-contract-freeze.md`](../../runbooks/managed-staging-one-tenant-contract-freeze.md)
**Seam test:** `tests/ci/test_managed_staging_proof_kit_seam.py`
**Instance date:** 2026-08-05 (UTC) · **Author lane:** fable-epsilon-0805-eve (build subagent, Sonnet 5)
**Supersedes:** `docs/evidence/OMN-15123/2026-08-03-one-tenant-contract-freeze.md` (does not edit that file — see "What changed" below)

> ## THIS IS STILL NOT A FREEZE. Read before reading the table.
>
> AWS SSO recovered 2026-08-04T15:47:47Z and is confirmed live this session
> (`aws sts get-caller-identity` succeeds — see `probes-round4/aws_sts_get_caller_identity.json`).
> That clears 6 of the 2026-08-03 instance's 11 BLOCKED fields with real, current
> command output. It does **not** clear the k8s/DB-dependent fields (direct `:6443`
> access from this host times out — see `probes-round4/kubectl_attempt.txt` — and
> `docs/runbooks/managed-staging-canary-postgres-provisioning.md` §3.4 is itself
> still `HELD`, i.e. the canary DB has never been provisioned). **`freeze_signature`
> is deliberately left unfilled**, for a reason beyond field-completeness this time:
> `omninode_infra#818` (migrate re-pin) and `#819` (runtime+composition re-pin) are
> both **OPEN, not merged** as of this instance — the round-4 tuple
> (`candidate-a853500-20260805212844`) is a *build artifact*, not yet the *pinned*
> candidate. AC #2 requires the digest to equal "the live candidate at freeze
> time"; freezing against an unmerged pin risks freezing against a tuple Codex
> never lands. **Re-freeze condition, stated explicitly per the prior instance's
> own request:** this instance itself goes stale the moment #818/#819 merge (a new,
> confirmed-pinned tuple) or Codex requests a differing digest.

## AC-by-AC disposition

| # | Acceptance criterion | Status | Why |
|---|---|---|---|
| 1 | Committed, immutable contract artifact enumerating all 19 fields | **NOT MET** | 14 of 19 fields now carry real values (up from 8/19); 5 remain BLOCKED (k8s/DB-dependent). Not immutable in the AC's sense until complete. |
| 2 | Digest fields read back equal to the live candidate at freeze time, command output attached | **PARTIAL** | `source_digest`/`image_digest`/`config`-adjacent (migrate) digests are filled with fresh, ECR-verified readback (below) — but the candidate is not yet the *pinned* one (#818/#819 open), so "at freeze time" cannot be asserted as final. `config_digest` (rendered k8s configmap hash) remains BLOCKED — no k8s access. |
| 3 | Rolling plan citation | **RESOLVED** (carried forward) | Per the 2026-08-04T03:58:31Z plan-governor reconcile (`ROLLING_WORK_LEDGER.md:12253`, plan commit `7fb2e9853`): OMN-15123 is bound at two live anchors — §0-CHAIN link 6 and §2 "Fastest readiness order" item 6 — both citing OMN-15123 by ID directly. No further action needed; confirmed present in the live plan by this instance's own grep (see `plan_row_binding` row). |
| 4 | OMN-14736 (B11 canary) references this frozen tuple as its input | **NOT MET** | The tuple isn't frozen (AC #1 unmet) — deferred to the complete instance, unchanged from 2026-08-03. |

## What changed vs. the 2026-08-03 instance

| Field | 2026-08-03 | 2026-08-05 round-4 | Why it changed |
|---|---|---|---|
| `aws_account` | BLOCKED | **`272493677981`** | AWS SSO now live |
| `aws_region` | BLOCKED | **`us-east-1`** | AWS SSO now live |
| `msk_cluster_arn` | BLOCKED | **`arn:aws:kafka:us-east-1:272493677981:cluster/omninode-dev-msk/88ad72bc-f70c-4549-93bc-c392b965f424-14`** | AWS SSO now live |
| `rds_instance_identifier` | BLOCKED | **`omninode-dev-postgres` @ `omninode-dev-postgres.cqjkkokeaqd2.us-east-1.rds.amazonaws.com:5432`** | AWS SSO now live |
| `source_digest` | BLOCKED (7d-stale ref only) | **`a853500ab3c74620acdba34b418f6bd087081153`** | Fresh candidate build, ECR-tag-verified |
| `image_digest` | BLOCKED (7d-stale ref only) | **`sha256:b829c56f547340363be347391275cc847b1f465c18fa275836dd098989f7ecb7`** (runtime) + **`sha256:64f519fe7907329e01bce38693f77f34a033ee9c33b8c91e22044248a3a7ec18`** (migrate) | Fresh candidate build, ECR-verified, both independently re-read (not trusted from the ground-truth prompt) |
| `k8s_namespace`, `gateway_endpoint`, `synthetic_tenant_id`, `config_digest`, `omnidash_exclusion` | BLOCKED | **still BLOCKED** | Direct `:6443` from this host times out (`kubectl --kubeconfig ~/.kube/omninode-dev get ns` → exit 124 after 6s, see `probes-round4/kubectl_attempt.txt`); ledger precedent + this session's own probe both confirm an SSM-tunnel route is required, not attempted here (out of this lane's scope — flagged as the operator/Daniyal handoff below) |
| `plan_row_binding` | BLOCKED (structural) | **RESOLVED** | 2026-08-04 plan-governor pass (see AC#3 above) |
| `freeze_signature` | deliberately unfilled | **still deliberately unfilled** | #818/#819 open, not merged — see the box above |
| `topic_catalog`, `zero_collision_readback`, `msk_epoch`, `group_start_reset_policy`, `rollback_authority`, `zero_prod_diff` | FILLED (offline) | **unchanged, carried forward verbatim** | These are offline, in-repo facts; SSO recovery does not affect them and re-running would not change the value |

## Fields

| Field | What it is | Value | Evidence source | Status |
|---|---|---|---|---|
| `aws_account` | AWS account | **`272493677981`** | `aws sts get-caller-identity` — `probes-round4/aws_sts_get_caller_identity.json` | **FILLED — live, this instance.** |
| `aws_region` | AWS region | **`us-east-1`** | `aws configure get region` — `probes-round4/aws_region.txt` | **FILLED — live, this instance.** |
| `k8s_namespace` | Kubernetes namespace (single canary namespace) | _BLOCKED_ | `kubectl get ns <ns> -o jsonpath=...` | **BLOCKED — direct `:6443` from this host times out** (`probes-round4/kubectl_attempt.txt`, `timeout 6 kubectl --kubeconfig ~/.kube/omninode-dev get ns` → exit 124). SSM-tunnel route required, not established in this lane. |
| `msk_cluster_arn` | MSK cluster ARN | **`arn:aws:kafka:us-east-1:272493677981:cluster/omninode-dev-msk/88ad72bc-f70c-4549-93bc-c392b965f424-14`** | `aws kafka list-clusters-v2 ...` — `probes-round4/msk_cluster_arn.txt` | **FILLED — live, this instance.** |
| `rds_instance_identifier` | RDS instance identifier | **`omninode-dev-postgres` (endpoint `omninode-dev-postgres.cqjkkokeaqd2.us-east-1.rds.amazonaws.com:5432`, `PubliclyAccessible: false`)** | `aws rds describe-db-instances ...` — `probes-round4/rds_instance.txt` | **FILLED — live, this instance.** |
| `gateway_endpoint` | The one gateway | _BLOCKED_ | `kubectl -n <ns> get svc,ingress ...` | **BLOCKED — no live k8s access** (same as above). |
| `synthetic_tenant_id` | The one synthetic tenant (UUID) | _BLOCKED_ | `psql "$CANARY_DSN" ...` | **BLOCKED — no DB access.** Additionally: `docs/runbooks/managed-staging-canary-postgres-provisioning.md` §3.4 ("Readback — prove the landing table exists") is itself still marked `HELD` in the live runbook — the canary DB has never been provisioned, independent of this host's reachability. Confirmed `nc -zv -w5 omninode-dev-postgres.cqjkkokeaqd2.us-east-1.rds.amazonaws.com 5432` → `Operation timed out` (VPC-private, `PubliclyAccessible: false`). |
| `source_digest` | Candidate source digest | **`a853500ab3c74620acdba34b418f6bd087081153`** | ECR image tag `candidate-a853500-20260805212844` / migrate tag `a853500ab3c74620acdba34b418f6bd087081153` both encode this commit SHA — `probes-round4/ecr_runtime_image.json`, `probes-round4/ecr_migrate_image.json` | **FILLED — live, this instance.** Full 40-char git SHA not independently re-derived from a `git rev-parse` against a cloned candidate tree at this exact ref (no such clone in this lane); taken from the ECR tag, which is the same provenance chain the round-3/round-4 pin PRs (`omninode_infra#814`, `#818`, `#819`) use. |
| `image_digest` | Candidate image digest | **runtime: `sha256:b829c56f547340363be347391275cc847b1f465c18fa275836dd098989f7ecb7`** (tag `candidate-a853500-20260805212844`) **· migrate: `sha256:64f519fe7907329e01bce38693f77f34a033ee9c33b8c91e22044248a3a7ec18`** (tag `a853500ab3c74620acdba34b418f6bd087081153`) | `aws ecr describe-images --repository-name omninode-runtime --image-ids imageTag=candidate-a853500-20260805212844` / `--repository-name omnibase-infra-migrate --image-ids imageTag=a853500ab3c74620acdba34b418f6bd087081153` — `probes-round4/ecr_runtime_image.json`, `probes-round4/ecr_migrate_image.json` | **FILLED — live, this instance, independently ECR-re-verified** (not trusted from the dispatch prompt's assertion — both digests matched exactly). **Caveat per the box above: not yet the pinned candidate** (#818/#819 open). |
| `config_digest` | Rendered runtime config digest | _BLOCKED_ | `kubectl -n <ns> get cm <configmap> ... \| shasum -a 256` | **BLOCKED — no live k8s access.** |
| `topic_catalog` | Approved `onex.mstg1.` topic/group catalog | **164 topics / 56 groups** | carried forward from 2026-08-03 instance, `probes/topic_catalog.txt` | **FILLED (carried forward, unchanged) — offline, in-repo; re-running would not change the value.** |
| `zero_collision_readback` | Zero-collision readback | **`is_clean: True` against an EMPTY snapshot — offline caveat unchanged** | carried forward from 2026-08-03 instance, `probes/zero_collision_readback_offline.txt` | **FILLED (carried forward), OFFLINE-ONLY.** Live Phase-3 readback against the actual cluster's live topics/groups remains BLOCKED (no VPC/broker access from this host for a `kafka-topics --list`-class read — see the OMN-15124 round-4 packet for the one broker-reachability finding this session did make). |
| `msk_epoch` | Unique MSK epoch | **`mstg1`** | carried forward, `..._namespace.yaml -> epoch` | **FILLED (carried forward, unchanged).** |
| `group_start_reset_policy` | Signed consumer-group start/reset policy | **`earliest` (policy value only — still NOT signed)** | carried forward, `..._namespace.yaml -> group_start_policy` | **PARTIALLY FILLED (carried forward, unchanged).** Operator signature line still absent. |
| `rollback_authority` | Named rollback authority | **See ownership table citation** | carried forward, `docs/runbooks/managed-staging-canary-teardown-rollback.md` §0 | **FILLED (carried forward, unchanged).** |
| `zero_prod_diff` | Zero-prod-diff assertion | **PASS (re-run against this file)** | `grep -nE 'omnibase-infra-prod\|:28085\|:28086' <this packet>` | **FILLED — re-run against this instance's own text.** Matches only this row's own evidence-source cell (the grep pattern quoting the strings it checks for); no actual prod resource named anywhere in the tuple's data rows. |
| `omnidash_exclusion` | Omnidash exclusion | _BLOCKED_ | `kubectl -n <ns> get deploy -o jsonpath=...` | **BLOCKED — no live k8s access.** |
| `freeze_signature` | Freeze signature | **See note — not a valid freeze signature yet** | `git log -1 --format='%H %aI %an' -- <path>` | **DELIBERATELY NOT FILLED.** #818/#819 (the pin PRs for this exact tuple) are OPEN — see the box at the top. Fill only on the instance minted after they merge (or a later, differing digest if Codex requests changes), with every remaining BLOCKED row cleared by a live k8s/DB readback. |
| `plan_row_binding` | Rolling plan bound to OMN-15123 | **RESOLVED — §0-CHAIN link 6 + §2 item 6, both cite `OMN-15123` by ID** | `git -C $OMNI_HOME/omni_home log -p -- docs/plans/ROLLING_SEVEN_DAY_PLAN.md \| grep -n 'OMN-15123'` — live grep against `ROLLING_SEVEN_DAY_PLAN.md` this instance shows `OMN-15123` present at line 38 (§0-CHAIN) and line 93 (§2 "Fastest readiness order" item 6) | **FILLED — resolved by the 2026-08-04T03:58:31Z plan-governor reconcile** (`ROLLING_WORK_LEDGER.md:12253`), re-confirmed live by this instance rather than trusted from the ledger claim alone. |

## Live probes committed alongside this packet

- `probes-round4/aws_sts_get_caller_identity.json`
- `probes-round4/aws_region.txt`
- `probes-round4/msk_cluster_arn.txt`
- `probes-round4/rds_instance.txt`
- `probes-round4/ecr_runtime_image.json`
- `probes-round4/ecr_migrate_image.json`
- `probes-round4/kubectl_attempt.txt` — proof of the k8s-access BLOCKED claim (timeout, not an assertion)

Carried forward, unchanged, from the 2026-08-03 instance: `docs/evidence/OMN-15123/probes/topic_catalog.txt`, `probes/zero_collision_readback_offline.txt`, `probes/zero_prod_diff_grep.txt`.

## Operator / Daniyal handoff — what closes the remaining 5 BLOCKED fields

1. **k8s access needs the SSM-tunnel route, not direct `:6443`.** This host's kubeconfig (`~/.kube/omninode-dev`) times out against the API server directly; per ledger precedent (referenced in the dispatching ground-truth, not independently re-derived here — no SSM-tunnel runbook was found under `docs/runbooks/` in this repo during this lane's search) an SSM port-forward session is the documented route. Standing that tunnel up is out of this lane's scope (infra/network setup, not an artifact-producing step) — hand off to whoever owns the SSM-tunnel procedure.
2. Once k8s is reachable: `k8s_namespace`, `gateway_endpoint`, `config_digest`, `omnidash_exclusion` are all single `kubectl` reads away.
3. `synthetic_tenant_id` additionally needs the canary Postgres surface actually provisioned — `docs/runbooks/managed-staging-canary-postgres-provisioning.md` §3.1–§3.4 are all still `HELD`. That is a separate, larger action (creating the canary DB/role) than a read-only field-fill and is explicitly an operator-gated step per that runbook's own markings — not attempted here.
4. Once #818/#819 merge (or land with a differing digest), mint the next dated instance, re-verify `source_digest`/`image_digest` still match the merged pin, and fill `freeze_signature` only then.

## Related

- `docs/evidence/OMN-15123/2026-08-03-one-tenant-contract-freeze.md` — prior partial instance (superseded, not deleted)
- `docs/runbooks/managed-staging-canary-postgres-provisioning.md` — B12 landing table + §3.4 readback (still HELD)
- `docs/runbooks/managed-staging-canary-teardown-rollback.md` — teardown / abort / rollback ownership (B13)
- `src/omnibase_infra/topics/managed_staging_canary_catalog.py` + `..._namespace.yaml` — B7 `onex.mstg1.` catalog, epoch, zero-collision readback (OMN-14727)
- `docs/runbooks/managed-staging-proof-kit/fields.yaml` — the field manifest (seam) this packet fills
- `omninode_infra#818` (migrate re-pin, OPEN) / `omninode_infra#819` (runtime+composition re-pin, OPEN, self-hash `156bade7...`) — the round-4 pin PRs this tuple targets
