# Managed-staging one-tenant contract — FREEZE PACKET (template)

**Ticket:** OMN-15123 · **Plan row:** rolling plan §3 B3 · **Parent epic:** OMN-14724
**Field manifest (the seam):** [`docs/runbooks/managed-staging-proof-kit/fields.yaml`](managed-staging-proof-kit/fields.yaml)
**Seam test:** `tests/ci/test_managed_staging_proof_kit_seam.py`

> **This is a template. It executes nothing and authorizes nothing.** Filling it in
> does not authorize any AWS / MSK / RDS / k8s mutation. Every live readback below is
> run by the operator (or by an agent with an explicit per-step GO), per
> `omni_home:docs/plans/2026-07-17-managed-staging-agent-driven-execution-plan.md`.

## How to use it

1. Copy this file to `docs/evidence/OMN-15123/<UTC-date>-one-tenant-contract-freeze.md`.
2. Run each row's **evidence source** and paste the *verbatim output* into the value
   cell (or into a linked `probes/<field>.txt` next to the copy). Prose in a value
   cell is not evidence.
3. Commit. **The commit that lands the filled packet IS the freeze event** — the
   `freeze_signature` row records that commit.
4. After freeze, the tuple is immutable. A change does not edit this artifact; it
   **bumps the epoch** (`mstg1` → `mstg2`) and mints a new freeze packet, per the
   epoch rules in `docs/runbooks/managed-staging-canary-teardown-rollback.md` §4.2.

## Acceptance gate (OMN-15123)

- Every row below has a pasted readback, not a claim.
- `image_digest` / `config_digest` read back **equal to the live candidate at freeze
  time** — this is the ticket's explicit "readback command output attached, not
  asserted prose" criterion.
- `zero_prod_diff` returns no matches and `omnidash_exclusion` shows no omnidash
  workload.
- The rolling plan §3 B3 row cites OMN-15123 (`plan_row_binding`), replacing its
  "unverifiable by construction" tag.
- OMN-14736 (B11 canary execution) links this packet as its input surface.


## Fields

Every row is required. `Value` is filled at run time from `Evidence source`;
an empty or prose-only value cell means the packet is not complete.

| Field | What it is | Value (paste verbatim readback) | Evidence source (command / path) |
|---|---|---|---|
| `aws_account` | AWS account | _(unfilled)_ | `aws sts get-caller-identity --query Account --output text` |
| `aws_region` | AWS region | _(unfilled)_ | `aws configure get region  # cross-check against the cluster ARN in msk_cluster_arn` |
| `k8s_namespace` | Kubernetes namespace (single canary namespace) | _(unfilled)_ | `kubectl get ns <ns> -o jsonpath='{.metadata.name}{"\t"}{.metadata.uid}'` |
| `msk_cluster_arn` | MSK cluster ARN | _(unfilled)_ | `aws kafka list-clusters-v2 --query 'ClusterInfoList[?ClusterName==`omninode-dev-msk`].ClusterArn' --output text` |
| `rds_instance_identifier` | RDS instance identifier | _(unfilled)_ | `aws rds describe-db-instances --db-instance-identifier omninode-dev-postgres --query 'DBInstances[0].[DBInstanceIdentifier,Endpoint.Address]' --output text` |
| `gateway_endpoint` | The one gateway (exactly one ingress path into the canary) | _(unfilled)_ | `kubectl -n <ns> get svc,ingress -o jsonpath='{range .items[*]}{.kind}/{.metadata.name}{"\n"}{end}'  # must resolve to exactly one externally reachable gateway` |
| `synthetic_tenant_id` | The one synthetic tenant (UUID) | _(unfilled)_ | `psql "$CANARY_DSN" -Atc "select distinct tenant_id from delivery_replay_canary_projection"  # see docs/runbooks/managed-staging-canary-postgres-provisioning.md §3.4` |
| `source_digest` | Candidate source digest (git commit the image was built from) | _(unfilled)_ | `git -C <candidate repo> rev-parse HEAD  # cross-check the image label: docker image inspect <image> --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'` |
| `image_digest` | Candidate image digest (immutable, by digest not tag) | _(unfilled)_ | `aws ecr describe-images --repository-name <repo> --image-ids imageTag=<tag> --query 'imageDetails[0].imageDigest' --output text` |
| `config_digest` | Rendered runtime config digest | _(unfilled)_ | `kubectl -n <ns> get cm <configmap> -o json \| jq -S '.data' \| shasum -a 256` |
| `topic_catalog` | Approved onex.mstg1. topic/group catalog (full generated list) | _(unfilled)_ | `uv run python -c 'from omnibase_infra.topics.managed_staging_canary_catalog import build_canary_catalog_from_candidate as b; c=b(); print("\n".join(sorted(c.topics)))'  # generator: src/omnibase_infra/topics/managed_staging_canary_catalog.py (OMN-14727)` |
| `zero_collision_readback` | Zero-collision readback against live topics + consumer groups | _(unfilled)_ | `uv run python -c 'from omnibase_infra.topics.managed_staging_canary_catalog import verify_zero_collision'  # run against the live topic/group snapshot; see the module's "Zero-collision readback" docstring` |
| `msk_epoch` | Unique MSK epoch (namespace segment; bump to re-run) | _(unfilled)_ | `src/omnibase_infra/topics/managed_staging_canary_catalog_namespace.yaml -> epoch` |
| `group_start_reset_policy` | Signed consumer-group start / reset policy | _(unfilled)_ | `src/omnibase_infra/topics/managed_staging_canary_catalog_namespace.yaml -> group_start_policy (+ the operator signature line in the filled packet)` |
| `rollback_authority` | Named rollback authority (who may execute the revert) | _(unfilled)_ | `docs/runbooks/managed-staging-canary-teardown-rollback.md §0 ownership table + §4 rollback path` |
| `zero_prod_diff` | Zero-prod-diff assertion (no prod resource in the tuple) | _(unfilled)_ | `grep -nE 'omnibase-infra-prod\|:28085\|:28086' <this packet>  # must return no matches; prod lanes are named in CLAUDE.md's .201 lane table` |
| `omnidash_exclusion` | Omnidash exclusion (omnidash carries no canary authority) | _(unfilled)_ | `kubectl -n <ns> get deploy -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'  # must contain no omnidash workload` |
| `freeze_signature` | Freeze signature (operator + UTC timestamp the tuple became immutable) | _(unfilled)_ | `git log -1 --format='%H %aI %an' -- <path of the filled packet>  # the commit that landed the packet IS the freeze event` |
| `plan_row_binding` | Rolling plan §3 B3 bound to OMN-15123 | _(unfilled)_ | `git -C $OMNI_HOME/omni_home log -p -- docs/plans/ROLLING_SEVEN_DAY_PLAN.md \| grep -n 'OMN-15123'  # cite the plan diff that replaced the 'unverifiable by construction' tag` |

## Related

- `docs/runbooks/managed-staging-canary-postgres-provisioning.md` — B12 landing table + §3.4 readback
- `docs/runbooks/managed-staging-canary-teardown-rollback.md` — teardown / abort / rollback ownership (B13)
- `src/omnibase_infra/topics/managed_staging_canary_catalog.py` + `..._namespace.yaml` — B7 `onex.mstg1.` catalog, epoch, zero-collision readback (OMN-14727)
- `scripts/proof/e2e_cloud_workflow_harness.py` — OMN-10858 end-to-end proof harness (`--live` defaults OFF)
- `omni_home:docs/plans/2026-07-17-managed-staging-verified-state-and-task-split.md` — lane B task split
