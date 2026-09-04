# Candidate-in-isolation compatibility proof (template)

**Ticket:** OMN-15124 · **Plan row:** rolling plan §3 B5 · **Parent epic:** OMN-14724
**Field manifest (the seam):** [`tests/ci/fixtures/managed_staging_proof_kit/fields.yaml`](fields.yaml)
**Seam test:** `tests/ci/test_managed_staging_proof_kit_seam.py`

> **This is a template. It executes nothing and authorizes nothing.** The isolation
> run itself is a live mutation of an **isolation lane only** — zero staging mutation,
> zero prod mutation — and is HELD FOR OPERATOR like every other live step in this
> lane.

## What this proves (and what it deliberately does not)

It proves the **candidate image** speaks MSK and RDS correctly *before* it is pointed
at the staging lane: real IAM/TLS signer, an observed token refresh, explicit topic
bootstrap against a broker with auto-create OFF, `verify-full` to RDS, config coming
from the typed authority, and — the load-bearing half — **negative controls** showing
the failure paths actually fail. It is *not* the cutover (OMN-14933) and *not* the
canary run (OMN-14736/B11).

## Why the negative controls are the point

Three of the rows below are negative controls
(`negative_control_out_of_catalog`, `no_raw_endpoint_fallback`, and the
`dashboard_zero_authority` readback). A green positive path with a silent plaintext
or auto-create fallback is a **false pass**: the candidate would "work" against the
wrong surface. Each negative control must show the *observed refusal* — pasted error
text, not "verified".

## How to use it

1. Copy to `docs/evidence/OMN-15124/<UTC-date>-candidate-isolation-proof.md`.
2. Name the `isolation_lane` **before** the run, in the committed copy.
3. Run each evidence source; paste verbatim output (or link `probes/<field>.txt`).
4. `candidate_image_digest` must equal `image_digest` in the OMN-15123 freeze packet.
   If it does not, this proof does not cover the candidate being promoted.


## Fields

Every row is required. `Value` is filled at run time from `Evidence source`;
an empty or prose-only value cell means the packet is not complete.

| Field | What it is | Value (paste verbatim readback) | Evidence source (command / path) |
|---|---|---|---|
| `isolation_lane` | Named isolation lane/host (zero staging + zero prod mutation) | _(unfilled)_ | `hostname && kubectl config current-context  # record both; the lane must be named in the packet before the run` |
| `candidate_image_digest` | Candidate image digest under test (must equal one_tenant_contract_freeze.image_digest) | _(unfilled)_ | `kubectl -n <ns> get pod <pod> -o jsonpath='{.status.containerStatuses[0].imageID}'` |
| `msk_iam_signer` | MSK IAM/TLS signer actually used (not plaintext, not a fallback) | _(unfilled)_ | `uv run python -c 'from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env as f; print({k: (type(v).__name__ if k=="sasl_oauth_token_provider" else v) for k,v in f().items()})'  # src/omnibase_infra/event_bus/kafka_auth.py:105` |
| `token_refresh_cycle` | At least one observed IAM token-refresh cycle (>=2 token mints, same session) | _(unfilled)_ | `kubectl -n <ns> logs <pod> --since=<soak window> \| grep -iE 'token\|refresh\|expiry'  # two distinct mint timestamps in one uninterrupted session` |
| `auto_create_off` | Broker auto-create is OFF | _(unfilled)_ | `aws kafka describe-configuration-revision --arn <config-arn> --revision <n> --query ServerProperties --output text \| base64 -d \| grep auto.create.topics.enable` |
| `explicit_topic_bootstrap` | Explicit topic bootstrap succeeded (every catalog topic created deliberately) | _(unfilled)_ | `kafka-topics --bootstrap-server <b> --command-config <iam.properties> --list \| grep '^onex.mstg1.'  # compare set-equal against one_tenant_contract_freeze.topic_catalog` |
| `negative_control_out_of_catalog` | NEGATIVE CONTROL: out-of-catalog topic/group name is denied | _(unfilled)_ | `kafka-topics --bootstrap-server <b> --command-config <iam.properties> --create --topic notonex.mstg1.denied.v1  # MUST fail with AccessDenied/TopicAuthorizationException; paste the error verbatim` |
| `broker_group_perms` | Broker + consumer-group permissions scoped to the catalog patterns | _(unfilled)_ | `aws iam get-role-policy --role-name <node-role> --policy-name <policy> --query 'PolicyDocument.Statement[?contains(Action, `kafka-cluster:*`)]'  # patterns must be onex.* / omninode.*` |
| `rds_verify_full` | RDS connection proven sslmode=verify-full | _(unfilled)_ | `psql "$CANARY_DSN" -Atc "select ssl, version, cipher from pg_stat_ssl join pg_stat_activity using (pid) where pid = pg_backend_pid()"  # plus the DSN readback showing sslmode=verify-full and a pinned root CA` |
| `typed_config_authority` | Typed config authority (config comes from the typed model, not ad-hoc env reads) | _(unfilled)_ | `uv run python -c 'from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env; print(build_aiokafka_auth_kwargs_from_env.__module__)'  # plus scripts/check_required_env_vars.py output for the candidate's env contract` |
| `no_raw_endpoint_fallback` | NEGATIVE CONTROL: no raw-endpoint / plaintext fallback path was exercised | _(unfilled)_ | `unset the IAM env, start the candidate, and show it FAILS CLOSED rather than falling back; static half: scripts/check_no_cloud_bus_wrapper.sh and grep -rn 'PLAINTEXT' src/omnibase_infra/event_bus/` |
| `dashboard_zero_authority` | Dashboard holds zero broker/RDS authority in the candidate config | _(unfilled)_ | `kubectl -n <ns> get cm,secret -o yaml \| grep -nE 'omnidash'  # must show no broker bootstrap or RDS DSN granted to any dashboard surface` |
| `plan_row_binding` | Rolling plan §3 B5 bound to OMN-15124 | _(unfilled)_ | `git -C $OMNI_HOME/omni_home log -p -- docs/plans/ROLLING_SEVEN_DAY_PLAN.md \| grep -n 'OMN-15124'` |

## Related

- `docs/runbooks/managed-staging-canary-postgres-provisioning.md` — B12 landing table + §3.4 readback
- `docs/runbooks/managed-staging-canary-teardown-rollback.md` — teardown / abort / rollback ownership (B13)
- `src/omnibase_infra/topics/managed_staging_canary_catalog.py` + `..._namespace.yaml` — B7 `onex.mstg1.` catalog, epoch, zero-collision readback (OMN-14727)
- `scripts/proof/e2e_cloud_workflow_harness.py` — OMN-10858 end-to-end proof harness (`--live` defaults OFF)
- `omni_home:docs/plans/2026-07-17-managed-staging-verified-state-and-task-split.md` — lane B task split
