# OMN-15124 — candidate-in-isolation compatibility proof: round-4 evidence (still PARTIAL)

**Ticket:** OMN-15124 · **Parent epic:** OMN-14724 · **Status:** PARTIAL, NOT a completed packet
**Field manifest (the seam):** [`docs/runbooks/managed-staging-proof-kit/fields.yaml`](../../runbooks/managed-staging-proof-kit/fields.yaml)
**Packet template (unfilled, unmodified by this doc):** [`docs/runbooks/managed-staging-candidate-isolation-compatibility-proof.md`](../../runbooks/managed-staging-candidate-isolation-compatibility-proof.md)
**Captured:** 2026-08-05 (UTC), host `Stickybeatz-Studio.local` (`.200`), repo `omnibase_infra` at `origin/dev` HEAD `d1c5927cb7`.
**Supersedes:** does not edit `docs/evidence/OMN-15124/2026-08-03-candidate-isolation-static-evidence-partial.md` — this is a new dated instance adding what AWS SSO recovery unlocks.

## Why this is still PARTIAL, not a filled packet

AWS SSO recovered 2026-08-04T15:47:47Z and is confirmed live this session. That
unlocks **control-plane, read-only AWS API calls** this instance adds below. It does
**not** unlock the ticket's core requirement: all 4 real ACs (AC1–AC4) need a **live
candidate process actually running in a named isolation lane**, pointed at real
MSK/RDS, with negative controls exercised against it. No isolation lane exists —
there is no k8s/kubectl access from this host (direct `:6443` times out, same finding
as the OMN-15123 round-4 packet), so no candidate pod is running anywhere this
session can observe. **0/5 ACs remain unchecked. Do not read this document as
progress against any AC checkbox.**

## What changed vs. the 2026-08-03 instance

The 2026-08-03 instance filled 2/12 fields (both pure static code-path checks, zero
AWS dependency). This instance re-runs those 2 unchanged (values do not depend on
AWS/network state) and adds **2 new fields answerable via live, read-only AWS
control-plane API now that SSO works** — neither requires a running candidate or a
named isolation lane, so neither closes an AC, but both are real signal a future live
run will consume:

| Field | 2026-08-03 | 2026-08-05 round-4 | Why |
|---|---|---|---|
| `typed_config_authority` | FILLED (static) | **FILLED (re-run, unchanged)** | Pure code-path check, no AWS dependency |
| `no_raw_endpoint_fallback` (static half) | FILLED (static) | **FILLED (re-run, unchanged)** | Pure code-path check, no AWS dependency |
| `auto_create_off` | BLOCKED (no AWS creds) | **FILLED — `auto.create.topics.enable=false`** | `aws kafka describe-configuration-revision`, live, read-only |
| *(new finding, not a manifest field)* broker network reachability | not probed | **MSK IAM broker port `9098` TCP-reachable from this host; RDS `5432` is NOT** | See "New finding" below — flags a possible narrower isolation-lane option, does not itself satisfy any field |
| `isolation_lane`, `candidate_image_digest`, `msk_iam_signer` (live half), `token_refresh_cycle`, `explicit_topic_bootstrap`, `negative_control_out_of_catalog`, `broker_group_perms`, `rds_verify_full`, `dashboard_zero_authority` | BLOCKED | **still BLOCKED** | No isolation lane / no k8s access / require the candidate's own scoped IAM identity, not this session's admin SSO role — see per-field reasons below |
| `plan_row_binding` | BLOCKED (structural) | **RESOLVED** | 2026-08-04 plan-governor reconcile, same fix as OMN-15123 (`ROLLING_WORK_LEDGER.md:12253`) |

## What was run (live, read-only AWS control-plane; zero mutation, zero isolation-lane execution)

### Field: `auto_create_off`

> Broker auto-create is OFF

```console
$ aws kafka describe-configuration-revision \
    --arn arn:aws:kafka:us-east-1:272493677981:configuration/omninode-dev-msk-config/930d30cc-105c-4c3e-ab2a-aa6cfb0a5b0b-14 \
    --revision 1 --region us-east-1 --query ServerProperties --output text | base64 -d
auto.create.topics.enable=false
default.replication.factor=2
delete.topic.enable=true
log.retention.hours=168
min.insync.replicas=1
num.partitions=3
```

Full output: `probes-round4/msk_configuration_server_properties.txt`. This is
revision **1** of the config, which is **not** proof it is the revision bound to the
live cluster — the round-4 evidence CodeRabbit review (2026-08-08) correctly flagged
that the field was FILLED off a configuration revision without confirming that
revision is the one the active cluster actually runs.

**Cluster-binding readback, added to close that gap** (2026-08-09, same AWS SSO
session, live/read-only):

```console
$ aws kafka describe-cluster --region us-east-1 \
    --cluster-arn arn:aws:kafka:us-east-1:272493677981:cluster/omninode-dev-msk/88ad72bc-f70c-4549-93bc-c392b965f424-14
ClusterArn:            arn:aws:kafka:us-east-1:272493677981:cluster/omninode-dev-msk/88ad72bc-f70c-4549-93bc-c392b965f424-14
State:                 ACTIVE
PublicAccess.Type:     DISABLED
CurrentBrokerSoftwareInfo.ConfigurationArn:      arn:aws:kafka:us-east-1:272493677981:configuration/omninode-dev-msk-config/930d30cc-105c-4c3e-ab2a-aa6cfb0a5b0b-14
CurrentBrokerSoftwareInfo.ConfigurationRevision: 2
```

Full output: `probes-round4/msk_describe_cluster.json`. **The active revision is 2,
not 1** — the original probe queried a stale revision. Revision 2 was re-read
directly to confirm the field value still holds on the actually-bound config:

```console
$ aws kafka describe-configuration-revision \
    --arn arn:aws:kafka:us-east-1:272493677981:configuration/omninode-dev-msk-config/930d30cc-105c-4c3e-ab2a-aa6cfb0a5b0b-14 \
    --revision 2 --region us-east-1 --query ServerProperties --output text | base64 -d
auto.create.topics.enable=false
default.replication.factor=2
delete.topic.enable=true
log.retention.hours=168
min.insync.replicas=1
num.partitions=3
```

Full output: `probes-round4/msk_configuration_revision2_server_properties.txt` —
identical values to revision 1, `auto.create.topics.enable=false` unchanged.
**FILLED — live, now bound to the cluster's active configuration (ArN + revision 2),
cluster state ACTIVE, `PublicAccess.Type: DISABLED` confirmed.**

### New finding (not a manifest field): broker network reachability from this host

```console
$ nc -zv -w5 b-1.omninodedevmsk.7ozyd3.c14.kafka.us-east-1.amazonaws.com 9098
Connection to b-1.omninodedevmsk.7ozyd3.c14.kafka.us-east-1.amazonaws.com port 9098 [tcp/*] succeeded!

$ nc -zv -w5 omninode-dev-postgres.cqjkkokeaqd2.us-east-1.rds.amazonaws.com 5432
nc: connectx to ... port 5432 (tcp) failed: Operation timed out
```

Full output: `probes-round4/nc_msk_broker_9098.txt`, `probes-round4/nc_rds_5432.txt`.
**Genuinely surprising** given the cluster's `PublicAccess.Type: DISABLED` and this
host being outside the VPC — a bare TCP handshake to the MSK IAM listener (9098)
succeeds, while the RDS Postgres port (5432, also VPC-private) times out as expected.
**This is recorded as a finding, not exploited.** It does not by itself prove
anything about the candidate: (1) TCP reachability is not the same as a completed
SASL/IAM handshake — no Kafka protocol exchange was attempted; (2) even if a real
handshake succeeded, testing with this session's own admin SSO identity
(`AROAT64PDRWOQRVJSLB6P:jonah-iam-main`) would prove that identity's permissions, not
the candidate's scoped node-role permissions — a negative-control test run under an
admin identity is not a real negative control (admin can do everything an
under-scoped role cannot, so a "denied" result would be impossible to obtain and a
"succeeded" result would prove nothing about the intended restriction). No topic
create/list/describe was attempted against the live broker in this lane — that would
require either the candidate's own IAM identity or would-be canary-scoped
credentials, neither available here, and any topic mutation is out of this lane's
authorized scope (artifacts/proofs only, no canary execution). **Handoff:** whoever
stands up the isolation lane should know this host may not need an SSM tunnel for
MSK specifically (unlike RDS and k8s, both confirmed still blocked) — worth a quick
re-check before assuming full VPC access is required for the MSK-facing half of this
proof.

### Fields carried forward unchanged (re-run, same result)

`typed_config_authority`:
```console
$ env -u PYTHONPATH uv run python -c "
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env
print(build_aiokafka_auth_kwargs_from_env.__module__)
"
omnibase_infra.event_bus.kafka_auth
```
Full output: `probes-round4/typed_config_authority.txt`. Unchanged from 2026-08-03 —
this module boundary has not moved.

`no_raw_endpoint_fallback` (static half):
```console
$ bash scripts/check_no_cloud_bus_wrapper.sh
(no output, exit 0)
$ grep -n "PLAINTEXT" src/omnibase_infra/event_bus/kafka_auth.py
107:    if config.security_protocol == "PLAINTEXT":
```
Full output: `probes-round4/check_no_cloud_bus_wrapper.txt`,
`probes-round4/grep_plaintext_kafka_auth.txt`. Same reading as the 2026-08-03
instance: this is a legitimate config-declared branch (local dev), not a silent
fallback — the live half (unset IAM env on a running candidate, observe fail-closed)
remains unexercised.

## Gap statement — everything still blocking AC1–AC4, and why

| Field | AC | Blocked on |
|---|---|---|
| `isolation_lane` | AC1 | No k8s/kubectl access from this host (`:6443` direct times out — same finding as OMN-15123 round-4). No named isolation host exists to record. |
| `candidate_image_digest` | AC1 | Requires `kubectl get pod ... imageID` against a live isolation lane — none exists. (Cross-reference only, not a fill: the round-4 target is `sha256:b829c56f...`, per the OMN-15123 round-4 packet — but this field means the digest **actually running in the isolation pod**, which this session cannot observe.) |
| `msk_iam_signer` (live half) | AC1 | Requires a running candidate process actually invoking the signer against the real broker; only the static module-resolution half is answerable without a live process. |
| `token_refresh_cycle` | AC1 | Requires a soaked live session with ≥2 observed token mints — no live process exists. |
| `explicit_topic_bootstrap` | AC2 | Requires `kafka-topics --list` with IAM auth configured — no `kafka-topics` CLI + AWS MSK IAM auth JAAS setup exists on this host, and even if it did, listing under this session's admin identity would not represent the candidate's actual bootstrap path. |
| `negative_control_out_of_catalog` | AC2 | Same identity problem as above, doubled: a negative control run under an admin identity cannot fail the way the candidate's scoped identity would — see "New finding" above. |
| `broker_group_perms` | (supporting AC2) | Requires `aws iam get-role-policy --role-name <node-role>` — the candidate's actual node/pod IAM role name is not known from this host (would need IRSA/service-account annotation lookup via `kubectl`, which is blocked). `aws iam list-roles` in this account shows only cluster/infra-management roles (`omninode-k3s-*-node-role`, ASG lifecycle, GitHub Actions) — none obviously named as the runtime pod's scoped role, and guessing would risk citing the wrong policy as evidence. |
| `rds_verify_full` | AC3 | RDS port 5432 confirmed unreachable from this host (`probes-round4/nc_rds_5432.txt`, timeout) — VPC-private, `PubliclyAccessible: false`. |
| `dashboard_zero_authority` | AC4 | Requires `kubectl get cm,secret` against a live isolation/candidate namespace — blocked, same as above. |
| `plan_row_binding` | AC5 | **RESOLVED**, not blocked — see below. |

### `plan_row_binding` (AC5) — resolved, not blocked

The 2026-08-03 instance found this structurally blocked (the plan's `§3 B5` row no
longer exists in that form). The 2026-08-04T03:58:31Z plan-governor reconcile
(`ROLLING_WORK_LEDGER.md:12253`, plan commit `7fb2e9853`) fixed this for both
OMN-15123 and OMN-15124 together: the plan now cites `OMN-15124` by ID directly at
two live anchors — `§0-CHAIN` link 6 and `§2` "Fastest readiness order" item 6 — both
independently re-confirmed by this instance's own grep against the live plan
(`OMN-15124` present at lines 38 and 93 of `docs/plans/ROLLING_SEVEN_DAY_PLAN.md`).
**AC5 is the one AC of the five that this instance closes cleanly.**

## Adjacent, out-of-scope context (unchanged from 2026-08-03, still relevant)

OMN-15639 tracks a separate, group-authorization-level MSK IAM defect
(`GroupAuthorizationFailedError` on `runtime-local-*` consumer groups) adjacent to
but explicitly out of this ticket's connection-level scope. Even a fully-satisfied
OMN-15124 does not clear OMN-15639.

## Operator / Daniyal handoff

1. Same SSM-tunnel gap as OMN-15123: k8s access from this host needs a tunnel, not
   direct `:6443`. Once available, `isolation_lane`, `candidate_image_digest`,
   `dashboard_zero_authority`, and `broker_group_perms` (via the pod's IRSA
   annotation) all become reachable.
2. **Worth a quick check before assuming MSK also needs the tunnel**: the MSK IAM
   listener (`:9098`) is TCP-reachable from this host right now, unlike RDS.
   Standing up a real candidate process (or even a scoped test client with the
   node's actual IAM identity, not admin SSO) directly from a host like this one
   might close `msk_iam_signer` (live half), `token_refresh_cycle`,
   `explicit_topic_bootstrap`, and `negative_control_out_of_catalog` without a full
   k8s isolation lane — worth scoping as a cheaper path than standing up the whole
   isolation lane, but requires the candidate's actual scoped credentials, which
   this session does not have and should not substitute admin credentials for (a
   negative control run under admin creds is not a real negative control).
3. `rds_verify_full` genuinely needs in-VPC or SSM-tunneled access; no shortcut
   found this session.

## Bottom line

0/5 ACs satisfied (AC5/`plan_row_binding` is resolved, but that resolution landed via
a separate plan-governor commit, not this PR, so it is recorded here rather than
claimed as this PR's own closure). This instance adds 2 new live, read-only AWS
findings (`auto_create_off` filled; MSK broker network reachability noted as a
scoping lead) on top of the 2 static fields carried forward unchanged. The
candidate-isolation manifest (`docs/runbooks/managed-staging-proof-kit/fields.yaml`,
`isolation_lane` … `dashboard_zero_authority`) declares 12 unique field IDs backing
AC1–AC4; `plan_row_binding` (AC5) is a separate, thirteenth field not counted in that
12. Of the 12: **3 FILLED** (`auto_create_off`, `typed_config_authority`,
`no_raw_endpoint_fallback` static half) and **9 BLOCKED** — every BLOCKED field is
enumerated with its reason in the gap table above (`isolation_lane`,
`candidate_image_digest`, `msk_iam_signer` live half, `token_refresh_cycle`,
`explicit_topic_bootstrap`, `negative_control_out_of_catalog`, `broker_group_perms`,
`rds_verify_full`, `dashboard_zero_authority`). A prior revision of this document
undercounted this as "8 of 12" — corrected here after CodeRabbit flagged the
mismatch (2026-08-08); each next session should derive the count from the gap
table's row count, not restate a number by hand.
