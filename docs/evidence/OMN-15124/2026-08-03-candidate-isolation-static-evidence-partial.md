# OMN-15124 — candidate-in-isolation compatibility proof: PARTIAL static evidence

**Ticket:** OMN-15124 · **Parent epic:** OMN-14724 · **Status:** PARTIAL, NOT a completed packet
**Field manifest (the seam):** [`docs/runbooks/managed-staging-proof-kit/fields.yaml`](../../runbooks/managed-staging-proof-kit/fields.yaml)
**Packet template (unfilled, unmodified by this doc):** [`docs/runbooks/managed-staging-candidate-isolation-compatibility-proof.md`](../../runbooks/managed-staging-candidate-isolation-compatibility-proof.md)
**Captured:** 2026-08-03T18:09:04Z, host `Stickybeatz-Studio.local` (`.200`), repo `omnibase_infra` at `origin/dev` HEAD `3860bec762`.

## Why this is PARTIAL, not a filled packet

The ticket's 5 acceptance criteria all require a **live isolation-environment run**
against real MSK and RDS with negative controls (AC1–AC4) plus a rolling-plan diff
citation for a plan row that does not exist in the current plan structure (AC5). AWS
SSO is dead on every host available to this session (human login pending) — every
live AWS/k8s/psql step is BLOCKED. None of the 5 ACs can be checked off by this
session. **Do not read this document as progress against any AC checkbox.**

Of the manifest's 12 `candidate_isolation_compatibility` fields, exactly **2** are
answerable with zero live AWS/network dependency — pure static code-path checks run
on this host today. They are recorded below as labeled evidence. The remaining 10 are
enumerated in the gap statement and require a live isolation lane this session does
not have.

## What was run (static, no AWS, no network)

### Field: `typed_config_authority`

> Typed config authority (config comes from the typed model, not ad-hoc env reads)

```
$ cd omnibase_infra && env -u PYTHONPATH uv run python -c "
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env
print(build_aiokafka_auth_kwargs_from_env.__module__)
"
omnibase_infra.event_bus.kafka_auth
```

Confirms the function the candidate must call for its Kafka auth kwargs resolves to
the typed `omnibase_infra.event_bus.kafka_auth` module, not an ad-hoc env-read call
site elsewhere. **This proves only that the typed entry point exists and is what a
correctly-wired caller would import** — it does not prove any live candidate process
actually calls it instead of a bypass, and it does not run
`scripts/check_required_env_vars.py` against a candidate's real env contract (that
script validates local `docker/docker-compose.infra.yml` + local env files, not an
isolation-lane candidate's live config — running it here would not add real signal
toward this field and was skipped as out of scope).

### Field: `no_raw_endpoint_fallback` — STATIC HALF ONLY

> NEGATIVE CONTROL: no raw-endpoint / plaintext fallback path was exercised

The manifest evidence source splits this field into a live half ("unset the IAM env,
start the candidate, and show it FAILS CLOSED") and a static half. Only the static
half was run:

```
$ cd omnibase_infra && bash scripts/check_no_cloud_bus_wrapper.sh
(no output, exit 0)

$ grep -rn "PLAINTEXT" src/omnibase_infra/event_bus/
src/omnibase_infra/event_bus/kafka_auth.py:107:    if config.security_protocol == "PLAINTEXT":
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:113:            Default: "PLAINTEXT"
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:114:            Options: "PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:119:            Requires: security_protocol must be SASL_PLAINTEXT or SASL_SSL
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:382:        default="PLAINTEXT",
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:385:            "Valid values: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL"
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:387:        pattern=r"^(PLAINTEXT|SSL|SASL_PLAINTEXT|SASL_SSL)$",
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:394:            "Requires security_protocol to be SASL_PLAINTEXT or SASL_SSL. "
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:562:        - If security_protocol is SASL_PLAINTEXT or SASL_SSL, sasl_mechanism must be set
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:563:        - If sasl_mechanism is set, security_protocol must be SASL_PLAINTEXT or SASL_SSL
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:579:            self.security_protocol in ("SASL_PLAINTEXT", "SASL_SSL")
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:590:            if self.security_protocol not in ("SASL_PLAINTEXT", "SASL_SSL"):
src/omnibase_infra/event_bus/models/config/model_kafka_event_bus_config.py:593:                    f"'SASL_PLAINTEXT' or 'SASL_SSL', got {self.security_protocol!r}",
```

Reading `kafka_auth.py:105-107`:

```python
def build_aiokafka_auth_kwargs(config: ModelKafkaEventBusConfig) -> dict[str, object]:
    """Build auth/TLS kwargs for aiokafka clients from runtime Kafka config."""
    if config.security_protocol == "PLAINTEXT":
        return {}
```

**This is not a fail-closed guard and must not be read as one.** `PLAINTEXT` is a
valid, config-declared value of `security_protocol` (used for local dev), and the
function honors it by returning empty auth kwargs — a legitimate branch, not a silent
fallback bypassing a failed IAM path. `check_no_cloud_bus_wrapper.sh` passed clean
(no disallowed direct cloud-bus construction found), and no hidden alternate
construction path exists in `src/omnibase_infra/event_bus/` outside this typed
module. **What this does NOT prove:** that a candidate configured for
`AWS_MSK_IAM` actually fails closed rather than silently degrading to `PLAINTEXT`
when its IAM env is unset — that is exactly the live half of this negative control
("unset the IAM env, start the candidate, show FAILS CLOSED"), which requires a
running candidate process in an isolation lane and was not exercised.

## Gap statement — everything else this ticket's ACs require, and why it is blocked

All 5 ACs remain **unchecked**. Per field:

| Field | AC | Blocked on |
|---|---|---|
| `isolation_lane` | AC1 | No isolation host/k8s context available this session — AWS SSO dead, human login pending. |
| `candidate_image_digest` | AC1 | Requires `kubectl get pod ... imageID` against a live isolation lane. |
| `msk_iam_signer` (live half) | AC1 | Requires a running candidate process actually invoking the MSK IAM signer against a real broker. |
| `token_refresh_cycle` | AC1 | Requires a soaked live session with ≥2 observed token mints — no live process exists. |
| `auto_create_off` | AC2 | Requires `aws kafka describe-configuration-revision` — AWS API, no credentials. |
| `explicit_topic_bootstrap` | AC2 | Requires a live broker + `kafka-topics --bootstrap-server` with IAM creds. |
| `negative_control_out_of_catalog` | AC2 | Requires a live broker to observe the actual denial. |
| `broker_group_perms` | (supporting AC2) | Requires `aws iam get-role-policy` — AWS API, no credentials. |
| `rds_verify_full` | AC3 | Requires a live `psql` connection to a real RDS instance. |
| `dashboard_zero_authority` | AC4 | Requires `kubectl get cm,secret` against a live isolation/candidate namespace. |
| `plan_row_binding` | AC5 | **Not AWS-blocked — structurally absent.** `docs/plans/ROLLING_SEVEN_DAY_PLAN.md` has no `§3 B5` row: live section headings are `0-AIM`, `0-CHAIN`, `0. Operating rules`, `1. Current ground state`, `2. Ranked seven-day work queue`, `3. Decisions`, `4. Parked...`, `5. Source trail`, `6. Revision log`, and `git log -p -- docs/plans/ROLLING_SEVEN_DAY_PLAN.md \| grep -n 'OMN-15124'` returns no plan-diff hit — only the single line-38 chain-list mention (`## 0-CHAIN`) and the §3b decisions table have no B5/OMN-15124 row. Same class of gap as OMN-15123 AC #3: the manifest's `plan_row: "rolling plan §3 B5"` cites a row that does not exist in the live plan document. This is a doc-authoring/reconciliation gap, not an AWS dependency, and is out of this session's scope to resolve unilaterally (it requires a plan-governance decision on how/whether to bind the tag). |

## Adjacent, out-of-scope context (recorded per ticket comment, not actioned here)

Per the 2026-08-01 ticket comment (Jonah Gray): OMN-15639 tracks a separate,
group-authorization-level MSK IAM defect (`GroupAuthorizationFailedError` on
`runtime-local-*` consumer groups) adjacent to but explicitly **out of** this
ticket's connection-level scope. Even a fully-satisfied OMN-15124 does not clear
OMN-15639 — recorded here for continuity, not addressed by this PR.

## Bottom line

0/5 ACs satisfied. This PR adds labeled PARTIAL static evidence for 2 of 12 manifest
fields and a field-by-field gap statement for the remaining 10, so the next session
with live AWS/isolation-lane access has a starting inventory instead of an empty
template. **No AC checkbox in the live ticket is flipped by this PR** — that would be
a false-Done claim: only a live isolation run (AC1–AC4) and an operator plan-binding
decision (AC5) close this ticket.
