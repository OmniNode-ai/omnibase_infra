#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
#
# smoke_delegation.sh -- submit ONE delegation through a lab lane's real
# onex-api as that lane's own tenant, wait for it to terminate, and read back
# every row the business proof is graded on. OMN-17530 (epic), OMN-17537,
# OMN-15583.
#
# ONE SCRIPT, TWO TARGETS. `--target compose` drives the .201 compose dev lane
# (the lab of record, operator ruling 2026-09-08); `--target k8s` drives the
# k3s onex-lab lane in namespace onex-dev. Everything that constitutes the PROOF
# -- the submission, the per-seam SQL, the reader comparison, the RLS posture
# read, the writer-log scan -- is written ONCE and shared. Only four adapters
# differ, and they are the first four functions below.
#
# That shape is deliberate. The alternative -- a k8s copy and a compose copy --
# diverges the moment one lane grows a seam, and the divergence is invisible:
# both scripts keep passing, and the seam only one of them checks is the one
# that breaks in front of a customer.
#
# WHAT MAKES THIS A PROOF RATHER THAN A PING. It exercises the whole chain the
# way a customer does and then checks each seam SEPARATELY, so a break is
# attributed rather than merely observed:
#
#   POST /v1/workflows            the gateway ingress, with a real credential
#     -> onex.cmd...delegation-request.v1   the command envelope on the bus
#     -> the runtime                        execution
#     -> delegation_events                  what the PROJECTION WRITER stamped
#     -> GET /v1/tenants/me/delegations     what the READER returns
#
# The last two are the seam OMN-15583 was about: the writer stamps the tenant's
# canonical UUID and the reader resolved the slug, so the reader answered from a
# partition the writer had stopped writing to and a stale page read exactly like
# a real answer. Comparing "the row is in the table" against "the reader returns
# THIS correlation id" is what makes that class visible; either check alone
# reports green through it.
#
# CREDENTIAL HANDLING. The API key is read into a variable and sent in a header.
# It is never echoed, never passed on a command line (where `ps` would show it)
# and never written to a file this script creates. On the compose target the
# mint step writes it inside the container with mode 0600; only the tenant id,
# the correlation id, row counts and sha256-12 fingerprints are ever printed.
#
# LANE-SCOPED BY CONSTRUCTION. No DSN argument, no host argument, no cloud
# credential. The compose target names one compose project and the k8s target
# one namespace; neither can be aimed at a cloud database or a cloud tenant.
set -euo pipefail

TARGET="${SMOKE_TARGET:-}"
ATTRIBUTION="tenant"   # tenant | house
TERMINAL_TIMEOUT="${TERMINAL_TIMEOUT:-180}"
PROJECTION_SETTLE="${PROJECTION_SETTLE:-20}"

usage() {
  cat >&2 <<'USAGE'
usage: smoke_delegation.sh --target compose|k8s [--attribution tenant|house]

  --target compose   the .201 compose dev lane (project $COMPOSE_PROJECT, default omnibase-infra)
  --target k8s       the k3s onex-lab lane (namespace $NAMESPACE, default onex-dev)
  --attribution      tenant (default) submits as the lane's own minted tenant;
                     house submits the platform-ladder run, which is what a
                     lane with no per-tenant credential can still execute.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --attribution) ATTRIBUTION="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
case "$TARGET" in
  compose|k8s) ;;
  *) echo "FATAL: --target must be compose or k8s (got '${TARGET:-<unset>}')" >&2; usage; exit 2 ;;
esac
case "$ATTRIBUTION" in
  tenant|house) ;;
  *) echo "FATAL: --attribution must be tenant or house" >&2; exit 2 ;;
esac

# ===========================================================================
# THE FOUR ADAPTERS. Everything below this section is target-independent.
# ===========================================================================
COMPOSE_PROJECT="${COMPOSE_PROJECT:-omnibase-infra}"
NAMESPACE="${NAMESPACE:-onex-dev}"
KUBECTL="${KUBECTL:-kubectl}"
CREDENTIAL_SECRET="${CREDENTIAL_SECRET:-onex-lab-tenant-credential}"
ANALYTICS_DB="${ANALYTICS_DB:-omnidash_analytics}"
# The mint writes here, inside the onex-api container, on the compose target.
TENANT_STATE_FILE="${TENANT_STATE_FILE:-/var/lib/onex-lab-tenant/credential.env}"
LAB_TENANT_SLUG="${LAB_TENANT_SLUG:-onex-lab-house}"
LAB_API_KEY_NAME="${LAB_API_KEY_NAME:-onex-lab-verification}"

compose_container() {
  # By compose label, not by container_name: a container_name is a convention
  # and a label is what the project actually stamps, so this cannot silently
  # pick up a same-named container from another lane on the same host.
  docker ps --filter "label=com.docker.compose.project=${COMPOSE_PROJECT}" \
            --filter "label=com.docker.compose.service=$1" \
            --filter "status=running" --format '{{.Names}}' | head -1
}

if [ "$TARGET" = compose ]; then
  API_UNIT="$(compose_container onex-api)"
  DB_UNIT="$(compose_container postgres)"
  WRITER_UNIT="$(compose_container projection-delegation-writer)"
  [ -n "$API_UNIT" ] || { echo "FATAL: no running onex-api container in compose project ${COMPOSE_PROJECT}" >&2; exit 1; }
  [ -n "$DB_UNIT" ]  || { echo "FATAL: no running postgres container in compose project ${COMPOSE_PROJECT}" >&2; exit 1; }

  # api_python_env NAME... -- run the python program on STDIN inside the API
  # container, forwarding the NAMED variables from this process's environment.
  # `docker exec -e NAME` (no `=value`) copies the value across without it ever
  # appearing in argv, which is what keeps the API key out of `ps`.
  api_python_env() {
    local args=() n
    for n in "$@"; do args+=(-e "$n"); done
    docker exec -i "${args[@]}" "$API_UNIT" python3 -
  }
  db_query() { docker exec -i "$DB_UNIT" psql -U postgres -d "$ANALYTICS_DB" -tAc "$1" 2>&1; }
  writer_log() {
    [ -n "$WRITER_UNIT" ] || { echo "  FATAL: no running delegation writer container" >&2; return 0; }
    docker logs "$WRITER_UNIT" --tail=400 2>&1
  }
else
  pod_for() {
    ${KUBECTL} get pods -n "${NAMESPACE}" \
      -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' \
      | grep "^${1}-" | head -1
  }
  API_UNIT="$(pod_for onex-api)"
  DB_UNIT="$(pod_for "${DB_DEPLOYMENT:-onex-lab-postgres}")"
  WRITER_UNIT="$(pod_for omnimarket-projection-delegation-writer)"
  [ -n "$API_UNIT" ] || { echo "FATAL: no running onex-api pod in ${NAMESPACE}" >&2; exit 1; }
  [ -n "$DB_UNIT" ]  || { echo "FATAL: no running database pod in ${NAMESPACE}" >&2; exit 1; }

  api_python_env() {
    local args=() n
    for n in "$@"; do eval "args+=(${n}=\"\${${n}}\")"; done
    ${KUBECTL} exec -i -n "${NAMESPACE}" "${API_UNIT}" -c onex-api -- env "${args[@]}" python3 -
  }
  db_query() { ${KUBECTL} exec -i -n "${NAMESPACE}" "${DB_UNIT}" -- psql -U postgres -d "$ANALYTICS_DB" -tAc "$1" 2>&1; }
  writer_log() {
    [ -n "$WRITER_UNIT" ] || { echo "  FATAL: no running delegation writer pod" >&2; return 0; }
    ${KUBECTL} logs -n "${NAMESPACE}" "${WRITER_UNIT}" --tail=400 2>&1
  }
fi

echo "== target: ${TARGET} (api unit ${API_UNIT}, db unit ${DB_UNIT})"

# ===========================================================================
# CREDENTIAL: resolve the lane's own tenant, minting it on first use.
# ===========================================================================
if [ "$TARGET" = k8s ]; then
  TENANT_ID="$(${KUBECTL} get secret "${CREDENTIAL_SECRET}" -n "${NAMESPACE}" -o jsonpath='{.data.TENANT_ID}' | base64 -d)"
  TENANT_SLUG="$(${KUBECTL} get secret "${CREDENTIAL_SECRET}" -n "${NAMESPACE}" -o jsonpath='{.data.TENANT_SLUG}' | base64 -d)"
  [ -n "$TENANT_ID" ] || { echo "FATAL: ${CREDENTIAL_SECRET} carries no TENANT_ID -- run the minting Job first" >&2; exit 1; }
  ONEX_API_KEY="$(${KUBECTL} get secret "${CREDENTIAL_SECRET}" -n "${NAMESPACE}" -o jsonpath='{.data.ONEX_API_KEY}' | base64 -d)"
else
  # The compose lane has no Secret store, so the mint runs IN the API container
  # and writes its own state file there with mode 0600. It calls the same two
  # real paths the k3s minting Job calls, for the same reasons: POST
  # /v1/tenants/bootstrap (the only endpoint that can create a FIRST tenant --
  # POST /v1/tenants requires an existing tenant credential) and, in-process,
  # auth_api_keys.create_api_key (the same function routers/api_keys.py calls,
  # so the prefix, the entropy and the sha256 key_hash are the production ones
  # and not a hand-rolled row).
  echo "== resolving the lane's own tenant (mint on first use) =="
  export SMOKE_TENANT_STATE_FILE="$TENANT_STATE_FILE"
  export SMOKE_TENANT_SLUG="$LAB_TENANT_SLUG"
  export SMOKE_KEY_NAME="$LAB_API_KEY_NAME"
  MINT_OUT="$(api_python_env SMOKE_TENANT_STATE_FILE SMOKE_TENANT_SLUG SMOKE_KEY_NAME <<'PY'
import hashlib, json, os, sys, time, urllib.error, urllib.request, uuid

STATE = os.environ["SMOKE_TENANT_STATE_FILE"]
SLUG = os.environ["SMOKE_TENANT_SLUG"]
KEY_NAME = os.environ["SMOKE_KEY_NAME"]
BASE = "http://127.0.0.1:8000"
ADMIN = os.environ.get("TENANT_BOOTSTRAP_ADMIN_SECRET", "")

# Idempotent: a lane whose credential file is already there does not mint a
# second tenant and does not mint a second key. A rerun that silently rotated
# the key would be indistinguishable, at the next 401, from the key having been
# wrong all along.
if os.path.exists(STATE):
    values = dict(
        line.split("=", 1)
        for line in open(STATE).read().splitlines()
        if "=" in line and not line.startswith("#")
    )
    print(json.dumps({
        "minted": False,
        "tenant_id": values.get("TENANT_ID", ""),
        "tenant_slug": values.get("TENANT_SLUG", ""),
        "key_sha256_12": hashlib.sha256(values.get("ONEX_API_KEY", "").encode()).hexdigest()[:12],
        "key_len": len(values.get("ONEX_API_KEY", "")),
    }))
    raise SystemExit(0)

if not ADMIN:
    sys.exit("FATAL: TENANT_BOOTSTRAP_ADMIN_SECRET is not set in the onex-api container -- "
             "POST /v1/tenants/bootstrap answers 503 without it. Run "
             "scripts/runtime_build/render_dev_lane_tenant_path_env.sh and refresh the lane.")


def call(path, *, data=None, method="GET", headers=None):
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read() or b"{}")


for attempt in range(1, 31):
    try:
        if call("/health")[0] == 200:
            break
    except Exception as exc:  # noqa: BLE001 -- report and retry
        print(f"waiting for onex-api ({attempt}/30): {exc}", file=sys.stderr)
    time.sleep(2)
else:
    sys.exit("FATAL: onex-api never answered /health from inside its own container")

tenant = None
try:
    _, tenant = call(
        "/v1/tenants/bootstrap",
        data=json.dumps({"tenant_slug": SLUG, "name": "onex-dev-lane"}).encode(),
        method="POST",
        headers={"X-Admin-Secret": ADMIN},
    )
except urllib.error.HTTPError as exc:
    # 409 is the idempotent case. Anything else is a real failure: a 503 here
    # means the admin secret is unset, which is a lane defect and not a
    # condition to work around.
    if exc.code != 409:
        sys.exit(f"FATAL: bootstrap returned {exc.code}: {exc.read()[:400]!r}")

sys.path.insert(0, "/app")
from auth_api_keys import create_api_key  # noqa: E402
from db.psycopg_repository import PsycopgRepository  # noqa: E402

if tenant is None:
    row = PsycopgRepository().fetch_one(
        "SELECT tenant_id FROM tenants WHERE tenant_slug = %s", (SLUG,)
    )
    if not row:
        sys.exit("FATAL: bootstrap said 409 but no tenant row exists")
    tenant_id = str(row["tenant_id"] if isinstance(row, dict) else row[0])
else:
    tenant_id = str(tenant["tenant_id"])

minted = create_api_key(uuid.UUID(tenant_id), KEY_NAME)
plaintext = minted.plaintext_key

# The plaintext is available exactly once. It goes to a 0600 file inside this
# container and never to stdout.
os.makedirs(os.path.dirname(STATE), exist_ok=True)
fd = os.open(STATE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as fh:
    fh.write(f"TENANT_ID={tenant_id}\nTENANT_SLUG={SLUG}\nONEX_API_KEY={plaintext}\n")

print(json.dumps({
    "minted": True,
    "tenant_id": tenant_id,
    "tenant_slug": SLUG,
    "api_key_id": str(minted.id),
    "key_sha256_12": hashlib.sha256(plaintext.encode()).hexdigest()[:12],
    "key_len": len(plaintext),
}))
PY
)"
  echo "${MINT_OUT}"
  TENANT_ID="$(printf '%s' "$MINT_OUT" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read().strip().splitlines()[-1])["tenant_id"])')"
  TENANT_SLUG="$LAB_TENANT_SLUG"
  # The key stays inside the container. Every call that needs it is made from
  # in there, reading the same 0600 file.
  ONEX_API_KEY=""
fi
echo "== lane tenant: ${TENANT_ID} (${TENANT_SLUG})"

# ===========================================================================
# SUBMIT. One delegation, waited to a terminal status.
# ===========================================================================
echo "== submitting one delegation-inference workflow (attribution=${ATTRIBUTION}) =="
export SMOKE_ATTRIBUTION="$ATTRIBUTION"
export SMOKE_TERMINAL_TIMEOUT="$TERMINAL_TIMEOUT"
export SMOKE_TENANT_STATE_FILE="$TENANT_STATE_FILE"
export ONEX_API_KEY
SUBMIT_OUT="$(api_python_env ONEX_API_KEY SMOKE_ATTRIBUTION SMOKE_TERMINAL_TIMEOUT SMOKE_TENANT_STATE_FILE <<'PY'
import json, os, sys, time, urllib.error, urllib.request

KEY = os.environ.get("ONEX_API_KEY") or ""
if not KEY:
    state = os.environ["SMOKE_TENANT_STATE_FILE"]
    KEY = dict(
        line.split("=", 1) for line in open(state).read().splitlines() if "=" in line
    )["ONEX_API_KEY"]

BASE = "http://127.0.0.1:8000"
BODY = json.dumps({
    "workflow_type": "delegation-inference",
    "payload": {"prompt": "Reply with the single word: lab.", "task_type": "test", "max_tokens": 16},
}).encode()


def call(path, *, data=None, method="GET"):
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json", "X-API-Key": KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {"raw": exc.read()[:600].decode("utf-8", "replace")}


status, body = call("/v1/workflows", data=BODY, method="POST")
if status != 202:
    print(json.dumps({"submit_status": status, "submit_body": body}))
    sys.exit(1)

workflow_id = body.get("workflow_id")
# Wait for a terminal status. A gateway that accepted the envelope but whose
# runtime never consumed it stays non-terminal, which is a DIFFERENT finding
# from a refused submission and is reported as such rather than as a timeout.
deadline = time.time() + int(os.environ.get("SMOKE_TERMINAL_TIMEOUT", "180"))
terminal = {"completed", "failed", "failed_publish", "cancelled", "timeout", "error"}
final = None
detail = None
while time.time() < deadline:
    st, payload = call(f"/v1/workflows/{workflow_id}/status")
    final = payload.get("status") if st == 200 else f"status_http_{st}"
    detail = payload.get("error_code") or payload.get("failure_reason") or detail
    if final in terminal:
        break
    time.sleep(3)

print(json.dumps({
    "submit_status": status,
    "workflow_id": workflow_id,
    "correlation_id": body.get("correlation_id"),
    "envelope_id": body.get("envelope_id"),
    "final_status": final,
    "final_detail": detail,
    "reached_terminal": final in terminal,
}))
PY
)" || true
echo "${SUBMIT_OUT}"

CORRELATION_ID="$(printf '%s' "${SUBMIT_OUT}" | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read().strip().splitlines()[-1]).get("correlation_id") or "")
except Exception: print("")')"
[ -n "${CORRELATION_ID}" ] || { echo "FATAL: no correlation id -- submission did not reach 202" >&2; exit 1; }

# The projection writer consumes asynchronously off the bus.
sleep "${PROJECTION_SETTLE}"

# ===========================================================================
# THE SEAMS. Each is read separately so a break is attributed, not observed.
# ===========================================================================
echo "== SEAM 1: delegation_events row for correlation ${CORRELATION_ID} =="
db_query "SELECT correlation_id, tenant_id, pg_typeof(tenant_id)::text, timestamp,
                 delegated_to, model_name, quality_gate_passed, cost_savings_usd
          FROM delegation_events WHERE correlation_id::text = '${CORRELATION_ID}'"

echo "== SEAM 2: is delegation_events.tenant_id the UUID or the slug? (OMN-15583) =="
db_query "SELECT CASE
            WHEN tenant_id = '${TENANT_ID}'   THEN 'uuid   (writer and reader agree)'
            WHEN tenant_id = '${TENANT_SLUG}' THEN 'slug   (OMN-15583 shape -- the reader will not find it)'
            ELSE 'neither: ' || tenant_id
          END
          FROM delegation_events WHERE correlation_id::text = '${CORRELATION_ID}'"

echo "== SEAM 3: the model actually used (a scored run needs a real model, never 'none') =="
db_query "SELECT model_name, delegated_to FROM delegation_events
          WHERE correlation_id::text = '${CORRELATION_ID}'"

echo "== SEAM 4: RLS posture AND the identity the writer connects as =="
# relrowsecurity/relforcerowsecurity alone is half the fact. Postgres exempts a
# table's OWNER and any BYPASSRLS role from row-level security unconditionally,
# FORCE included -- so FORCE on a table whose only writer is the owner is inert,
# and "no RLS errors" read from such a lane is a false clean rather than
# evidence. Both halves are printed together for that reason.
db_query "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class
          WHERE relname IN ('delegation_events','delegation_judge_verdict_events')"
db_query "SELECT rolname, rolsuper, rolbypassrls, rolcanlogin FROM pg_roles
          WHERE rolname IN ('role_omnidash','role_omninode','tenant_projection_writer','postgres')
          ORDER BY rolname"
db_query "SELECT c.relname, pg_get_userbyid(c.relowner) AS owner FROM pg_class c
          WHERE c.relname IN ('delegation_events','delegation_judge_verdict_events')"

echo "== SEAM 5: delegation_events rows for this tenant, by tenant_id FORM =="
db_query "SELECT tenant_id, count(*) FROM delegation_events
          WHERE tenant_id IN ('${TENANT_ID}', '${TENANT_SLUG}') GROUP BY tenant_id"

echo "== SEAM 6: quality-gate verdict row, and its timestamp (OMN-15583 / omnimarket#2406) =="
# A delegation that terminalizes on a typed refusal never reaches a judge, so
# ZERO rows here is the CORRECT reading for such a run and must not be graded as
# a writer failure. The count is printed next to the terminal outcome above so
# the two are read together.
db_query "SELECT to_regclass('public.delegation_judge_verdict_events')"
db_query "SELECT count(*) FROM delegation_judge_verdict_events
          WHERE correlation_id::text = '${CORRELATION_ID}'"
db_query "SELECT correlation_id, tenant_id, timestamp IS NOT NULL AS timestamp_populated
          FROM delegation_judge_verdict_events
          WHERE correlation_id::text = '${CORRELATION_ID}'"

echo "== SEAM 7: reader -- GET /v1/tenants/me/delegations as the lane tenant =="
export SMOKE_CORRELATION_ID="$CORRELATION_ID"
api_python_env ONEX_API_KEY SMOKE_CORRELATION_ID SMOKE_TENANT_STATE_FILE <<'PY'
import json, os, urllib.error, urllib.request

KEY = os.environ.get("ONEX_API_KEY") or ""
if not KEY:
    state = os.environ["SMOKE_TENANT_STATE_FILE"]
    KEY = dict(line.split("=", 1) for line in open(state).read().splitlines() if "=" in line)["ONEX_API_KEY"]

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/tenants/me/delegations?limit=25",
    headers={"X-API-Key": KEY},
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        status, body = resp.status, json.loads(resp.read() or b"{}")
except urllib.error.HTTPError as exc:
    status, body = exc.code, {"raw": exc.read()[:600].decode("utf-8", "replace")}

wanted = os.environ["SMOKE_CORRELATION_ID"]
runs = body.get("delegations") or body.get("runs") or []
print(json.dumps({
    "reader_status": status,
    "reader_rows": len(runs) if isinstance(runs, list) else None,
    "reader_returns_correlated_row": any(wanted in json.dumps(r) for r in runs) if isinstance(runs, list) else False,
}))
PY

echo "== SEAM 8: delegation projection writer log, last errors =="
writer_log | grep -iE "error|23502|null value|violat|refus|permission denied" | tail -15 \
  || echo "  (no error lines in the last 400)"

echo "== smoke complete: target=${TARGET} attribution=${ATTRIBUTION} correlation=${CORRELATION_ID} =="
