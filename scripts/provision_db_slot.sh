#!/bin/sh
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# provision_db_slot.sh — per-slot databases AND per-slot principals (OMN-18892)
#
# Provisions an isolated database set for one ephemeral pre-PR verify slot on a
# SHARED Postgres server, and mints the slot its own login principals.
#
# WHY PER-SLOT PRINCIPALS AND NOT JUST PER-SLOT DATABASE NAMES
# -------------------------------------------------------------
# Roles in Postgres are CLUSTER-WIDE objects. Suffixing the database names while
# leaving the roles alone does not isolate a slot, and the two provisioning seams
# in this repository both reset the password of every role they manage on their
# existing-role branch, UNCONDITIONALLY:
#
#   docker/migrations/forward/000_create_multiple_databases.sh
#       create_role(), create_login_only_role()   -> ALTER ROLE ... PASSWORD
#   scripts/run-forward-migrations.sh
#       reassert_login_only_role_credential()     -> ALTER ROLE ... PASSWORD
#       reassert_service_role_database_access()   -> ALTER ROLE ... PASSWORD
#
# A second consumer of the shared server running either seam with its own
# credentials in the environment rewrites the DEV LANE's role passwords for the
# whole cluster. The dev lane's containers hold the old values in their
# environment and start failing authentication at their next reconnect. The
# alternative -- reusing the dev passwords -- is not isolation at all: the slot's
# credentials would BE the dev lane's, with full access to the dev databases.
#
# The initialiser is the seam the plan named. It is NOT the seam a slot reaches:
# Postgres runs it from /docker-entrypoint-initdb.d only when the data directory
# is empty, and a slot shares a warm volume. The seam on a slot's path is
# run-forward-migrations.sh, which the forward-migration one-shot runs on EVERY
# compose up. Both carry the hazard and both are fenced (see ONEX_DB_SLOT there).
#
# THE FENCE IS TWO INDEPENDENT CONTROLS, AND NEITHER REPLACES THE OTHER
# ---------------------------------------------------------------------
#   1. NAME fence     -- every object this tool names must carry the slot suffix.
#                        Bounds what can be addressed.
#   2. OWNERSHIP fence -- a PRE-EXISTING role is altered only when it is a member
#                        of this slot's own group role. Bounds what can be
#                        mutated among the names control 1 admits. Suffix
#                        matching alone is refutable by a role that merely ends
#                        in the same characters; group membership answers "did
#                        this tool create it", which is the question that
#                        matters.
#
# WHAT THIS TOOL DELIBERATELY DOES NOT DO
# ----------------------------------------
# It never runs with an empty slot token. The unsuffixed set is the dev lane's
# and is provisioned by NOT running this tool. An empty token is a mis-set
# variable, and treating it as "the dev path" is the exact accident this whole
# change exists to prevent, so it is a refusal with its own exit code.
#
# POSIX sh, no bashisms: this runs in postgres:16-alpine beside the other
# migration one-shots, which ships busybox ash and no bash.
#
# Ticket: OMN-18892. Parent epic: OMN-18888 (AC-4). Depends on: OMN-18890.

set -e
set -u

# ---------------------------------------------------------------------------
# Named refusal codes (OMN-18892)
# ---------------------------------------------------------------------------
# Every refusal exits with its own status. A caller -- and the test suite --
# reads the status rather than matching prose, so rewording a message can never
# turn a real refusal into a pass. Pinned by
# tests/unit/infra/test_db_slot_provisioner_omn18892.py.
EXIT_SLOT_TOKEN_MISSING=2
EXIT_SLOT_TOKEN_MALFORMED=3
EXIT_DERIVED_NAME_OUT_OF_FENCE=4
EXIT_ROLE_OUTSIDE_SLOT_GROUP=5
EXIT_REACHES_UNSUFFIXED_DATABASE=6
EXIT_CANNOT_REACH_OWN_DATABASE=7
EXIT_IDENTIFIER_TOO_LONG=8
EXIT_USAGE=9
EXIT_NOT_SUPERUSER=10
# OMN-19415. Pinned by tests/unit/infra/test_db_slot_connection_budget.py.
EXIT_SERVER_UNDERSIZED=11
EXIT_ROLE_WITHOUT_BUDGET=12

# Postgres truncates an identifier longer than 63 bytes SILENTLY. A truncated
# name can collide with another slot's, which is isolation failing with no error
# at all, so length is a refusal and never a truncation.
PG_MAX_IDENTIFIER_BYTES=63

slot_fence_refusal() {
    _code="$1"
    shift
    echo "[provision-db-slot] slot_fence_refusal: $*" >&2
    exit "$_code"
}

fail() {
    _code="$1"
    shift
    echo "[provision-db-slot] refused: $*" >&2
    exit "$_code"
}

# ---------------------------------------------------------------------------
# The managed set
# ---------------------------------------------------------------------------
# These three maps MIRROR docker/migrations/forward/000_create_multiple_databases.sh
# and are pinned equal to it by
# tests/unit/infra/test_db_slot_provisioner_omn18892.py. A database added there
# and not here is a slot that silently provisions a short set -- the runtime then
# fails at connect on the one database nobody noticed was missing, which reads as
# a branch defect rather than as a provisioning gap.
SERVICE_DB_MAP="omnibase_infra:role_omnibase
omniintelligence:role_omniintelligence
omniclaude:role_omniclaude
omnimemory:role_omnimemory
omninode_cloud:role_omninode
omnidash_analytics:role_omnidash"

INFRA_DATABASES="infisical_db
omniweb"

# The topology-governed login-only principals. They get a LOGIN credential and
# NOTHING else here, exactly as in the initialiser: their AUTHORIZATION is
# declared by the topology and issued by the topology-derived migrations. Running
# them through the grant helper would hand them CREATE on schema public, and a
# role that can OWN a table is exempt from that table's row-level security
# unconditionally -- which would make the forced RLS on the tenant tables inert
# on every slot, on the very lane the slot exists to prove things on.
LOGIN_ONLY_ROLES="omninode_runtime
tenant_projection_writer
chain_canary_reader"

# ---------------------------------------------------------------------------
# Connection budget (OMN-19415)
# ---------------------------------------------------------------------------
# A slot is a GUEST on the dev lane's Postgres server. Until OMN-19415 nothing
# bounded how many connections it could open there, and the server ran the
# stock max_connections of 100 (3 reserved for superusers). Measured on the lab
# host on 2026-09-24 with a read-only 2-second pg_stat_activity sampler:
#
#   * slot prepr-1 boot 5 held 63 connections at 16:27:12Z. role_omnibase_prepr1
#     peaked at 50 (slot runtime, runtime-worker, runtime-effects and
#     tenant-projection-writer at 14/14/14/9, most opened by asyncpg pool floors
#     and never used), omninode_runtime_prepr1 at 9, role_omnidash_prepr1 at 8,
#     role_omniintelligence_prepr1 at 3. The server sat at 100/100 from
#     16:27:06Z and the DEV runtime logged 1171 refusals in four minutes.
#   * the dev lane ALONE, with no slot running, reached 93 at 16:50:07Z, five
#     minutes after a dev runtime redeploy (16:45:08Z): the same four containers
#     at 18/13/13/13 as the `postgres` role, before idle pool members aged out
#     back to about 31.
#
# So a slot booting next to a fresh dev restart overflows a 100-connection
# server on either side's demand alone. The fix is two bounds that together make
# the dev lane's share structural rather than a matter of timing:
#
#   1. every slot principal carries an explicit CONNECTION LIMIT from this
#      table. A slot that wants more than its budget is refused INSIDE the slot
#      ("too many connections for role"), as the slot's own finding, and never
#      by starving the dev lane;
#   2. --apply refuses (EXIT_SERVER_UNDERSIZED) to provision a slot on a server
#      whose max_connections does not cover the dev lane's budget plus every
#      pool slot's full budget plus the server's own reservations. The declared
#      value lives on the dev lane's postgres in docker/docker-compose.dev-lane.yml
#      (not the base file, which other lanes inherit), and
#      tests/unit/infra/test_db_slot_connection_budget.py pins it against the
#      same arithmetic.
#
# Each limit is the measured slot peak or the dev-lane equivalent, whichever is
# higher, with headroom; a principal never seen connected gets a small floor
# rather than zero, because 0 locks it out and -1 is "unlimited", which is the
# defect. A base role missing from this table fails closed at scope derivation
# (EXIT_ROLE_WITHOUT_BUDGET): an unbudgeted principal is exactly the leak this
# table exists to close.
CONNECTION_BUDGET_MAP="role_omnibase:64
role_omniintelligence:6
role_omniclaude:4
role_omnimemory:4
role_omninode:6
role_omnidash:12
omninode_runtime:16
tenant_projection_writer:4
chain_canary_reader:2"

# The dev lane is not fenced by this tool and carries no per-role limit; this is
# the share of the server the capacity preflight RESERVES for it. The measured
# dev-only peak is 93 (16:50:07Z, post-redeploy transient); 110 leaves about 18%
# over it.
DEV_LANE_CONNECTION_BUDGET=110

# Must equal len(SLOTS) in scripts/runtime_build/prepr_slot_policy.py, pinned by
# the OMN-19415 test. The preflight sizes the server for every slot of the pool
# at once, because slots boot independently and the server cannot tell them
# apart from the dev lane's own demand until it refuses someone.
PREPR_POOL_SLOT_COUNT=2

# ---------------------------------------------------------------------------
# Slot token
# ---------------------------------------------------------------------------
# Grammar: lowercase letter, then lowercase letters and digits, 1..12 bytes.
#   * no underscore -- the underscore is the SEPARATOR, and a token carrying one
#     makes "does this name end in _<slot>" ambiguous, which is the name fence's
#     whole predicate;
#   * no uppercase -- Postgres folds an unquoted identifier to lowercase, so
#     `Prepr1` and `prepr1` are the same object through one path and different
#     strings through another;
#   * bounded length -- so a derived name stays inside the 63-byte limit for
#     every base name in the maps above.
# A malformed token FAILS CLOSED. A typo would otherwise provision an unfenced
# or a differently-fenced set, which reads exactly like isolation working.
validate_slot_token() {
    _token="$1"
    if [ -z "$_token" ]; then
        fail "$EXIT_SLOT_TOKEN_MISSING" \
            "ONEX_DB_SLOT is unset or empty. This tool provisions a SLOT; the
       unsuffixed dev set is reached by not running it. An empty token is a
       mis-set variable, not a request for the dev set."
    fi
    if ! printf '%s' "$_token" | grep -Eq '^[a-z][a-z0-9]{0,11}$'; then
        fail "$EXIT_SLOT_TOKEN_MALFORMED" \
            "ONEX_DB_SLOT '${_token}' is malformed. Expected ^[a-z][a-z0-9]{0,11}\$."
    fi
}

# NAME FENCE. Every object name this tool addresses passes through here before
# any statement naming it is issued.
assert_in_fence() {
    _name="$1"
    _what="$2"
    case "$_name" in
        *"_${SLOT}") ;;
        *)
            slot_fence_refusal "$EXIT_DERIVED_NAME_OUT_OF_FENCE" \
                "${_what} '${_name}' does not carry the slot suffix '_${SLOT}'. Refusing to
       issue any statement naming it -- an out-of-fence name is an object
       belonging to the dev lane or to another slot."
            ;;
    esac
    if [ "${#_name}" -gt "$PG_MAX_IDENTIFIER_BYTES" ]; then
        fail "$EXIT_IDENTIFIER_TOO_LONG" \
            "${_what} '${_name}' is ${#_name} bytes, over the ${PG_MAX_IDENTIFIER_BYTES}-byte
       Postgres limit. Postgres would TRUNCATE it silently and the truncated
       name can collide with another slot's."
    fi
}

derive() {
    printf '%s_%s' "$1" "$SLOT"
}

# The CONNECTION LIMIT for one derived slot role, from CONNECTION_BUDGET_MAP.
# Fails closed on a role the table does not name or a value that is not a
# positive integer.
connection_limit_for() {
    _cl_base="${1%_"${SLOT}"}"
    for _cl_entry in $CONNECTION_BUDGET_MAP; do
        if [ "${_cl_entry%%:*}" = "$_cl_base" ]; then
            _cl_limit="${_cl_entry#*:}"
            case "$_cl_limit" in
                ''|*[!0-9]*|0) ;;
                *) printf '%s' "$_cl_limit"; return 0 ;;
            esac
            fail "$EXIT_ROLE_WITHOUT_BUDGET" \
                "role '${1}' has connection budget '${_cl_limit}' in
       CONNECTION_BUDGET_MAP; expected a positive integer. 0 locks the principal
       out and -1 is Postgres for unlimited."
        fi
    done
    fail "$EXIT_ROLE_WITHOUT_BUDGET" \
        "role '${1}' (base '${_cl_base}') has no entry in CONNECTION_BUDGET_MAP.
       An unbudgeted slot principal can take the shared server's connections
       from the dev lane (OMN-19415); add it to the table with a measured bound."
}

# ---------------------------------------------------------------------------
# Mode dispatch — before any connection, so --print-scope needs no database
# ---------------------------------------------------------------------------
MODE=""
ENV_FILE=""
CHECK_IDENTIFIER=""

usage() {
    cat >&2 <<'USAGE'
usage: provision_db_slot.sh <mode> [options]

modes:
  --print-scope            derive and fence-check the slot's scope, print it, and
                           open NO connection. The whole refusal table is
                           reachable here, which is what makes it testable on a
                           host with no Postgres and no Docker.
  --apply                  create the slot's databases and principals, grant them
                           their own set, and prove by readback that they reach
                           nothing else.
  --drop                   tear the slot down. Refuses every object that is not
                           both inside the name fence and owned by the slot group.

options:
  --env-file <path>        required with --apply: where the slot's DSNs are
                           written, mode 0600. Credentials are never printed to
                           stdout.
  --check-identifier <base>
                           additionally run the name fence over <base>, to check
                           before adding a service whether its derived name would
                           stay inside the Postgres identifier limit under this
                           slot.

environment:
  ONEX_DB_SLOT             the slot token. Required. ^[a-z][a-z0-9]{0,11}$
  POSTGRES_USER/PASSWORD/HOST/PORT/DB   superuser connection, for --apply/--drop
USAGE
    exit "$EXIT_USAGE"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --print-scope|--apply|--drop)
            [ -n "$MODE" ] && usage
            MODE="$1"
            ;;
        --env-file)
            shift || usage
            [ "$#" -gt 0 ] || usage
            ENV_FILE="$1"
            ;;
        --check-identifier)
            shift || usage
            [ "$#" -gt 0 ] || usage
            CHECK_IDENTIFIER="$1"
            ;;
        -h|--help) usage ;;
        *) usage ;;
    esac
    shift
done

[ -n "$MODE" ] || usage

SLOT="${ONEX_DB_SLOT:-}"
validate_slot_token "$SLOT"

GROUP_ROLE="onex_slot_${SLOT}"

# The group role is itself inside the fence by construction, but it is asserted
# rather than assumed: it is the object the OWNERSHIP fence reads, so a bug that
# mis-derived it would disable control 2 while control 1 still looked healthy.
case "$GROUP_ROLE" in
    *"_${SLOT}") ;;
    *) slot_fence_refusal "$EXIT_DERIVED_NAME_OUT_OF_FENCE" \
        "group role '${GROUP_ROLE}' is out of fence" ;;
esac

if [ -n "$CHECK_IDENTIFIER" ]; then
    assert_in_fence "$(derive "$CHECK_IDENTIFIER")" "checked identifier"
fi

# Derive the whole scope up front and fence-check every name BEFORE any
# statement is issued. A fence breach must abort the run rather than abort it
# halfway through, leaving a slot half-provisioned on a shared server.
SLOT_DATABASES=""
SLOT_ROLES=""
SLOT_DB_ROLE_PAIRS=""

for _entry in $SERVICE_DB_MAP; do
    _db="${_entry%%:*}"
    _role="${_entry#*:}"
    _sdb="$(derive "$_db")"
    _srole="$(derive "$_role")"
    assert_in_fence "$_sdb" "database"
    assert_in_fence "$_srole" "role"
    SLOT_DATABASES="${SLOT_DATABASES}${_sdb}
"
    SLOT_ROLES="${SLOT_ROLES}${_srole}
"
    SLOT_DB_ROLE_PAIRS="${SLOT_DB_ROLE_PAIRS}${_sdb}:${_srole}
"
done

for _db in $INFRA_DATABASES; do
    _sdb="$(derive "$_db")"
    assert_in_fence "$_sdb" "database"
    SLOT_DATABASES="${SLOT_DATABASES}${_sdb}
"
done

for _role in $LOGIN_ONLY_ROLES; do
    _srole="$(derive "$_role")"
    assert_in_fence "$_srole" "role"
    SLOT_ROLES="${SLOT_ROLES}${_srole}
"
done

# Every slot role's budget, resolved before any connection so that a missing
# entry refuses in --print-scope exactly as it would in --apply.
SLOT_ROLE_LIMITS=""
SLOT_CONNECTION_BUDGET=0
for _role in $SLOT_ROLES; do
    _limit="$(connection_limit_for "$_role")"
    SLOT_ROLE_LIMITS="${SLOT_ROLE_LIMITS}${_role}:${_limit}
"
    SLOT_CONNECTION_BUDGET=$((SLOT_CONNECTION_BUDGET + _limit))
done

if [ "$MODE" = "--print-scope" ]; then
    echo "slot=${SLOT}"
    echo "group_role=${GROUP_ROLE}"
    for _n in $SLOT_DATABASES; do echo "database=${_n}"; done
    for _n in $SLOT_ROLES; do echo "role=${_n}"; done
    for _n in $SLOT_ROLE_LIMITS; do echo "connection_limit=${_n}"; done
    echo "slot_connection_budget=${SLOT_CONNECTION_BUDGET}"
    echo "dev_lane_connection_budget=${DEV_LANE_CONNECTION_BUDGET}"
    echo "pool_slot_count=${PREPR_POOL_SLOT_COUNT}"
    exit 0
fi

# ---------------------------------------------------------------------------
# From here on a connection is required
# ---------------------------------------------------------------------------
PGUSER="${POSTGRES_USER:-postgres}"
PGHOST="${POSTGRES_HOST:-postgres}"
PGPORT="${POSTGRES_PORT:-5432}"
PGADMINDB="${POSTGRES_DB:-postgres}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set — provisioning a slot needs the superuser connection}"
PGPASSWORD="$POSTGRES_PASSWORD"
export PGPASSWORD

# ---------------------------------------------------------------------------
# HOW psql IS REACHED (OMN-18893)
#
# This script originally invoked a bare `psql`, which assumes a PostgreSQL
# client on the host. The lab host has none -- no binary on PATH, none under
# /usr/lib/postgresql, and no postgresql-client package -- and it never needed
# one, because every migration seam in this repository runs psql INSIDE a
# `postgres:16-alpine` container against the lane's own network. The dev lane's
# forward-migration, cloud-migration, intelligence-migration and migration-gate
# services are all that shape.
#
# Found on the first live slot provisioning: `psql: command not found`, after
# the source snapshot had been staged and the lane lock taken.
#
# So psql is resolved rather than assumed: the host binary when one exists (a
# developer machine, and what the unit tests exercise), otherwise a throwaway
# container on the lane network. Both paths take the password from the
# environment and never from argv, which is why the container form passes a
# bare `-e PGPASSWORD` rather than an inline value -- a value there would be
# visible in `docker inspect` and in the host's process list.
#
# The container path adds no host package and no image pull: postgres:16-alpine
# is already resident, because the dev lane's own migration one-shots run it.
# ---------------------------------------------------------------------------
ONEX_PSQL_IMAGE="${ONEX_PSQL_IMAGE:-postgres:16-alpine}"
ONEX_PSQL_NETWORK="${ONEX_PSQL_NETWORK:-omnibase-infra-network}"

if command -v psql >/dev/null 2>&1; then
    ONEX_PSQL_VIA="host-binary"
    psql_run() { psql "$@"; }
elif command -v docker >/dev/null 2>&1; then
    ONEX_PSQL_VIA="container:${ONEX_PSQL_IMAGE}@${ONEX_PSQL_NETWORK}"
    psql_run() {
        # -i so a heredoc on stdin reaches psql; --rm so nothing accumulates.
        docker run --rm -i --network "$ONEX_PSQL_NETWORK" \
            -e PGPASSWORD "$ONEX_PSQL_IMAGE" psql "$@"
    }
else
    fail "$EXIT_USAGE" \
        "no way to reach PostgreSQL: there is no psql on PATH and no docker to
       run one in. Slot provisioning refuses rather than guessing, because a
       half-provisioned slot on a SHARED server is worse than an unprovisioned
       one."
fi

# Printed rather than merely chosen: which path answered is part of the
# provisioning evidence, and a run that silently switched paths between two
# hosts is a difference an operator should be able to see in the log.
echo "[provision-db-slot] psql via ${ONEX_PSQL_VIA}" >&2

psql_admin() {
    psql_run -v ON_ERROR_STOP=1 -X -q -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGADMINDB" "$@"
}

psql_admin_in() {
    _target_db="$1"
    shift
    psql_run -v ON_ERROR_STOP=1 -X -q -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$_target_db" "$@"
}

scalar() {
    psql_run -X -qAt -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$PGADMINDB" -c "$1"
}

# The whole fence rests on this connection being able to see and refuse
# cluster-wide objects. A non-superuser would fail later, in the middle of
# provisioning, with a permission error that reads like a bug in the maps.
_is_super="$(scalar "SELECT rolsuper FROM pg_roles WHERE rolname = current_user")"
if [ "$_is_super" != "t" ]; then
    fail "$EXIT_NOT_SUPERUSER" \
        "connected as '${PGUSER}', which is not a superuser. Slot provisioning
       creates cluster-wide roles and databases and must run as the server's
       superuser. The slot's OWN principals are deliberately not superusers."
fi

new_password() {
    # Hex only, matching the repository's password validator in both
    # provisioning seams. 32 bytes of urandom, rendered hex, no openssl
    # dependency: busybox od and tr are present in postgres:16-alpine.
    od -An -tx1 -N32 /dev/urandom | tr -d ' \n'
}

role_exists() {
    [ "$(scalar "SELECT 1 FROM pg_roles WHERE rolname = '$1'")" = "1" ]
}

database_exists() {
    [ "$(scalar "SELECT 1 FROM pg_database WHERE datname = '$1'")" = "1" ]
}

# OWNERSHIP FENCE. A pre-existing role is touched only when it is a member of
# THIS slot's group role. That is the difference between "a role whose name ends
# in the same characters" and "a role this tool created", and it is the control
# that makes it structurally impossible for a mis-edited map to reset a dev-lane
# role's password.
# THE ONE EXEMPTION, AND IT IS A CHECKED PROPERTY RATHER THAN A NAME
# --------------------------------------------------------------------
# The cluster's maintenance database is reachable by every role on any stock
# Postgres: `postgres` ships with PUBLIC's default CONNECT and is not created by
# this repository, so no seam here owns its ACL. It is exempt from the isolation
# proof ONLY while it holds no lane data, and that is verified live rather than
# asserted -- a database in this list that has gained an application table makes
# the proof refuse again, naming it.
#
# This is deliberately not an allowlist of convenience. Membership buys nothing
# on its own: the property below is what grants the exemption, and a database
# not in the list is never exempt however empty it is.
NO_LANE_DATA_DATABASES="postgres"

database_holds_no_lane_data() {
    _nld_db="$1"
    _nld_listed=0
    for _nld_candidate in $NO_LANE_DATA_DATABASES; do
        [ "$_nld_candidate" = "$_nld_db" ] && _nld_listed=1
    done
    [ "$_nld_listed" -eq 1 ] || return 1
    _nld_tables="$(psql_run -X -qAt -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$_nld_db" \
        -c "SELECT count(*) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema')")"
    if [ "$_nld_tables" != "0" ]; then
        echo "[provision-db-slot]   '${_nld_db}' is listed as holding no lane data but reports ${_nld_tables} application tables — the exemption does not apply" >&2
        return 1
    fi
    echo "[provision-db-slot]   '${_nld_db}' is reachable and exempt: 0 application tables (verified, not assumed)"
    return 0
}

assert_owned_by_slot() {
    _role="$1"
    _member="$(scalar "
        SELECT 1 FROM pg_auth_members m
        JOIN pg_roles r ON r.oid = m.roleid
        JOIN pg_roles g ON g.oid = m.member
        WHERE r.rolname = '${GROUP_ROLE}' AND g.rolname = '${_role}'")"
    if [ "$_member" != "1" ]; then
        slot_fence_refusal "$EXIT_ROLE_OUTSIDE_SLOT_GROUP" \
            "role '${_role}' already exists and is NOT a member of '${GROUP_ROLE}'.
       Refusing to alter it. A role this tool did not create belongs to the dev
       lane or to another slot, and altering it would reset a credential its
       owner is still holding in a running container's environment."
    fi
}

if [ "$MODE" = "--apply" ]; then
    [ -n "$ENV_FILE" ] || {
        echo "[provision-db-slot] --apply requires --env-file <path>" >&2
        usage
    }

    # --- capacity preflight (OMN-19415) -------------------------------------
    # Before anything is created: is this server sized for the dev lane plus
    # every pool slot at full budget? An unreadable answer is a refusal, not a
    # pass -- a preflight that cannot see the setting has not checked it.
    _capacity="$(scalar "SELECT current_setting('max_connections') || ' ' || current_setting('superuser_reserved_connections') || ' ' || coalesce(current_setting('reserved_connections', true), '0')")" || _capacity=""
    # shellcheck disable=SC2086  # deliberate field split of three integers
    set -- $_capacity
    _max_conn="${1:-}"
    _su_reserved="${2:-}"
    _reserved="${3:-}"
    for _v in "$_max_conn" "$_su_reserved" "$_reserved"; do
        case "$_v" in
            ''|*[!0-9]*)
                fail "$EXIT_SERVER_UNDERSIZED" \
                    "could not read the server's connection capacity (got '${_capacity}').
       Refusing to provision a slot on a shared server whose capacity is
       unknown (OMN-19415)."
                ;;
        esac
    done
    _required=$((DEV_LANE_CONNECTION_BUDGET + PREPR_POOL_SLOT_COUNT * SLOT_CONNECTION_BUDGET + _su_reserved + _reserved))
    echo "[provision-db-slot] capacity: max_connections=${_max_conn}, required=${_required} (dev lane ${DEV_LANE_CONNECTION_BUDGET} + ${PREPR_POOL_SLOT_COUNT} slots x ${SLOT_CONNECTION_BUDGET} + reserved ${_su_reserved}+${_reserved})"
    if [ "$_max_conn" -lt "$_required" ]; then
        fail "$EXIT_SERVER_UNDERSIZED" \
            "the shared server's max_connections is ${_max_conn}; the dev lane
       plus ${PREPR_POOL_SLOT_COUNT} pool slots need ${_required}. A slot booted here
       takes its connections from the dev lane (OMN-19415: 100/100 and 1171 dev
       refusals on 2026-09-24). The server's size is declared on the dev lane's
       postgres in docker/docker-compose.dev-lane.yml; it takes effect when that
       container is recreated, which is a dev-lane restart."
    fi

    echo "[provision-db-slot] slot '${SLOT}' — group role ${GROUP_ROLE}"

    # The group role is NOLOGIN and holds no privilege. It exists only as the
    # ownership marker the fence above reads.
    psql_admin <<EOSQL
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${GROUP_ROLE}') THEN
        CREATE ROLE "${GROUP_ROLE}" WITH NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
END
\$\$;
COMMENT ON ROLE "${GROUP_ROLE}" IS 'OMN-18892 pre-PR verify slot ${SLOT}: ownership marker for this slot''s principals. Membership is what permits this tool to alter a role.';
EOSQL

    # --- principals ---------------------------------------------------------
    # Attributes are pinned explicitly on CREATE and never left to cluster
    # defaults. NOSUPERUSER and NOBYPASSRLS are the two load-bearing ones: they
    # are what makes FORCE ROW LEVEL SECURITY on the tenant tables enforced
    # against the connection the database actually sees, rather than inert.
    # NOCREATEROLE and NOCREATEDB are what BOUND the migration path -- a branch
    # migration running as this principal that reaches for a cluster-wide object
    # fails as the slot's finding instead of succeeding as everyone's problem.
    : > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "# OMN-18892 slot '${SLOT}' credentials. Generated, never supplied by a caller." >> "$ENV_FILE"
    echo "ONEX_DB_SLOT=${SLOT}" >> "$ENV_FILE"

    for _role in $SLOT_ROLES; do
        assert_in_fence "$_role" "role"
        if role_exists "$_role"; then
            assert_owned_by_slot "$_role"
            echo "[provision-db-slot]   role ${_role} exists and is owned by this slot — rotating its own credential"
        else
            echo "[provision-db-slot]   creating role ${_role}"
        fi
        _pw="$(new_password)"
        _limit="$(connection_limit_for "$_role")"
        psql_admin <<EOSQL
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${_role}') THEN
        CREATE ROLE "${_role}" WITH
            LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION
            CONNECTION LIMIT ${_limit}
            PASSWORD '${_pw}';
    ELSE
        ALTER ROLE "${_role}" WITH LOGIN CONNECTION LIMIT ${_limit} PASSWORD '${_pw}';
    END IF;
END
\$\$;
GRANT "${GROUP_ROLE}" TO "${_role}";
EOSQL
        printf '%s_PASSWORD=%s\n' \
            "$(printf '%s' "$_role" | tr '[:lower:]' '[:upper:]')" "$_pw" >> "$ENV_FILE"
        unset _pw
    done

    # --- databases ----------------------------------------------------------
    for _db in $SLOT_DATABASES; do
        assert_in_fence "$_db" "database"
        if database_exists "$_db"; then
            echo "[provision-db-slot]   database ${_db} already present"
        else
            echo "[provision-db-slot]   creating database ${_db}"
            psql_admin <<EOSQL
SELECT 'CREATE DATABASE "${_db}"'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${_db}')\gexec
EOSQL
        fi
        # PUBLIC holds CONNECT on a new database by default. Revoking it here
        # rather than relying on the cluster already having been hardened is the
        # difference between a fail-closed provisioner and one whose isolation
        # is a property of somebody else's earlier run.
        psql_admin -c "REVOKE CONNECT ON DATABASE \"${_db}\" FROM PUBLIC;"
    done

    # --- grants, slot's own set only ----------------------------------------
    for _pair in $SLOT_DB_ROLE_PAIRS; do
        _db="${_pair%%:*}"
        _role="${_pair#*:}"
        assert_in_fence "$_db" "database"
        assert_in_fence "$_role" "role"
        echo "[provision-db-slot]   granting ${_role} its own database ${_db}"
        psql_admin -c "GRANT CONNECT ON DATABASE \"${_db}\" TO \"${_role}\";"
        psql_admin_in "$_db" <<EOSQL
GRANT USAGE, CREATE ON SCHEMA public TO "${_role}";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "${_role}";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO "${_role}";
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "${_role}";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "${_role}";
EOSQL
    done

    # omninode_cloud's corpus opens with CREATE EXTENSION pgcrypto, and
    # `trusted = t` waives the SUPERUSER requirement, never the PRIVILEGE
    # requirement -- CREATE on the DATABASE is a different privilege from CREATE
    # on schema public. Mirrors the corpus-applier seam in both existing
    # provisioning scripts (OMN-18508); without it the slot's cloud migration
    # dies on its first statement.
    _cloud_db="$(derive omninode_cloud)"
    _cloud_role="$(derive role_omninode)"
    assert_in_fence "$_cloud_db" "database"
    assert_in_fence "$_cloud_role" "role"
    psql_admin -c "GRANT CREATE ON DATABASE \"${_cloud_db}\" TO \"${_cloud_role}\";"
    _cloud_create_ok="$(scalar "SELECT has_database_privilege('${_cloud_role}', '${_cloud_db}', 'CREATE')")"
    if [ "$_cloud_create_ok" != "t" ]; then
        fail 1 "${_cloud_role} still lacks CREATE on ${_cloud_db} after the grant.
       A GRANT issued without grant option on the object warns and returns
       success rather than raising, so this readback is the only thing between a
       real grant and a silent no-op."
    fi

    # --- the falsifier, run in-band -----------------------------------------
    # AC-4. This is not a post-hoc probe somebody may forget to run: the
    # provisioner refuses to report success until it has proven, from the
    # catalog, that every principal it minted reaches the slot's databases and
    # NOTHING ELSE on the server.
    #
    # An all-refused result is worthless without a positive control -- a broken
    # query returns exactly that. So the same pass asserts the slot's OWN
    # databases ARE reachable, and a failure of the control is its own exit code.
    echo "[provision-db-slot] proving isolation from the catalog"
    _out_of_fence_dbs="$(scalar "
        SELECT datname FROM pg_database
        WHERE datname NOT IN ('template0','template1')
          AND datname NOT LIKE '%\\_${SLOT}'
        ORDER BY datname")"
    _checked=0
    _exempted=0
    for _role in $SLOT_ROLES; do
        for _db in $_out_of_fence_dbs; do
            _reach="$(scalar "SELECT has_database_privilege('${_role}', '${_db}', 'CONNECT')")"
            if [ "$_reach" != "f" ]; then
                if database_holds_no_lane_data "$_db"; then
                    _exempted=$((_exempted + 1))
                    continue
                fi
                slot_fence_refusal "$EXIT_REACHES_UNSUFFIXED_DATABASE" \
                    "slot principal '${_role}' holds CONNECT on out-of-fence database
       '${_db}'. The slot is NOT isolated from the dev lane. Nothing has been
       torn down; inspect before re-running.
       The remedy is to close the database, never to narrow this test:
         REVOKE CONNECT ON DATABASE \"${_db}\" FROM PUBLIC;
       A database created by an \`onex-create-database\` directive is closed for
       every lane by the OMN-18892 seam in scripts/run-forward-migrations.sh."
            fi
            _checked=$((_checked + 1))
        done
    done
    if [ "$_checked" -eq 0 ]; then
        fail "$EXIT_REACHES_UNSUFFIXED_DATABASE" \
            "the isolation proof checked zero pairs. An empty result is not
       evidence of absence -- it means the out-of-fence database enumeration
       returned nothing, which on a shared server is itself the defect."
    fi
    echo "[provision-db-slot]   refused: ${_checked} (principal, out-of-fence database) pairs, ${_exempted} exempt as holding no lane data"

    _controls=0
    for _pair in $SLOT_DB_ROLE_PAIRS; do
        _db="${_pair%%:*}"
        _role="${_pair#*:}"
        _reach="$(scalar "SELECT has_database_privilege('${_role}', '${_db}', 'CONNECT')")"
        if [ "$_reach" != "t" ]; then
            fail "$EXIT_CANNOT_REACH_OWN_DATABASE" \
                "positive control FAILED: slot principal '${_role}' cannot reach its own
       database '${_db}'. Every refusal above is therefore unreadable — a
       predicate that refuses everything proves nothing."
        fi
        _controls=$((_controls + 1))
    done
    echo "[provision-db-slot]   positive control: ${_controls} principals reach their own database"

    # Attributes read back from what the database itself reports, not from what
    # the CREATE statement said. AC: the slot's connection is non-superuser and
    # no-bypass.
    for _role in $SLOT_ROLES; do
        _attrs="$(scalar "SELECT rolsuper::text || ',' || rolbypassrls::text || ',' || rolcreaterole::text || ',' || rolcreatedb::text FROM pg_roles WHERE rolname = '${_role}'")"
        if [ "$_attrs" != "false,false,false,false" ]; then
            fail 1 "role '${_role}' reports attributes '${_attrs}', expected
       'false,false,false,false' (super,bypassrls,createrole,createdb). A
       BYPASSRLS principal makes the forced row-level security on the tenant
       tables inert; a CREATEROLE one unbounds the migration path."
        fi
    done
    echo "[provision-db-slot]   all ${_controls} service principals + login-only principals report NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB"

    # OMN-19415: the connection budget, read back from pg_roles.rolconnlimit
    # rather than trusted from the statement that set it.
    for _pair in $SLOT_ROLE_LIMITS; do
        _role="${_pair%%:*}"
        _limit="${_pair#*:}"
        _actual="$(scalar "SELECT rolconnlimit FROM pg_roles WHERE rolname = '${_role}'")"
        if [ "$_actual" != "$_limit" ]; then
            fail 1 "role '${_role}' reports rolconnlimit '${_actual}', expected
       ${_limit}. An unbounded slot principal can take the shared server's
       connections from the dev lane (OMN-19415)."
        fi
    done
    echo "[provision-db-slot]   connection budget: ${SLOT_CONNECTION_BUDGET} across the slot's principals, each read back from pg_roles.rolconnlimit"

    for _db in $SLOT_DATABASES; do echo "SLOT_DATABASE=${_db}" >> "$ENV_FILE"; done
    echo "[provision-db-slot] slot '${SLOT}' provisioned. Credentials in ${ENV_FILE} (mode 0600)."
    exit 0
fi

if [ "$MODE" = "--drop" ]; then
    echo "[provision-db-slot] tearing down slot '${SLOT}'"
    # ENUMERATE FROM THE CATALOG, NOT FROM THE MAPS. A slot acquires objects the
    # maps do not name: scripts/run-forward-migrations.sh creates one database
    # per `onex-create-database` directive, suffixed under a slot, and
    # `keycloak_p18892` is exactly the database a map-driven teardown left behind
    # on this tool's first live run. A teardown that misses an object leaks it
    # onto a SHARED server once per slot claim, forever, and Task 7's reaper
    # would inherit the leak rather than fix it.
    #
    # Enumerating from pg_database/pg_auth_members is complete by construction:
    # it cannot miss an object however it was created. The name fence still runs
    # on every row, so widening the SOURCE of the list does not widen what may
    # be dropped -- an unsuffixed row would refuse rather than be removed.
    _live_databases="$(scalar "
        SELECT datname FROM pg_database
        WHERE datname LIKE '%\\_${SLOT}' ESCAPE '\\'
          AND datname NOT IN ('template0','template1')
        ORDER BY datname")"
    for _db in $_live_databases; do
        assert_in_fence "$_db" "database"
        echo "[provision-db-slot]   dropping database ${_db}"
        psql_admin -c "DROP DATABASE IF EXISTS \"${_db}\" WITH (FORCE);"
    done
    # Roles are enumerated by MEMBERSHIP of the slot group rather than by name:
    # membership is what the ownership fence reads, so a role that is a member is
    # exactly a role this tool created, and a role that merely ends in the slot's
    # characters is not dropped.
    _live_roles="$(scalar "
        SELECT g.rolname FROM pg_auth_members m
        JOIN pg_roles r ON r.oid = m.roleid
        JOIN pg_roles g ON g.oid = m.member
        WHERE r.rolname = '${GROUP_ROLE}'
        ORDER BY g.rolname")"
    for _role in $_live_roles; do
        assert_in_fence "$_role" "role"
        if role_exists "$_role"; then
            # The ownership fence applies to teardown exactly as it applies to
            # provisioning. A role this tool did not create is never dropped,
            # whatever its name ends in.
            assert_owned_by_slot "$_role"
            echo "[provision-db-slot]   dropping role ${_role}"
            psql_admin -c "REVOKE \"${GROUP_ROLE}\" FROM \"${_role}\";"
            psql_admin -c "DROP ROLE IF EXISTS \"${_role}\";"
        fi
    done
    if role_exists "$GROUP_ROLE"; then
        psql_admin -c "DROP ROLE IF EXISTS \"${GROUP_ROLE}\";"
    fi
    echo "[provision-db-slot] slot '${SLOT}' torn down"
    exit 0
fi

usage
