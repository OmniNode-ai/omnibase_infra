#!/bin/bash
# create-multiple-databases.sh
# PostgreSQL initialization script to create multiple databases and per-service roles.
#
# This script is executed during PostgreSQL container initialization
# (only when the data directory is empty, i.e., first startup).
# It is idempotent — safe to re-run via manual invocation.
#
# Databases provisioned (DB-SPLIT-05 / OMN-2056):
#   omnibase_infra, omniintelligence, omniclaude,
#   omnimemory, omninode_cloud, omnidash_analytics
#
# Additional databases (infrastructure):
#   infisical_db  (Infisical secrets management)
#
# Per-service roles:
#   role_omnibase, role_omniintelligence, role_omniclaude,
#   role_omnimemory, role_omninode, role_omnidash
#
# Each role can ONLY access its own database. Cross-DB access is revoked.

set -e
set -u
# Note: set -u with ${!password_var:-} (indirect expansion + default) means
# typos in SERVICE_DB_MAP password_var fields will silently default to empty
# rather than erroring. This is intentional — empty means "skip this role".
# Verify env var names match docker-compose.infra.yml when editing SERVICE_DB_MAP.

: "${POSTGRES_USER:?POSTGRES_USER must be set}"
: "${POSTGRES_DB:?POSTGRES_DB must be set}"

# =============================================================================
# Configuration: database → role mapping
# =============================================================================
# Format: "database:role:password_env_var"
# The password_env_var names the environment variable holding the role password.
SERVICE_DB_MAP=(
    "omnibase_infra:role_omnibase:ROLE_OMNIBASE_PASSWORD"
    "omniintelligence:role_omniintelligence:ROLE_OMNIINTELLIGENCE_PASSWORD"
    "omniclaude:role_omniclaude:ROLE_OMNICLAUDE_PASSWORD"
    "omnimemory:role_omnimemory:ROLE_OMNIMEMORY_PASSWORD"
    "omninode_cloud:role_omninode:ROLE_OMNINODE_PASSWORD"
    "omnidash_analytics:role_omnidash:ROLE_OMNIDASH_PASSWORD"
)

# Additional databases without dedicated roles (managed by superuser)
INFRA_DATABASES=("infisical_db")

# =============================================================================
# Helper functions
# =============================================================================

validate_identifier() {
    local name="$1"
    local context="${2:-Identifier}"
    if [ ${#name} -gt 63 ]; then
        echo "ERROR: $context '$name' exceeds 63-character limit" >&2
        return 1
    fi
    if ! echo "$name" | grep -qE '^[a-zA-Z_][a-zA-Z0-9_-]*$'; then
        echo "ERROR: Invalid $context '$name' - must match ^[a-zA-Z_][a-zA-Z0-9_-]*$" >&2
        return 1
    fi
}

validate_password() {
    local password="$1"
    local context="$2"
    if [ -z "$password" ]; then
        echo "ERROR: Empty password for $context" >&2
        return 1
    fi
    if echo "$password" | grep -qE '^__REPLACE_WITH_.*__$'; then
        echo "ERROR: Password for $context is still a placeholder." >&2
        echo "       Replace with: openssl rand -hex 32" >&2
        return 1
    fi
    if ! echo "$password" | grep -qE '^[0-9a-fA-F]+$'; then
        echo "ERROR: Password for $context contains non-hex characters." >&2
        echo "       Generate with: openssl rand -hex 32" >&2
        return 1
    fi
}

create_database() {
    local database="$1"
    validate_identifier "$database" "Database name" || return 1
    echo "  Creating database: $database"
    # Safety: $database is used in two SQL contexts below:
    #   - Double-quoted identifier ("$database") for CREATE DATABASE
    #   - Single-quoted string literal ('$database') for the pg_database lookup
    # Both are safe because validate_identifier restricts to [a-zA-Z_][a-zA-Z0-9_-]*
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        SELECT 'CREATE DATABASE "$database"'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
EOSQL
    echo "  Database '$database' ready."
}

create_role() {
    local role_name="$1"
    local role_password="$2"
    validate_identifier "$role_name" "Role name" || return 1
    validate_password "$role_password" "$role_name" || return 1
    # Escape single quotes for safe SQL interpolation (' → '')
    # Note: validate_password enforces hex-only ([0-9a-fA-F]+) so single quotes
    # cannot appear in practice, but the escaping is retained for defense-in-depth.
    local escaped_password="${role_password//\'/\'\'}"
    echo "  Creating role: $role_name"
    # Safety: $role_name is used in dual SQL contexts below (double-quoted identifier
    # and single-quoted string literal), safe because validate_identifier restricts
    # to [a-zA-Z_][a-zA-Z0-9_-]*. Same rationale as create_database().
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$role_name') THEN
                CREATE ROLE "$role_name" WITH LOGIN PASSWORD '$escaped_password';
                RAISE NOTICE 'Created role: $role_name';
            ELSE
                -- Update password on re-run to ensure it stays in sync with env
                ALTER ROLE "$role_name" WITH LOGIN PASSWORD '$escaped_password';
                RAISE NOTICE 'Role $role_name already exists, password updated';
            END IF;
        END
        \$\$;
EOSQL
}

grant_role_to_database() {
    local role_name="$1"
    local database="$2"
    validate_identifier "$role_name" "Role name" || return 1
    validate_identifier "$database" "Database name" || return 1
    echo "  Granting $role_name full access to $database"
    # CONNECT privilege
    # Note: explicit || return 1 because set -e is disabled when caller uses ||
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL || return 1
        GRANT CONNECT ON DATABASE "$database" TO "$role_name";
EOSQL
    # Schema and table privileges (must run against the target database)
    # NOTE: ALTER DEFAULT PRIVILEGES only applies to objects created by the
    # CURRENT user (postgres superuser). If migrations run as the service role,
    # you must run ALTER DEFAULT PRIVILEGES as that role too (or run migrations
    # as the superuser and grant to the service role).
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$database" <<-EOSQL || return 1
        GRANT USAGE, CREATE ON SCHEMA public TO "$role_name";
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "$role_name";
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
            GRANT USAGE, SELECT ON SEQUENCES TO "$role_name";
        -- Grant on any existing tables/sequences
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "$role_name";
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "$role_name";
EOSQL
}

revoke_cross_db_access() {
    local role_name="$1"
    local own_database="$2"
    validate_identifier "$role_name" "Role name" || return 1
    validate_identifier "$own_database" "Database name" || return 1
    echo "  Revoking cross-DB access for $role_name (allowed: $own_database only)"
    # Note: explicit || return 1 because set -e is disabled when caller uses ||
    for entry in "${SERVICE_DB_MAP[@]}"; do
        IFS=':' read -r db _ _ <<< "$entry"
        if [ "$db" != "$own_database" ]; then
            psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL || return 1
                REVOKE CONNECT ON DATABASE "$db" FROM "$role_name";
EOSQL
        fi
    done
    # Also revoke from infrastructure databases
    for db in "${INFRA_DATABASES[@]}"; do
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL || return 1
            REVOKE CONNECT ON DATABASE "$db" FROM "$role_name";
EOSQL
    done
    # Also revoke from the default database (if different from own_database)
    if [ "$POSTGRES_DB" != "$own_database" ]; then
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL || return 1
            REVOKE CONNECT ON DATABASE "$POSTGRES_DB" FROM "$role_name";
EOSQL
    fi
}

# =============================================================================
# Phase 1: Create all databases
# =============================================================================
echo "============================================="
echo "Phase 1: Creating databases"
echo "============================================="

# Service databases
for entry in "${SERVICE_DB_MAP[@]}"; do
    IFS=':' read -r db _ _ <<< "$entry"
    create_database "$db" || { echo "FATAL: Failed to create database '$db'" >&2; exit 1; }
done

# Infrastructure databases
for db in "${INFRA_DATABASES[@]}"; do
    create_database "$db" || { echo "FATAL: Failed to create database '$db'" >&2; exit 1; }
done

echo ""

# =============================================================================
# Phase 2: Create per-service roles
# =============================================================================
echo "============================================="
echo "Phase 2: Creating per-service roles"
echo "============================================="

ROLES_CREATED=0
ROLES_SKIPPED=0

for entry in "${SERVICE_DB_MAP[@]}"; do
    IFS=':' read -r db role_name password_var <<< "$entry"
    role_password="${!password_var:-}"

    if [ -z "$role_password" ]; then
        echo "  SKIP: $role_name — $password_var not set"
        ROLES_SKIPPED=$((ROLES_SKIPPED + 1))
        continue
    fi

    # Pre-check: validate here for user-facing SKIP message.
    # create_role() also validates internally as its own safety guard.
    validate_password "$role_password" "$role_name" || {
        echo "  SKIP: $role_name — invalid password"
        ROLES_SKIPPED=$((ROLES_SKIPPED + 1))
        continue
    }

    create_role "$role_name" "$role_password" || {
        echo "  FAIL: $role_name — create_role failed" >&2
        ROLES_SKIPPED=$((ROLES_SKIPPED + 1))
        continue
    }
    ROLES_CREATED=$((ROLES_CREATED + 1))
done

echo ""
echo "  Roles created/updated: $ROLES_CREATED, skipped: $ROLES_SKIPPED"
echo ""

# =============================================================================
# Phase 3: Grant per-service access
# =============================================================================
echo "============================================="
echo "Phase 3: Granting per-service access"
echo "============================================="

for entry in "${SERVICE_DB_MAP[@]}"; do
    IFS=':' read -r db role_name password_var <<< "$entry"
    role_password="${!password_var:-}"

    # Skip roles that weren't created (empty or invalid password)
    if [ -z "$role_password" ] || ! validate_password "$role_password" "$role_name" 2>/dev/null; then
        continue
    fi

    grant_role_to_database "$role_name" "$db" || {
        echo "  WARNING: grant failed for $role_name on $db" >&2
    }
done

echo ""

# =============================================================================
# Phase 4: Revoke cross-database access
# =============================================================================
echo "============================================="
echo "Phase 4: Revoking cross-database access"
echo "============================================="

# Collect ALL managed databases for comprehensive PUBLIC revocation.
# This covers: service DBs, infrastructure DBs, and the default POSTGRES_DB.
ALL_MANAGED_DBS=()
for entry in "${SERVICE_DB_MAP[@]}"; do
    IFS=':' read -r db _ _ <<< "$entry"
    ALL_MANAGED_DBS+=("$db")
done
for db in "${INFRA_DATABASES[@]}"; do
    ALL_MANAGED_DBS+=("$db")
done
# Include POSTGRES_DB if not already in the list (avoids double-revoke, though REVOKE is idempotent)
_pg_db_found=0
for db in "${ALL_MANAGED_DBS[@]}"; do
    if [ "$db" = "$POSTGRES_DB" ]; then _pg_db_found=1; break; fi
done
if [ "$_pg_db_found" -eq 0 ]; then
    ALL_MANAGED_DBS+=("$POSTGRES_DB")
fi

# Revoke PUBLIC connect on every managed database (default PostgreSQL allows everyone).
# Superusers bypass all permission checks, so this is safe for the postgres user.
for db in "${ALL_MANAGED_DBS[@]}"; do
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL || { echo "  WARNING: Failed to revoke PUBLIC connect on $db" >&2; }
        REVOKE CONNECT ON DATABASE "$db" FROM PUBLIC;
EOSQL
done

# Then revoke cross-DB access per role
for entry in "${SERVICE_DB_MAP[@]}"; do
    IFS=':' read -r db role_name password_var <<< "$entry"
    role_password="${!password_var:-}"

    # Skip roles that weren't created (empty or invalid password)
    if [ -z "$role_password" ] || ! validate_password "$role_password" "$role_name" 2>/dev/null; then
        continue
    fi

    revoke_cross_db_access "$role_name" "$db" || {
        echo "  WARNING: revoke_cross_db_access failed for $role_name" >&2
    }
done

echo ""

# =============================================================================
# Verification
# =============================================================================
echo "============================================="
echo "Verification"
echo "============================================="

echo ""
echo "Databases:"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "SELECT datname FROM pg_database WHERE datname NOT IN ('template0','template1') ORDER BY datname;"

echo ""
echo "Roles:"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "SELECT rolname, rolcanlogin FROM pg_roles WHERE rolname LIKE 'role_%' ORDER BY rolname;"

echo ""
echo "============================================="
echo "Database provisioning complete."
echo "============================================="
