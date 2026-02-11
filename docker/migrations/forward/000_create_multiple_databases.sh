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

validate_db_name() {
    local name="$1"
    if [ ${#name} -gt 63 ]; then
        echo "ERROR: Database name '$name' exceeds 63-character limit" >&2
        return 1
    fi
    if ! echo "$name" | grep -qE '^[a-zA-Z_][a-zA-Z0-9_-]*$'; then
        echo "ERROR: Invalid database name '$name' - must match ^[a-zA-Z_][a-zA-Z0-9_-]*$" >&2
        return 1
    fi
}

create_database() {
    local database="$1"
    validate_db_name "$database"
    echo "  Creating database: $database"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        SELECT 'CREATE DATABASE "$database"'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
EOSQL
    echo "  Database '$database' ready."
}

create_role() {
    local role_name="$1"
    local role_password="$2"
    # Escape single quotes for safe SQL interpolation (' → '')
    local escaped_password="${role_password//\'/\'\'}"
    echo "  Creating role: $role_name"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$role_name') THEN
                CREATE ROLE "$role_name" WITH LOGIN PASSWORD '$escaped_password';
                RAISE NOTICE 'Created role: $role_name';
            ELSE
                -- Update password on re-run to ensure it stays in sync with env
                ALTER ROLE "$role_name" WITH PASSWORD '$escaped_password';
                RAISE NOTICE 'Role $role_name already exists, password updated';
            END IF;
        END
        \$\$;
EOSQL
}

grant_role_to_database() {
    local role_name="$1"
    local database="$2"
    echo "  Granting $role_name full access to $database"
    # CONNECT privilege
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        GRANT CONNECT ON DATABASE "$database" TO "$role_name";
EOSQL
    # Schema and table privileges (must run against the target database)
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$database" <<-EOSQL
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
    echo "  Revoking cross-DB access for $role_name (allowed: $own_database only)"
    for entry in "${SERVICE_DB_MAP[@]}"; do
        IFS=':' read -r db _ _ <<< "$entry"
        if [ "$db" != "$own_database" ]; then
            psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
                REVOKE CONNECT ON DATABASE "$db" FROM "$role_name";
EOSQL
        fi
    done
    # Also revoke from infrastructure databases
    for db in "${INFRA_DATABASES[@]}"; do
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
            REVOKE CONNECT ON DATABASE "$db" FROM "$role_name";
EOSQL
    done
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
    create_database "$db"
done

# Infrastructure databases
for db in "${INFRA_DATABASES[@]}"; do
    create_database "$db"
done

# Legacy database (POSTGRES_MULTIPLE_DATABASES support for backwards compat)
if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
    echo "  Additional databases from POSTGRES_MULTIPLE_DATABASES: $POSTGRES_MULTIPLE_DATABASES"
    IFS=',' read -ra EXTRA_DBS <<< "$POSTGRES_MULTIPLE_DATABASES"
    for db in "${EXTRA_DBS[@]}"; do
        db=$(echo "$db" | xargs)
        if [ -n "$db" ] && [ "$db" != "$POSTGRES_DB" ]; then
            create_database "$db"
        fi
    done
fi

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

    create_role "$role_name" "$role_password"
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

    if [ -z "$role_password" ]; then
        continue  # Skip roles that weren't created
    fi

    grant_role_to_database "$role_name" "$db"
done

echo ""

# =============================================================================
# Phase 4: Revoke cross-database access
# =============================================================================
echo "============================================="
echo "Phase 4: Revoking cross-database access"
echo "============================================="

# First, revoke PUBLIC connect on all managed databases (default allows everyone)
for entry in "${SERVICE_DB_MAP[@]}"; do
    IFS=':' read -r db _ _ <<< "$entry"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        REVOKE CONNECT ON DATABASE "$db" FROM PUBLIC;
EOSQL
done
for db in "${INFRA_DATABASES[@]}"; do
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        REVOKE CONNECT ON DATABASE "$db" FROM PUBLIC;
EOSQL
done

# Then revoke cross-DB access per role
for entry in "${SERVICE_DB_MAP[@]}"; do
    IFS=':' read -r db role_name password_var <<< "$entry"
    role_password="${!password_var:-}"

    if [ -z "$role_password" ]; then
        continue
    fi

    revoke_cross_db_access "$role_name" "$db"
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
