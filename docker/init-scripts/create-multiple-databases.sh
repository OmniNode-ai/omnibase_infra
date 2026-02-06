#!/bin/bash
# create-multiple-databases.sh
# PostgreSQL initialization script to create multiple databases
#
# This script is executed during PostgreSQL container initialization
# (only when the data directory is empty, i.e., first startup).
#
# Usage:
#   Set POSTGRES_MULTIPLE_DATABASES environment variable to a comma-separated
#   list of database names to create in addition to the default POSTGRES_DB.
#
# Example:
#   POSTGRES_MULTIPLE_DATABASES="omninode_bridge,infisical_db"
#
# Note: The default database (POSTGRES_DB) is created automatically by the
# official PostgreSQL image, so it doesn't need to be in this list.

set -e
set -u

# Function to create a database if it doesn't exist
create_database() {
    local database="$1"

    # Validate database name: only alphanumeric, underscore, and hyphen allowed
    if ! echo "$database" | grep -qE '^[a-zA-Z_][a-zA-Z0-9_-]*$'; then
        echo "ERROR: Invalid database name '$database' - must match ^[a-zA-Z_][a-zA-Z0-9_-]*$" >&2
        return 1
    fi

    echo "Creating database: $database"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        SELECT 'CREATE DATABASE "$database"'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
EOSQL
    echo "Database '$database' created or already exists."
}

# Main script
if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
    echo "Multiple database creation requested: $POSTGRES_MULTIPLE_DATABASES"

    # Split the comma-separated list and create each database
    IFS=',' read -ra DATABASES <<< "$POSTGRES_MULTIPLE_DATABASES"
    for db in "${DATABASES[@]}"; do
        # Trim whitespace
        db=$(echo "$db" | xargs)
        if [ -n "$db" ]; then
            # Skip if it's the same as the default database (already created)
            if [ "$db" != "$POSTGRES_DB" ]; then
                create_database "$db"
            else
                echo "Skipping '$db' - already created as default database."
            fi
        fi
    done

    echo "All databases created successfully."
else
    echo "POSTGRES_MULTIPLE_DATABASES not set - only default database will be created."
fi

# List all databases for verification
echo "Available databases:"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "\l"
