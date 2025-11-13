#!/bin/bash

# Migration Rollback Script
# Rolls back all migrations in reverse order
# Usage: ./rollback_migrations.sh [database_url]

set -e  # Exit on error

# Database connection parameters
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-omninode_bridge}"
DB_USER="${POSTGRES_USER:-postgres}"

# Override with command line argument if provided
DB_URL="${1:-}"

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}=== OmniNode Bridge Database Rollback ===${NC}"
echo "Database: ${DB_NAME}"
echo "Host: ${DB_HOST}:${DB_PORT}"
echo "User: ${DB_USER}"
echo ""

# Confirm rollback
echo -e "${YELLOW}WARNING: This will drop all tables and data!${NC}"
read -p "Are you sure you want to rollback? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Rollback cancelled."
    exit 0
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find all rollback files and sort in REVERSE order
ROLLBACKS=$(find "$SCRIPT_DIR" -name "*rollback*.sql" | sort -r)

# Count rollbacks
TOTAL=$(echo "$ROLLBACKS" | wc -l | tr -d ' ')
echo -e "${YELLOW}Found ${TOTAL} rollback scripts to run${NC}"
echo ""

# Run each rollback
COUNT=0
for rollback in $ROLLBACKS; do
    COUNT=$((COUNT + 1))
    FILENAME=$(basename "$rollback")

    echo -e "${RED}[$COUNT/$TOTAL] Running: ${FILENAME}${NC}"

    if [ -z "$DB_URL" ]; then
        # Use connection parameters
        PGPASSWORD="${POSTGRES_PASSWORD}" psql \
            -h "$DB_HOST" \
            -p "$DB_PORT" \
            -U "$DB_USER" \
            -d "$DB_NAME" \
            -f "$rollback" \
            -v ON_ERROR_STOP=1 \
            --quiet
    else
        # Use connection URL
        psql "$DB_URL" -f "$rollback" -v ON_ERROR_STOP=1 --quiet
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        exit 1
    fi
    echo ""
done

echo -e "${GREEN}=== All rollbacks completed successfully ===${NC}"

# Verify tables dropped
echo ""
echo -e "${YELLOW}Verifying schema...${NC}"
if [ -z "$DB_URL" ]; then
    PGPASSWORD="${POSTGRES_PASSWORD}" psql \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        -c "\dt"
else
    psql "$DB_URL" -c "\dt"
fi
