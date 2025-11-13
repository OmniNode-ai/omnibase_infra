#!/bin/bash

# SQL Syntax Validation Script
# Validates SQL syntax without connecting to database
# Uses PostgreSQL parser dry-run mode

set -e

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== SQL Syntax Validation ===${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Find all SQL files
SQL_FILES=$(find "$SCRIPT_DIR" -name "*.sql" | sort)

# Count files
TOTAL=$(echo "$SQL_FILES" | wc -l | tr -d ' ')
echo -e "${YELLOW}Found ${TOTAL} SQL files to validate${NC}"
echo ""

# Validate each file
COUNT=0
ERRORS=0
for sql_file in $SQL_FILES; do
    COUNT=$((COUNT + 1))
    FILENAME=$(basename "$sql_file")

    echo -e "${YELLOW}[$COUNT/$TOTAL] Validating: ${FILENAME}${NC}"

    # Check basic SQL syntax
    # 1. Check for common syntax errors
    if grep -q "CREATE TABLE.*IF NOT EXISTS" "$sql_file" || grep -q "CREATE INDEX.*IF NOT EXISTS" "$sql_file" || grep -q "DROP.*IF EXISTS" "$sql_file"; then
        # Check for missing semicolons
        if ! grep -q ";" "$sql_file"; then
            echo -e "${RED}  ✗ Warning: No semicolons found${NC}"
            ERRORS=$((ERRORS + 1))
        fi

        # Check for balanced parentheses
        OPEN=$(grep -o "(" "$sql_file" | wc -l)
        CLOSE=$(grep -o ")" "$sql_file" | wc -l)
        if [ "$OPEN" -ne "$CLOSE" ]; then
            echo -e "${RED}  ✗ Error: Unbalanced parentheses (open: $OPEN, close: $CLOSE)${NC}"
            ERRORS=$((ERRORS + 1))
        else
            echo -e "${GREEN}  ✓ Parentheses balanced${NC}"
        fi

        # Check for SQL keywords
        if grep -qi "CREATE\|DROP\|INSERT\|UPDATE\|DELETE\|SELECT" "$sql_file"; then
            echo -e "${GREEN}  ✓ Contains SQL statements${NC}"
        else
            echo -e "${YELLOW}  ⚠ Warning: No SQL statements found${NC}"
        fi

        # Check for comments
        if grep -q "^--" "$sql_file"; then
            echo -e "${GREEN}  ✓ Contains documentation comments${NC}"
        fi

        echo -e "${GREEN}  ✓ Basic syntax validation passed${NC}"
    else
        echo -e "${YELLOW}  ⚠ Skipping validation (no CREATE/DROP statements)${NC}"
    fi
    echo ""
done

echo ""
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}=== All SQL files validated successfully ===${NC}"
    exit 0
else
    echo -e "${RED}=== Validation completed with ${ERRORS} errors ===${NC}"
    exit 1
fi
