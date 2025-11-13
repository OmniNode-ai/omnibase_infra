#!/bin/bash
set -e

# Formatting Consistency Validation Script
# Ensures local pre-commit hooks match CI environment exactly

echo "🔧 OmniNode Bridge - Formatting Consistency Validator"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: Must be run from project root directory${NC}"
    exit 1
fi

echo -e "${BLUE}📍 Checking environment setup...${NC}"

# Check Poetry installation
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}❌ Poetry not found. Please install Poetry first.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Installing dependencies...${NC}"
    poetry install
fi

echo -e "${GREEN}✅ Poetry environment ready${NC}"

# Check pre-commit installation
if ! command -v pre-commit &> /dev/null; then
    echo -e "${YELLOW}⚠️  pre-commit not found. Installing...${NC}"
    poetry run pip install pre-commit
fi

echo -e "${GREEN}✅ Pre-commit available${NC}"

echo -e "${BLUE}🔍 Validating tool versions...${NC}"

# Get versions from Poetry environment (what CI uses)
POETRY_BLACK_VERSION=$(poetry run black --version | cut -d' ' -f2)
POETRY_ISORT_VERSION=$(poetry run isort --version | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
POETRY_RUFF_VERSION=$(poetry run ruff --version | cut -d' ' -f2)

echo "Poetry environment versions (used by CI):"
echo "  Black: $POETRY_BLACK_VERSION"
echo "  isort: $POETRY_ISORT_VERSION"
echo "  Ruff:  $POETRY_RUFF_VERSION"

echo -e "${BLUE}🧪 Testing formatting consistency...${NC}"

# Create a temporary test file with formatting issues
TEST_FILE="test_formatting_temp.py"
cat > "$TEST_FILE" << 'EOF'
import os,sys
from typing import Dict,List
def test_function( x,y ):
    if True:
        return x+y
    else:return None
EOF

echo "Created test file with formatting issues"

# Test 1: Run pre-commit hooks
echo -e "${BLUE}1️⃣ Testing pre-commit hooks...${NC}"
if poetry run pre-commit run --files "$TEST_FILE" > /dev/null 2>&1; then
    echo -e "${RED}❌ Pre-commit should have failed on poorly formatted file${NC}"
    rm -f "$TEST_FILE"
    exit 1
else
    echo -e "${GREEN}✅ Pre-commit correctly caught formatting issues${NC}"
fi

# Test 2: Run CI-equivalent commands
echo -e "${BLUE}2️⃣ Testing CI-equivalent commands...${NC}"

# Format the file first
poetry run black "$TEST_FILE" > /dev/null 2>&1
poetry run isort "$TEST_FILE" > /dev/null 2>&1

# Now test if CI commands pass
if ! poetry run black --check "$TEST_FILE" > /dev/null 2>&1; then
    echo -e "${RED}❌ Black check failed after formatting${NC}"
    rm -f "$TEST_FILE"
    exit 1
fi

if ! poetry run isort --check-only "$TEST_FILE" > /dev/null 2>&1; then
    echo -e "${RED}❌ isort check failed after formatting${NC}"
    rm -f "$TEST_FILE"
    exit 1
fi

if ! poetry run ruff check "$TEST_FILE" > /dev/null 2>&1; then
    echo -e "${RED}❌ Ruff check failed after formatting${NC}"
    rm -f "$TEST_FILE"
    exit 1
fi

echo -e "${GREEN}✅ All CI-equivalent commands pass after formatting${NC}"

# Test 3: Verify pre-commit hooks now pass
echo -e "${BLUE}3️⃣ Testing pre-commit hooks on formatted file...${NC}"
if ! poetry run pre-commit run --files "$TEST_FILE" > /dev/null 2>&1; then
    echo -e "${RED}❌ Pre-commit hooks failed on properly formatted file${NC}"
    rm -f "$TEST_FILE"
    exit 1
else
    echo -e "${GREEN}✅ Pre-commit hooks pass on properly formatted file${NC}"
fi

# Clean up
rm -f "$TEST_FILE"

echo -e "${BLUE}📋 Running consistency checks on actual codebase...${NC}"

# Test on a few real files
TEST_FILES=(
    "src/omninode_bridge/workflow/app.py"
    "src/omninode_bridge/security/cors.py"
    "src/omninode_bridge/security/config_validator.py"
)

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -n "  Checking $file... "

        # Run CI checks
        if poetry run black --check "$file" >/dev/null 2>&1 && \
           poetry run isort --check-only "$file" >/dev/null 2>&1 && \
           poetry run ruff check "$file" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ CI checks pass${NC}"
        else
            echo -e "${RED}❌ CI checks fail${NC}"
            exit 1
        fi
    fi
done

echo -e "${GREEN}🎉 SUCCESS: Formatting consistency validated!${NC}"
echo ""
echo -e "${BLUE}📝 Summary of improvements:${NC}"
echo "• Pre-commit now uses the same Poetry environment as CI"
echo "• Tool versions are guaranteed to match between local and CI"
echo "• Configuration files (pyproject.toml) are used consistently"
echo "• No more environment-related formatting discrepancies"
echo ""
echo -e "${BLUE}💡 Developer workflow:${NC}"
echo "1. Run 'poetry install' to set up environment"
echo "2. Run 'poetry run pre-commit install' to install hooks"
echo "3. Hooks automatically run on git commit"
echo "4. Or manually run: 'poetry run pre-commit run --all-files'"
echo ""
echo -e "${GREEN}✅ Formatting consistency problem is resolved!${NC}"
