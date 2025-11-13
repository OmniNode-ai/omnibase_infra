#!/bin/bash
# ONEX v2.0 Node Type Validation Script
# Validates NODE_TYPE environment variable before container startup
# Exit codes: 0 = success, 1 = validation failure, 2 = configuration error

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to print error messages
error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

# Function to print warning messages
warning() {
    echo -e "${YELLOW}WARNING: $1${NC}" >&2
}

# Function to print success messages
success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

# Validate NODE_TYPE is set and non-empty
if [ -z "${NODE_TYPE:-}" ]; then
    error "NODE_TYPE environment variable is not set"
    echo "" >&2
    echo "REQUIRED: Set NODE_TYPE to specify which node to run" >&2
    echo "" >&2
    echo "Valid values by Dockerfile:" >&2
    echo "  - Dockerfile.generic-effect:" >&2
    echo "      database_adapter_effect, store_effect" >&2
    echo "  - Dockerfile.generic-orchestrator:" >&2
    echo "      codegen_orchestrator, workflow_orchestrator" >&2
    echo "  - Dockerfile.generic-reducer:" >&2
    echo "      codegen_metrics_reducer, aggregation_reducer" >&2
    echo "" >&2
    echo "Example docker-compose.yml configuration:" >&2
    echo "  environment:" >&2
    echo "    - NODE_TYPE=database_adapter_effect" >&2
    echo "" >&2
    echo "Example docker run command:" >&2
    echo "  docker run -e NODE_TYPE=database_adapter_effect <image>" >&2
    echo "" >&2
    exit 1
fi

# Validate NODE_TYPE contains only valid characters (alphanumeric and underscore)
if ! [[ "${NODE_TYPE}" =~ ^[a-zA-Z0-9_]+$ ]]; then
    error "NODE_TYPE contains invalid characters: '${NODE_TYPE}'"
    echo "" >&2
    echo "NODE_TYPE must contain only alphanumeric characters and underscores" >&2
    echo "Current value: ${NODE_TYPE}" >&2
    exit 1
fi

# Validate NODE_TYPE length (reasonable bounds)
if [ ${#NODE_TYPE} -lt 3 ]; then
    error "NODE_TYPE is too short: '${NODE_TYPE}'"
    echo "" >&2
    echo "NODE_TYPE must be at least 3 characters long" >&2
    exit 1
fi

if [ ${#NODE_TYPE} -gt 100 ]; then
    error "NODE_TYPE is too long: '${NODE_TYPE}'"
    echo "" >&2
    echo "NODE_TYPE must be at most 100 characters long" >&2
    exit 1
fi

# Check if the node module path exists (if running validation after source copy)
NODE_MODULE_PATH="/app/src/omninode_bridge/nodes/${NODE_TYPE}/v1_0_0/node.py"
if [ -f "${NODE_MODULE_PATH}" ]; then
    success "NODE_TYPE validated: '${NODE_TYPE}' (module found at ${NODE_MODULE_PATH})"
elif [ -d "/app/src/omninode_bridge/nodes" ]; then
    # Directory exists but module not found - warn but don't fail
    # (module might be created later or mounted)
    warning "NODE_TYPE set to '${NODE_TYPE}' but module not found at ${NODE_MODULE_PATH}"
    echo "  This may be expected if the module will be mounted or created later" >&2
    echo "  Available nodes:" >&2
    find /app/src/omninode_bridge/nodes -maxdepth 1 -type d -name "*_*" | sed 's|/app/src/omninode_bridge/nodes/||' | sed 's/^/    - /' >&2
else
    # Source not copied yet (early validation during build)
    success "NODE_TYPE validated: '${NODE_TYPE}' (source validation skipped)"
fi

exit 0
