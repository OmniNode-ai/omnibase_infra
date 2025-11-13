#!/bin/bash
#
# Entrypoint script for adapter nodes (LLM Effect, Vault Effect)
# Verifies omnibase_core is properly installed
#

set -e

# Function to verify omnibase_core installation
verify_omnibase_core() {
    if ! python -c "import omnibase_core" 2>/dev/null; then
        echo "ERROR: omnibase_core is not installed!"
        echo "This should have been installed via Poetry from Git dependency."
        echo "Check that the Dockerfile includes 'git' in build dependencies."
        exit 1
    fi

    # Verify it's the real package, not stubs
    OMNIBASE_PATH=$(python -c "import omnibase_core; print(omnibase_core.__file__)")
    if [[ "$OMNIBASE_PATH" == *"/app/src/omnibase_core"* ]]; then
        echo "ERROR: Using stub omnibase_core from src/ instead of installed package!"
        echo "Location: $OMNIBASE_PATH"
        exit 1
    fi

    echo "✓ omnibase_core verified: $OMNIBASE_PATH"
}

# Main logic
echo "Starting adapter node..."

# Verify omnibase_core is properly installed
verify_omnibase_core

# Execute the command
echo "Running command: $@"
exec "$@"
