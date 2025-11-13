#!/bin/bash
set -e

echo "=== MetadataStampingService Container Starting ===" >&2
echo "Working directory: $(pwd)" >&2
echo "PYTHONPATH: $PYTHONPATH" >&2
echo "Python version: $(python --version)" >&2
echo "User: $(whoami)" >&2

# List installed packages
echo "Checking key dependencies..." >&2
python -c "import fastapi; print('  fastapi:', fastapi.__version__)" 2>&1 || echo "  fastapi: MISSING!" >&2
python -c "import uvicorn; print('  uvicorn:', uvicorn.__version__)" 2>&1 || echo "  uvicorn: MISSING!" >&2
python -c "import asyncpg; print('  asyncpg:', asyncpg.__version__)" 2>&1 || echo "  asyncpg: MISSING!" >&2
python -c "import pydantic; print('  pydantic:', pydantic.__version__)" 2>&1 || echo "  pydantic: MISSING!" >&2

# Try to import the service module
echo "Testing module import..." >&2
python -c "from omninode_bridge.services.metadata_stamping import main; print('  Module import: SUCCESS')" 2>&1 || {
    echo "  Module import: FAILED!" >&2
    echo "Attempting to show import error:" >&2
    python -c "from omninode_bridge.services.metadata_stamping import main" 2>&1 || true
    exit 1
}

# If we got here, start the service
echo "Starting MetadataStampingService..." >&2
exec python -m omninode_bridge.services.metadata_stamping.main "$@"
