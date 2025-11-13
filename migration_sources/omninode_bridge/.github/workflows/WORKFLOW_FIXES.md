# CI Workflow Required Fixes

## Priority 1: docker-build.yml - Critical Issues

### Fix 1: Correct Import Paths in Image Tests
**Lines 92-95, 165-168**

```yaml
# WRONG - Current
python -c "from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeOrchestratorV1; print('✅ Orchestrator import successful')"

# CORRECT - Should be
python -c "from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator; print('✅ Orchestrator import successful')"
```

### Fix 2: Use Local Images for Testing
**Lines 231-242 (image-analysis job)**

```yaml
# WRONG - Tries to pull unpushed images
- name: Analyze orchestrator image size
  run: |
    docker pull ${{ env.REGISTRY }}/${{ github.repository_owner }}/${{ env.IMAGE_PREFIX }}-orchestrator:${{ github.sha }}

# CORRECT - Use locally built images
- name: Analyze orchestrator image size
  run: |
    # Use local image tag from build job
    ORCH_SIZE=$(docker image inspect ${IMAGE_TAG_ORCHESTRATOR} --format='{{.Size}}')
```

### Fix 3: Fix Trivy Scan for Non-Pushed Images
**Lines 195-208**

```yaml
# Add conditional to only scan pushed images OR use local images
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    # Use local image if not pushed
    image-ref: ${{ github.ref == 'refs/heads/main' && format('{0}/{1}/{2}-{3}:{4}', env.REGISTRY, github.repository_owner, env.IMAGE_PREFIX, matrix.image, github.sha) || format('local-{0}', matrix.image) }}
```

## Priority 2: test-bridge-nodes.yml

### Fix 1: Remove Docker Image Tests
**Lines 89-95 should be REMOVED** - These belong in docker-build.yml

### Fix 2: Verify Integration Test File Exists
**Add check before line 203:**

```yaml
- name: Check integration test exists
  run: |
    if [ ! -f tests/integration/test_orchestrator_reducer_flow.py ]; then
      echo "⚠️ Integration test file not found - skipping"
      exit 0
    fi
```

## Priority 3: lint-and-type-check.yml

### Fix 1: Add Bandit to Dependencies
**pyproject.toml needs:**

```toml
[tool.poetry.group.dev.dependencies]
bandit = "^1.7.9"
```

### Fix 2: Add File Existence Checks
**Before line 291:**

```yaml
- name: Validate ONEX naming conventions
  run: |
    if [ ! -f src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py ]; then
      echo "⚠️ Orchestrator node file not found - skipping validation"
      exit 0
    fi

    if grep -r "class.*Orchestrator" src/omninode_bridge/nodes/orchestrator/v1_0_0/node.py; then
      echo "✅ Orchestrator node naming convention validated"
    fi
```

## Testing the Fixes

### 1. Test YAML Syntax
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/docker-build.yml'))"
```

### 2. Test Locally
```bash
# Run lint checks
poetry run black --check src/omninode_bridge/nodes/
poetry run ruff check src/omninode_bridge/nodes/

# Run unit tests
poetry run pytest tests/unit/nodes/ -v
```

### 3. Test Docker Build
```bash
# Build images locally
docker build -f docker/bridge-nodes/Dockerfile.orchestrator -t test-orchestrator .
docker build -f docker/bridge-nodes/Dockerfile.reducer -t test-reducer .

# Test imports
docker run --rm test-orchestrator python -c "from omninode_bridge.nodes.orchestrator.v1_0_0.node import NodeBridgeOrchestrator; print('OK')"
docker run --rm test-reducer python -c "from omninode_bridge.nodes.reducer.v1_0_0.node import NodeBridgeReducer; print('OK')"
```

## Recommended Workflow Improvements

### 1. Add Job Dependencies
Ensure proper execution order:

```yaml
jobs:
  lint:
    name: Code Quality
    runs-on: ubuntu-latest

  unit-tests:
    needs: [lint]

  integration-tests:
    needs: [unit-tests]

  docker-build:
    needs: [integration-tests]
```

### 2. Add Workflow Dispatch Testing
All workflows should have:

```yaml
on:
  workflow_dispatch:
    inputs:
      debug_enabled:
        description: 'Enable debug mode'
        required: false
        default: false
```

### 3. Add Matrix Strategy for Python Versions
Currently only testing Python 3.12, should test:

```yaml
strategy:
  matrix:
    python-version: ['3.11', '3.12', '3.13']
  fail-fast: false
```
