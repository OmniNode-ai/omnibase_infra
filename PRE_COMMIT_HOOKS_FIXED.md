# Pre-Commit Hooks Fixed - Root Cause Analysis

## Problem Statement

Pre-commit hooks were **not installed** in `.git/hooks/`, causing linting failures to reach CI when they should have been caught locally.

## Root Cause

**Configuration existed** (`.pre-commit-config.yaml`) but hooks were **never installed** in the git repository.

- ✅ Configuration: `/workspace/omnibase_infra2/.pre-commit-config.yaml` (mirrors omnibase_core)
- ❌ Installation: `git/hooks/pre-commit` and `.git/hooks/pre-push` were NOT present
- ❌ Result: No validation ran before commits/pushes

## Fix Applied

```bash
poetry run pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
```

**Verification:**
```bash
ls -la .git/hooks/ | grep -E "pre-commit|pre-push"
-rwxr-xr-x  1 jonah dialout  663 Dec 16 16:24 pre-commit
-rwxr-xr-x  1 jonah dialout  661 Dec 16 16:24 pre-push
```

## Underlying Tech Debt Revealed

Running pre-commit exposed **108 Union type violations** (max allowed: 30):

```
src/omnibase_infra/mixins/mixin_async_circuit_breaker.py: Line 273, 370
src/omnibase_infra/runtime/*.py: 23 violations
src/omnibase_infra/errors/*.py: 14 violations
src/omnibase_infra/event_bus/kafka_event_bus.py: 16 violations
src/omnibase_infra/handlers/*.py: 7 violations
... and 48 more across the codebase
```

**This is PRE-EXISTING tech debt**, not introduced by PR #37 fixes.

## Required Next Steps

### Option 1: Fix Union Violations (Recommended)
Convert all `Union[X, None]` → `Optional[X]` across the codebase

**Scope:** 108 violations across 15+ files
**Effort:** ~30-60 minutes with automated fixes
**Benefit:** Aligns with omnibase_core standards

### Option 2: Increase Threshold Temporarily
Update `.pre-commit-config.yaml` to allow higher threshold temporarily:

```yaml
- id: onex-validate-unions
  args: ['--allow-invalid', '108']  # Current baseline
```

**Benefit:** Unblock commits immediately
**Drawback:** Allows tech debt to persist

### Option 3: Disable Union Validation Temporarily
Comment out union validation hook entirely (NOT RECOMMENDED)

## Recommended Action

**Run automated fix for Union types:**
1. Use polymorphic agent to fix all 108 violations
2. Re-run pre-commit to verify
3. Commit with proper hooks enabled
4. Future commits will enforce Union standards

## Prevention

✅ **Hooks now installed** - will run automatically on every commit/push
✅ **Aligned with omnibase_core** - same validation standards
✅ **Two-stage validation**:
- Pre-commit: Fast formatting and basic checks
- Pre-push: Type checking and deep validation

## Comparison with omnibase_core

| Aspect | omnibase_core | omnibase_infra (before) | omnibase_infra (now) |
|--------|---------------|-------------------------|----------------------|
| Hooks installed? | ✅ Yes | ❌ **NO** | ✅ **YES** |
| Union threshold | 160 | 30 | 30 |
| Union violations | ~160 (tracked) | 108 (unknown!) | 108 (now visible) |
| Pre-commit runs? | ✅ Every commit | ❌ Never | ✅ Every commit |

## Files That Need Union Fixes

High priority (most violations):
1. `src/omnibase_infra/runtime/runtime_host_process.py` (6 violations)
2. `src/omnibase_infra/runtime/handler_registry.py` (5 violations)
3. `src/omnibase_infra/errors/infra_errors.py` (14 violations)
4. `src/omnibase_infra/event_bus/kafka_event_bus.py` (16 violations)
5. `src/omnibase_infra/event_bus/models/model_event_headers.py` (9 violations)

Total: 108 violations across 15+ files
