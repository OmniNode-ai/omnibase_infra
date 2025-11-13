# Dependency Strategy - omninode_bridge

## omnibase_core Dependency

### Current Configuration

**Branch**: `feature/comprehensive-onex-cleanup`
**Repository**: `https://github.com/OmniNode-ai/omnibase_core.git`
**Installation Method**: Git dependency via pyproject.toml

```toml
[tool.poetry.dependencies]
omnibase_core = {git = "https://github.com/OmniNode-ai/omnibase_core.git", branch = "feature/comprehensive-onex-cleanup"}
```

### Rationale

This project uses the `feature/comprehensive-onex-cleanup` branch of omnibase_core for development because:

1. **ONEX v2.0 Compliance**: Contains the latest ONEX v2.0 architecture patterns and base classes
2. **Active Development**: This branch contains updates required for LlamaIndex Workflows integration
3. **Bridge Node Support**: Includes NodeOrchestrator and NodeReducer base classes with proper contract support
4. **Development Coordination**: Allows synchronized development between omnibase_core and omninode_bridge

### Migration to Stable Release

**When `feature/comprehensive-onex-cleanup` is merged to main:**

1. Update pyproject.toml to use main branch or tagged release:
   ```toml
   omnibase_core = {git = "https://github.com/OmniNode-ai/omnibase_core.git", tag = "v2.0.0"}
   ```

2. Or for production deployments, use PyPI release:
   ```toml
   omnibase_core = "^2.0.0"
   ```

3. Run dependency update:
   ```bash
   poetry update omnibase_core
   poetry lock
   ```

### Fallback Strategy for Missing Modules

**Issue**: Some tests reference `utility_reference_resolver` module which may not exist in current omnibase_core branch.

**Current Handling**:
- Tests that import unavailable modules should gracefully skip with `pytest.mark.skipif`
- Use `pytest.importorskip()` for conditional imports in test files
- Document expected modules in test docstrings

**Example**:
```python
pytest.importorskip("omnibase_core.utility_reference_resolver",
                    reason="utility_reference_resolver not available in current omnibase_core branch")
```

### CI Configuration

**GitHub Actions**: CI uses the same git dependency configuration via Poetry's automatic authentication.

**Authentication**:
- CI authenticates using `GH_PAT` secret for private repository access
- Local development uses personal GitHub credentials via Poetry

**Lock File**:
- `poetry.lock` pins exact commit hashes for reproducible builds
- Update lock file with `poetry lock --no-update` after branch changes

### Monitoring Dependencies

**Check for updates**:
```bash
# Check if upstream branch has new commits
git ls-remote https://github.com/OmniNode-ai/omnibase_core.git feature/comprehensive-onex-cleanup

# Update to latest commit on branch
poetry update omnibase_core
```

**Verify installation**:
```bash
# Show installed version
poetry show omnibase_core

# Check for available modules
python -c "import omnibase_core; print(omnibase_core.__file__)"
```

### Breaking Changes Mitigation

**If omnibase_core introduces breaking changes:**

1. **Pin to Known Working Commit**:
   ```toml
   omnibase_core = {git = "https://github.com/OmniNode-ai/omnibase_core.git", rev = "abc123def"}  # pragma: allowlist secret
   ```

2. **Create Compatibility Shims**:
   ```python
   # src/omninode_bridge/compat/omnibase_compat.py
   try:
       from omnibase_core.new_module import NewClass
   except ImportError:
       from omnibase_core.old_module import OldClass as NewClass
   ```

3. **Document Required Version**:
   - Update this file with minimum required commit hash
   - Add version compatibility matrix in README.md

### Testing Strategy

**Unit Tests**:
- Mock omnibase_core dependencies where possible
- Use `pytest.importorskip()` for optional dependencies
- Skip tests that require unavailable modules

**Integration Tests**:
- Require full omnibase_core installation
- Document required modules in test suite documentation
- Use `pytest.mark.requires_infrastructure` for tests needing complete omnibase_core

### Production Deployment Recommendations

1. **Use Tagged Releases**: Never deploy with branch dependencies in production
2. **Verify Compatibility**: Run full test suite before deploying
3. **Pin Exact Versions**: Use `poetry.lock` for reproducible deployments
4. **Monitor Updates**: Subscribe to omnibase_core releases for security updates
5. **Fallback Plan**: Keep previous working version available for rollback

### Support and Issues

**If you encounter dependency issues:**

1. Check if omnibase_core branch still exists: `git ls-remote`
2. Verify authentication is working: `poetry update --dry-run`
3. Clear cache if needed: `poetry cache clear pypi --all`
4. Review Poetry lock file: `poetry lock --no-update`
5. Open issue with details at: https://github.com/OmniNode-ai/omninode_bridge/issues

### Changelog

| Date | Change | Reason |
|------|--------|--------|
| 2025-01-13 | Initial documentation | Document dependency strategy for PR #24 |
| 2025-01-13 | Using feature/comprehensive-onex-cleanup | Required for ONEX v2.0 and LlamaIndex Workflows integration |
