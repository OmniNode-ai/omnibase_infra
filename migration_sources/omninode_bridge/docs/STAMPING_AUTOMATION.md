# Stamping & OnexTree Ingestion Automation

**Status**: ✅ Implemented (October 2025)
**Tool**: `scripts/stamp_and_ingest.py`

## Overview

Automatic file stamping and OnexTree intelligence ingestion tool for the omninode_bridge repository. This tool provides:

- **Universal File Support**: Stamps any text-based file (Python, docs, configs, etc.)
- **Flexible Modes**: Pre-push hook, manual stamping, directory scanning, repository backfill
- **Graceful Degradation**: Works even when services are unavailable
- **Performance**: < 5s for typical pushes (5-10 files), < 15s for large pushes (50+ files)

---

## Features

### 1. Pre-Push Hook (Default)

Automatically stamps changed files before git push.

```bash
# Runs automatically on git push
git push origin feature-branch

# Or manually test the pre-push hook
poetry run python scripts/stamp_and_ingest.py
```

**What it does:**
- Detects files changed since last push to `origin/main`
- Generates BLAKE3 hashes for each file
- Creates metadata stamps with O.N.E. v0.1 compliance
- Ingests code patterns into OnexTree for intelligence
- **Non-blocking**: Always allows push even if stamping fails

---

### 2. Manual File Stamping

Stamp specific files on demand.

```bash
# Stamp a single file
poetry run python scripts/stamp_and_ingest.py --file src/module.py

# Stamp a Python file
poetry run python scripts/stamp_and_ingest.py --file src/omninode_bridge/clients/metadata_stamping_client.py

# Stamp a markdown doc
poetry run python scripts/stamp_and_ingest.py --file docs/ARCHITECTURE.md

# Stamp a YAML config
poetry run python scripts/stamp_and_ingest.py --file deployment/docker-compose.yml
```

---

### 3. Directory Stamping

Stamp all files in a directory.

```bash
# Stamp directory (non-recursive)
poetry run python scripts/stamp_and_ingest.py --directory src/omninode_bridge/clients/

# Stamp directory recursively
poetry run python scripts/stamp_and_ingest.py --directory src/ --recursive

# Stamp with pattern filter (docs only)
poetry run python scripts/stamp_and_ingest.py --directory docs/ --recursive --pattern "*.md"

# Stamp Python files only
poetry run python scripts/stamp_and_ingest.py --directory src/ --recursive --pattern "*.py"

# Stamp configs only
poetry run python scripts/stamp_and_ingest.py --directory deployment/ --recursive --pattern "*.yml"
```

---

### 4. Repository Backfill

Stamp all stampable files in the repository (for historical data ingestion).

```bash
# Backfill entire repository
poetry run python scripts/stamp_and_ingest.py --backfill

# This will:
# - Recursively scan the entire repository
# - Skip binary files, hidden files, and common directories (.git, node_modules, etc.)
# - Stamp all text-based files
# - Ingest patterns into OnexTree
```

**Use cases:**
- Initial setup of stamping system
- Migrating to new stamping infrastructure
- Rebuilding OnexTree intelligence after data loss

---

## Supported File Types

### Text Files (Stamped)

**Source Code:**
- `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.c`, `.cpp`, `.h`, `.hpp`
- `.rs`, `.go`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`

**Configuration:**
- `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.conf`

**Documentation:**
- `.md`, `.rst`, `.txt`, `.adoc`

**Web:**
- `.html`, `.htm`, `.css`, `.scss`, `.sass`, `.xml`

**Shell:**
- `.sh`, `.bash`, `.zsh`, `.fish`

**Other:**
- `.sql`, `.graphql`, `.proto`, `.thrift`

### Binary Files (Skipped)

- Images: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.ico`, `.webp`, `.svg`
- Archives: `.zip`, `.tar`, `.gz`, `.bz2`, `.xz`, `.7z`, `.rar`
- Executables: `.exe`, `.dll`, `.so`, `.dylib`
- Media: `.mp4`, `.avi`, `.mov`, `.mp3`, `.wav`, `.flac`
- Databases: `.db`, `.sqlite`, `.sqlite3`
- Other: `.pdf`, `.pyc`, `.pyo`, `.whl`

---

## Configuration

### Environment Variables

```bash
# Service URLs
export STAMPING_SERVICE_URL="http://localhost:8053"
export ONEXTREE_SERVICE_URL="http://localhost:8054"

# Disable stamping (for testing)
export SKIP_STAMPING="true"

# Disable OnexTree ingestion (for testing)
export SKIP_ONEXTREE="true"
```

### Command-Line Options

```bash
# Stamp with custom namespace
poetry run python scripts/stamp_and_ingest.py \
  --file src/module.py \
  --namespace "omninode.bridge.experimental"

# Skip stamping (dry run)
poetry run python scripts/stamp_and_ingest.py \
  --directory src/ --recursive \
  --skip-stamping --skip-onextree
```

---

## Pre-Commit Integration

The tool is integrated into `.pre-commit-config.yaml` as a pre-push hook:

```yaml
# Pre-push hooks for comprehensive testing and automation
- repo: local
  hooks:
    # Automatic stamping and OnexTree ingestion
    - id: stamp-and-ingest
      name: Stamp changed files and ingest to OnexTree (pre-push)
      entry: poetry run python scripts/stamp_and_ingest.py
      language: system
      stages: [push]
      pass_filenames: false
      verbose: true
```

**Installation:**

```bash
# Install pre-commit hooks
pre-commit install --hook-type pre-push

# Test pre-push hook
pre-commit run stamp-and-ingest --hook-stage push
```

---

## Service Requirements

### MetadataStamping Service (Port 8053)

**Status**: Required for stamping
**Degradation**: Skips stamping if unavailable (non-blocking)

```bash
# Start MetadataStamping service
docker compose -f deployment/docker-compose.yml up -d metadata-stamping

# Check health
curl http://localhost:8053/health
```

### OnexTree Service (Port 8054)

**Status**: Optional for intelligence ingestion
**Degradation**: Skips ingestion if unavailable (non-blocking)

```bash
# Start OnexTree service
docker compose -f deployment/docker-compose.yml up -d onextree

# Check health
curl http://localhost:8054/health
```

---

## Performance

### Benchmarks

| Scenario | Files | Time | Throughput |
|----------|-------|------|------------|
| Typical push | 5-10 files | <5s | ~2 files/sec |
| Large push | 50 files | <15s | ~3.3 files/sec |
| Directory scan | 100 files | <30s | ~3.3 files/sec |
| Full backfill | 500+ files | <3 min | ~3 files/sec |

### Optimization Tips

1. **Skip OnexTree for large backfills:**
   ```bash
   poetry run python scripts/stamp_and_ingest.py --backfill --skip-onextree
   ```

2. **Use pattern filters to reduce scope:**
   ```bash
   poetry run python scripts/stamp_and_ingest.py \
     --directory src/ --recursive --pattern "*.py"
   ```

3. **Run during off-hours for full backfills:**
   ```bash
   # Background job
   nohup poetry run python scripts/stamp_and_ingest.py --backfill > backfill.log 2>&1 &
   ```

---

## Output Examples

### Single File

```bash
$ poetry run python scripts/stamp_and_ingest.py --file README.md

Processing 1 files...
Processing 1/1 files...
============================================================
File Stamping & Ingestion Summary
============================================================
Total files processed:    1
Successfully stamped:     1
Already stamped:          0
Skipped:                  0
Failed:                   0
============================================================
```

### Directory (Recursive)

```bash
$ poetry run python scripts/stamp_and_ingest.py --directory docs/ --recursive --pattern "*.md"

Processing 17 files...
Processing 10/17 files...Processing 17/17 files...
============================================================
File Stamping & Ingestion Summary
============================================================
Total files processed:    17
Successfully stamped:     12
Already stamped:          5
Skipped:                  0
Failed:                   0
============================================================
```

### Backfill

```bash
$ poetry run python scripts/stamp_and_ingest.py --backfill

Processing 523 files...
Processing 100/523 files...Processing 200/523 files...Processing 523/523 files...
============================================================
File Stamping & Ingestion Summary
============================================================
Total files processed:    523
Successfully stamped:     487
Already stamped:          32
Skipped:                  4
Failed:                   0
============================================================
```

---

## Troubleshooting

### Services Unavailable

**Problem**: MetadataStamping or OnexTree services are not running

**Solution**:
```bash
# Check service status
docker compose -f deployment/docker-compose.yml ps

# Start services
docker compose -f deployment/docker-compose.yml up -d

# Check logs
docker compose -f deployment/docker-compose.yml logs metadata-stamping
docker compose -f deployment/docker-compose.yml logs onextree
```

### No Files Detected

**Problem**: Script reports "No files to process"

**Possible causes:**
- All files are already stamped (check with `--force` if that flag is added)
- Files are binary (check file types in summary)
- No changes since last push (expected for pre-push mode)

### Permission Errors

**Problem**: Cannot read certain files

**Solution**:
```bash
# Check file permissions
ls -la path/to/file

# Fix permissions if needed
chmod 644 path/to/file
```

---

## Future Enhancements

### Planned Features

- [ ] **Force re-stamping**: `--force` flag to re-stamp already stamped files
- [ ] **Parallel processing**: Stamp multiple files concurrently
- [ ] **Diff-based stamping**: Only stamp changed portions of files
- [ ] **Cache optimization**: Local cache for stamp lookups
- [ ] **Progress bar**: Rich progress bar for large operations
- [ ] **Summary export**: Export stamping report to JSON/CSV

### Integration Opportunities

- [ ] **CI/CD Integration**: Automatic stamping in GitHub Actions
- [ ] **IDE Integration**: VS Code extension for on-save stamping
- [ ] **Git Hooks**: Post-commit hook for immediate stamping
- [ ] **Monitoring**: Grafana dashboard for stamping metrics

---

## References

- **MetadataStamping API**: [docs/api/API_REFERENCE.md](./api/API_REFERENCE.md)
- **OnexTree Intelligence**: [docs/ONEXTREE_INTEGRATION.md](./design/ONEXTREE_INTEGRATION.md)
- **Pre-Commit Hooks**: [docs/design/PRE_COMMIT_HOOKS.md](./design/PRE_COMMIT_HOOKS.md)
- **O.N.E. v0.1 Compliance**: [docs/validation/ONEX_VALIDATION_REPORT.md](./validation/ONEX_VALIDATION_REPORT.md)

---

**Last Updated**: October 24, 2025
**Maintainer**: OmniNode Bridge Team
