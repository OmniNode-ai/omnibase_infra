# Pre-Public Repository Audit Checklist

**Repository**: omnibase_infra
**Target**: Public development on GitHub
**License**: MIT
**Date**: 2025-12-15

---

## 🔒 Security & Secrets Audit

### Commit History Scan
- [ ] Run `git log --all --pretty=format: --name-only | sort -u` to list all files ever tracked
- [ ] Search commit history for secrets:
  ```bash
  git log -p | grep -i "password\|secret\|api_key\|token\|private_key"
  ```
- [ ] Check for `.env` files in history:
  ```bash
  git log --all --full-history -- "**/.env"
  ```
- [ ] Verify no AWS/GCP/Azure credentials in configs
- [ ] Check `docker-compose.yml` for hardcoded credentials

### Current Files Audit
- [ ] Review `.gitignore` - ensure it includes:
  - `.env`, `.env.*`
  - `*.key`, `*.pem`
  - `secrets/`, `credentials/`
  - `tmp/` (already tracked but verify no secrets)
- [ ] Scan for TODO/FIXME with sensitive info:
  ```bash
  grep -r "TODO.*password\|FIXME.*secret" src/
  ```
- [ ] Check `pyproject.toml` for internal registry URLs
- [ ] Verify Docker configs don't expose internal infrastructure

**Action if secrets found**: Use `git filter-repo` or BFG Repo-Cleaner to remove

---

## 📄 Documentation Readiness

### Essential Files
- [ ] **README.md** - Create comprehensive public README:
  - [ ] Project description and purpose
  - [ ] OmniNode ecosystem context (how it fits with core/spi)
  - [ ] Installation instructions (`poetry install`)
  - [ ] Quick start example
  - [ ] Architecture overview (link to CLAUDE.md for details)
  - [ ] Contribution guidelines (or link to CONTRIBUTING.md)
  - [ ] Link to other repos (omnibase_core, omnibase_spi)
  - [ ] Badge for build status (GitHub Actions)

- [ ] **LICENSE** - Add MIT license:
  ```bash
  # Add LICENSE file with MIT text and copyright holder
  ```

- [ ] **CONTRIBUTING.md** (optional but recommended):
  - [ ] How to set up development environment
  - [ ] How to run tests
  - [ ] Code style guidelines (ruff, mypy)
  - [ ] PR process
  - [ ] Reference to CLAUDE.md for ONEX standards

### Documentation Cleanup
- [ ] Review `CLAUDE.md` - is it appropriate for public?
  - [ ] Remove any internal team references
  - [ ] Keep ONEX principles (valuable for contributors)

- [ ] Check `docs/` directory (if exists):
  - [ ] Remove internal-only documentation
  - [ ] Ensure examples don't reference internal services

- [ ] Verify `tmp/HANDOFF_PR36.md` is in `.gitignore` (private handoff notes)

---

## 🧹 Code Quality & Cleanup

### Active Development
- [ ] **PR #36 (PolicyRegistry)** - Decision needed:
  - [ ] Option A: Merge before going public (cleaner)
  - [ ] Option B: Keep open as example of active development

- [ ] Rename `PolicyRegistry.list()` → `list_policies()`:
  - [ ] Update `src/omnibase_infra/runtime/policy_registry.py`
  - [ ] Update `tests/unit/runtime/test_policy_registry.py`
  - [ ] Remove `import builtins` hack
  - [ ] Update docstrings/examples

### Code Audit
- [ ] Search for embarrassing TODOs:
  ```bash
  grep -r "TODO\|FIXME\|HACK\|XXX" src/ tests/
  ```
  - [ ] Remove or clean up any that are too revealing

- [ ] Check for hardcoded paths:
  ```bash
  grep -r "/Users/\|/home/" src/ tests/
  ```

- [ ] Verify no internal hostnames/IPs in configs:
  ```bash
  grep -r "\.internal\|192\.168\|10\.\|172\." .
  ```

- [ ] Remove debug print statements:
  ```bash
  grep -r "print(" src/
  ```

### Tests
- [ ] All tests passing:
  ```bash
  poetry run pytest -v
  ```
- [ ] Coverage report shows reasonable coverage (>70% recommended)
- [ ] No skipped tests that should be fixed

### Validation
- [ ] All ONEX validators passing:
  ```bash
  poetry run python scripts/validate.py all --verbose
  ```
- [ ] Ruff formatting clean:
  ```bash
  poetry run ruff check . && poetry run ruff format --check .
  ```
- [ ] Mypy type checking clean:
  ```bash
  poetry run mypy src/omnibase_infra
  ```

---

## 🔗 Dependencies & Integration

### Dependency Repositories
- [x] **omnibase_core** status:
  - [x] Is it already public? **YES - on PyPI at 0.4.0**
  - [x] Update `pyproject.toml` if needed (no private git URLs) **DONE**

- [x] **omnibase_spi** status:
  - [x] Is it already public? **YES - on PyPI at 0.4.0**
  - [x] Update `pyproject.toml` if needed **DONE**

- [x] Verify all dependencies are from public registries (PyPI) **VERIFIED**

### Integration Points
- [ ] Document external service requirements:
  - [ ] Kafka (required for event bus)
  - [ ] PostgreSQL (required for data)
  - [ ] Consul (service discovery)
  - [ ] Vault (secrets management)

- [ ] Docker compose configurations:
  - [ ] Verify `docker-compose.yml` works with public images
  - [ ] Update image tags to use public registries
  - [ ] Document how to run the stack

---

## ⚙️ Repository Settings

### GitHub Settings (to configure after making public)
- [ ] Set repository description
- [ ] Add topics/tags: `python`, `onex`, `event-driven`, `architecture`
- [ ] Enable GitHub Actions (if not already)
- [ ] Configure branch protection for `main`:
  - [ ] Require PR reviews
  - [ ] Require status checks to pass
  - [ ] Require conversation resolution

- [ ] Set up GitHub Pages (optional):
  - [ ] Host generated docs from `docs/`

- [ ] Configure issue templates:
  - [ ] Bug report template
  - [ ] Feature request template

- [ ] Configure PR template:
  - [ ] Checklist for contributors

### CI/CD
- [ ] GitHub Actions workflow exists (`.github/workflows/`)
- [ ] CI runs tests on PR
- [ ] CI validates ONEX compliance
- [ ] CI checks formatting/linting
- [ ] All CI badges ready for README

---

## 👥 Community Readiness

### Communication
- [ ] **Code of Conduct** (optional but recommended):
  - [ ] Add `CODE_OF_CONDUCT.md` (use Contributor Covenant template)

- [ ] **Security Policy** (recommended):
  - [ ] Add `SECURITY.md` with vulnerability reporting instructions

- [ ] **Issue/PR Templates**:
  - [ ] Create `.github/ISSUE_TEMPLATE/bug_report.md`
  - [ ] Create `.github/ISSUE_TEMPLATE/feature_request.md`
  - [ ] Create `.github/PULL_REQUEST_TEMPLATE.md`

### Branding
- [ ] Repository has clear description
- [ ] Logo/icon (optional)
- [ ] Consistent naming across all omnibase_* repos

---

## 🚀 Pre-Launch Checklist

### Final Review
- [ ] All above sections completed
- [ ] Final commit with license and docs
- [ ] Tag release as `v0.1.0-public` (or similar)
- [ ] Update CHANGELOG.md with public release notes

### Launch Steps
1. [ ] Make repository public on GitHub
2. [ ] Announce in relevant communities (if desired)
3. [ ] Add to omnibase_core/spi README as related project
4. [ ] Monitor for initial issues/questions
5. [ ] Be responsive to first contributors (sets the tone!)

---

## 📊 Post-Public Monitoring

### First Week
- [ ] Check for opened issues
- [ ] Respond to questions promptly
- [ ] Monitor stars/forks (gauge interest)
- [ ] Update documentation based on feedback

### Ongoing
- [ ] Keep CI green
- [ ] Respond to PRs within 48 hours
- [ ] Update roadmap based on community interest
- [ ] Consider creating Discord/Slack for community

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Secrets exposed in history | Low | Critical | Run commit history audit |
| Missing documentation confuses users | Medium | High | Comprehensive README + examples |
| Dependencies on private repos | High | Critical | Verify all deps are public |
| Low-quality first impression | Medium | Medium | Clean up TODOs, ensure CI passes |
| Lack of community engagement | Medium | Low | Active initial responses, clear contrib guide |

---

## Notes

- **Timing**: Consider going public after PR #36 is merged for cleaner state
- **Coordination**: If omnibase_core/spi aren't public, coordinate simultaneous release
- **Support**: Be prepared for initial questions about ONEX architecture
- **Iteration**: Don't wait for perfection - can iterate after going public

---

**Status**: 🔴 Not Ready (Checklist not complete)
**Target Date**: TBD
**Owner**: Jonah Gray
