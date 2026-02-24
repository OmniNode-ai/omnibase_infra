# Standard Documentation Layout

> **Status**: Current | **Last Updated**: 2026-02-19

Prescriptive structure for the `docs/` directory in `omnibase_infra`.

---

## Table of Contents

1. [Required Directories](#required-directories)
2. [Optional Directories](#optional-directories)
3. [File Naming](#file-naming)
4. [Required Sections in Every Doc File](#required-sections-in-every-doc-file)
5. [Doc Authority Model](#doc-authority-model)
6. [Deleted Content Policy](#deleted-content-policy)
7. [See Also](#see-also)

---

## Required Directories

```text
docs/
├── architecture/          # System design, data flow, protocol topology
├── conventions/           # Coding standards, naming conventions, terminology
├── decisions/             # ADRs — why things work the way they do
├── getting-started/       # Installation, quick start, first node
├── guides/                # Step-by-step tutorials and how-to guides
├── operations/            # Runbooks, bootstrap procedures, infrastructure ops
├── patterns/              # Implementation patterns (circuit breaker, FSM, error handling)
├── reference/             # Contract specs, API reference, service wrappers
├── standards/             # Normative specs (terminology, topic taxonomy, this file)
└── testing/               # Test strategy, integration testing, E2E infrastructure
```

## Optional Directories

```text
docs/
├── migration/             # Migration guides for breaking changes
├── performance/           # Benchmark results and threshold definitions
├── plugins/               # Plugin system documentation
├── validation/            # Validation framework documentation
```

---

## File Naming

| Pattern | Use |
|---------|-----|
| `UPPER_SNAKE_CASE.md` | All documentation files |
| `README.md` | Directory index files only |
| `ADR-NNN-<slug>.md` | Architecture Decision Records in `decisions/` |

**Never**:
- Create versioned directories (`v1/`, `v2_0_0/`) — version through `contract.yaml` fields only
- Use lowercase filenames for documentation (only `README.md` is excepted)
- Use spaces or hyphens in documentation filenames (only ADR slugs use hyphens)

---

## Required Sections in Every Doc File

Every documentation file must begin with:

```markdown
# Title

> **Status**: Current | **Last Updated**: YYYY-MM-DD
```

Valid status values:

| Value | Meaning |
|-------|---------|
| `Current` | Actively maintained, accurate |
| `Draft` | Work in progress, may be incomplete |
| `Deprecated` | Superseded by another document (include link) |

The body must include at minimum:

1. **Overview** — one or two sentences stating what this document covers and who it is for
2. **Body sections** — the substantive content
3. **See Also** — a table linking to related documents

**Index files** (`README.md`) follow a simplified template:

```markdown
# Directory Name

> **Status**: Current | **Last Updated**: YYYY-MM-DD

Brief description of this directory's purpose.

| Document | Description |
|----------|-------------|
| [DOC_NAME.md](./DOC_NAME.md) | Short description |
```

---

## Doc Authority Model

| Location | Contains | Does NOT Contain |
|----------|----------|------------------|
| **CLAUDE.md** | Hard constraints, invariants, quick-reference rules, navigation pointers | Tutorials, architecture deep dives, code examples, full API reference |
| **docs/** | Explanations, tutorials, deep dives, architecture, reference | Rules that override CLAUDE.md |

**No duplication**: CLAUDE.md links to `docs/` sections. CLAUDE.md does not re-explain what `docs/` already covers.

---

## Deleted Content Policy

- Completed plans, stale analyses, and point-in-time reports are **deleted outright**
- No `archive/` directories — if unused, delete it
- Inbound links to deleted files must be removed or updated in the same commit

---

## See Also

| Topic | Document |
|-------|----------|
| Terminology definitions | [ONEX_TERMINOLOGY.md](./ONEX_TERMINOLOGY.md) |
| Topic naming standard | [TOPIC_TAXONOMY.md](./TOPIC_TAXONOMY.md) |
| Naming conventions | [../conventions/NAMING_CONVENTIONS.md](../conventions/NAMING_CONVENTIONS.md) |
| Test strategy | [../testing/CI_TEST_STRATEGY.md](../testing/CI_TEST_STRATEGY.md) |
