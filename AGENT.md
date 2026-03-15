# AGENT.md -- omnibase_infra

> LLM navigation guide. Points to context sources -- does not duplicate them.

## Context

- **Architecture**: `docs/architecture/`
- **Node inventory**: `docs/NODE_INVENTORY.md`
- **Migration guides**: `docs/migration/`
- **Conventions**: `CLAUDE.md`

## Commands

- Tests: `uv run pytest -m unit`
- Lint: `uv run ruff check src/ tests/`
- Type check: `uv run mypy src/omnibase_infra/ --strict`
- Pre-commit: `pre-commit run --all-files`
- Validate: `uv run python scripts/validate.py all`

## Cross-Repo

- Shared platform standards: `~/.claude/CLAUDE.md`
- Core models: `omnibase_core/CLAUDE.md`

## Rules

- Source `~/.omnibase/.env` before any DB, Kafka, or Infisical operation
- Never hardcode broker addresses -- use env vars
- Contract-driven handler registration (no wire_default_handlers for new code)
