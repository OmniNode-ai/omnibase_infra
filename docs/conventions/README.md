# Conventions

> **Status**: Current | **Last Updated**: 2026-02-19

This directory contains coding conventions and standards specific to `omnibase_infra`. For shared platform-wide standards (Python version, git rules, testing, environment configuration, infrastructure topology), see `~/.claude/CLAUDE.md`. The authoritative quick-reference tables for naming and model patterns live in `CLAUDE.md`; the documents below expand on those with infra-specific examples, rationale, and edge cases.

---

## Index

| Document | Description |
|----------|-------------|
| [NAMING_CONVENTIONS.md](NAMING_CONVENTIONS.md) | File and class naming patterns for all infra artifact types, with real examples from `src/omnibase_infra/` |
| [PYDANTIC_BEST_PRACTICES.md](PYDANTIC_BEST_PRACTICES.md) | ConfigDict requirements, field patterns, immutability rules, and custom `__bool__` documentation |
| [ERROR_HANDLING_BEST_PRACTICES.md](ERROR_HANDLING_BEST_PRACTICES.md) | Infra error hierarchy, `ModelInfraErrorContext` usage, error class selection, and sanitization rules |
| [TERMINOLOGY_GUIDE.md](TERMINOLOGY_GUIDE.md) | Canonical definitions for ONEX architectural terms used in code, comments, and documentation |

---

## How to Use These Docs

- **Starting a new node?** Read `NAMING_CONVENTIONS.md` first, then `PYDANTIC_BEST_PRACTICES.md` for its models.
- **Raising an error?** `ERROR_HANDLING_BEST_PRACTICES.md` has the selection table and mandatory `with_correlation()` pattern.
- **Writing a docstring or comment?** `TERMINOLOGY_GUIDE.md` clarifies which term to use (effect vs Effect vs EFFECT, intent vs action, etc.).
- **Unsure if a pattern applies here or in `omnibase_core`?** The infra conventions cover infrastructure-layer concerns; `omnibase_core/docs/conventions/` covers the core node archetypes and topic naming.

---

## Related Documentation

- `CLAUDE.md` (repo root) — Authoritative naming table, architecture invariants, handler system, intent model
- `docs/patterns/` — Deep-dive implementation guides (circuit breaker, dispatcher resilience, container DI, etc.)
- `docs/decisions/` — Architecture decision records
- `~/.claude/CLAUDE.md` — Platform-wide shared standards
