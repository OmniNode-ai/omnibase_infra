# Migration Guides

This directory contains migration guides for breaking changes and terminology updates in omnibase_infra.

## Available Migrations

| Guide | Version | Status | Description |
|-------|---------|--------|-------------|
| [Handler to Dispatcher](./HANDLER_TO_DISPATCHER_MIGRATION.md) | 0.4.0 | Active | Terminology migration from "Handler" to "Dispatcher" in dispatch engine |

## Migration Philosophy

ONEX follows a **no backwards compatibility** policy. However, we provide:

1. **Clear documentation** of what changed and why
2. **Before/after code examples** for common patterns
3. **Automated migration scripts** where feasible
4. **Verification checklists** to confirm successful migration

## When to Create a Migration Guide

Create a migration guide when:

- Renaming public API methods, classes, or modules
- Changing the structure of models or enums
- Modifying configuration file formats
- Updating database schemas
- Changing event/message formats

## Migration Guide Template

```markdown
# [Feature] Migration Guide

**Version**: X.Y.Z
**Status**: Active | Deprecated
**Ticket**: OMN-XXX

## Overview
[Brief description of the change and why it was made]

## Scope of Changes
[What is affected, what is not]

## Migration Steps
[Step-by-step instructions]

## Backward Compatibility
[Any temporary compatibility measures]

## Verification Checklist
[How to verify migration was successful]

## FAQ
[Common questions]
```
