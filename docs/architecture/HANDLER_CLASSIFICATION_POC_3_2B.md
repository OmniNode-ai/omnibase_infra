> **Navigation**: [Home](../index.md) > [Architecture](README.md) > Handler Classification POC 3.2b

# Handler Classification Assessment: Projector Mixins (POC 3.2b)

**Ticket**: OMN-4009
**Epic**: OMN-4014 — Epic 3: Mixin/Service -> Handler Refactoring
**Status**: Complete — KEEP AS MIXIN (both candidates)
**Last Updated**: 2026-03-08

---

## Executive Summary

Both projector mixins score **0/5** on the OMN-4004 classification rubric. Decision:
**KEEP AS MIXIN** for `MixinProjectorSqlOperations` and `MixinProjectorNotificationPublishing`.

These mixins are class decomposition aids for `ProjectorShell` — splitting a large class
into focused, composable units without transferring I/O ownership or lifecycle responsibility.
Converting them to handlers would add injection complexity with no architectural gain.

---

## Candidates Evaluated

| Mixin | File | Lines | Used By |
|-------|------|-------|---------|
| `MixinProjectorSqlOperations` | `runtime/mixins/mixin_projector_sql_operations.py` | 785 | `ProjectorShell` only |
| `MixinProjectorNotificationPublishing` | `runtime/mixins/mixin_projector_notification_publishing.py` | 566 | `ProjectorShell` only |

---

## Classification Rubric — MixinProjectorSqlOperations

### Criterion 1: I/O Ownership

**Score: NO**

`MixinProjectorSqlOperations` does not own an `asyncpg.Pool`. The pool is declared as a type
hint (`_pool: asyncpg.Pool`) provided by the host class (`ProjectorShell`). The mixin borrows
the pool via composition — it never initializes, manages, or tears down the connection.

Contrast with `MixinLlmHttpTransport` (OMN-4008, 5/5 score): that mixin creates and owns its
`httpx.AsyncClient`, managing its full lifecycle. The projector mixin has no equivalent ownership.

### Criterion 2: Lifecycle Manageability

**Score: NO**

No `initialize()` or `shutdown()` methods exist or are needed. The pool lifecycle belongs
entirely to `ProjectorShell`. The mixin has no teardown path because it owns no resources.

### Criterion 3: Dispatch Entry Point Clarity

**Score: NO**

The mixin exposes **seven** methods used by `ProjectorShell`:
- `normalize_value()` / `_normalize_values()` — value normalization helpers
- `_upsert()` — INSERT ON CONFLICT DO UPDATE
- `_insert()` — INSERT fail-on-conflict
- `_append()` — INSERT append-only (delegates to `_insert`)
- `_partial_upsert()` — partial INSERT/UPDATE with RETURNING
- `_partial_update()` — partial UPDATE on specific columns
- `_parse_row_count()` — result parsing utility

This is intentional multiple-integration-point composition, not a single dispatch entry point.
`ProjectorShell` legitimately uses all of these for different projection modes and operation types.

### Criterion 4: Testability Without Subclassing

**Score: NO**

Testing `MixinProjectorSqlOperations` requires providing `_contract`, `_pool`, `_query_timeout`,
and `projector_id` — all of which are attributes that only exist on the host class. Tests must
subclass or construct a host class to exercise mixin methods. This is inherent to the class
decomposition pattern, not a testability deficiency.

### Criterion 5: Cross-Layer Leakage Risk

**Score: NO**

`MixinProjectorSqlOperations` is used exclusively by `ProjectorShell`, which is itself an
infrastructure-layer class with legitimate database I/O ownership. No compute or orchestrator
node inherits this mixin. There is no cross-layer boundary violation.

**Total score: 0/5 — KEEP AS MIXIN**

---

## Classification Rubric — MixinProjectorNotificationPublishing

### Criterion 1: I/O Ownership

**Score: NO**

The mixin uses two I/O resources: `asyncpg.Pool` (for pre-state fetch) and
`ProtocolTransitionNotificationPublisher` (for Kafka publishing). Neither is owned
by the mixin — both are declared as type hints provided by `ProjectorShell`. The mixin
receives these from the host; it does not instantiate, configure, or manage them.

### Criterion 2: Lifecycle Manageability

**Score: NO**

No lifecycle methods. The Kafka publisher (`ProtocolTransitionNotificationPublisher`) and
pool are managed by `ProjectorShell` and the broader DI container. The mixin has no
resources to initialize or shut down.

### Criterion 3: Dispatch Entry Point Clarity

**Score: NO**

The mixin exposes **six** methods used at different points in `ProjectorShell.project()`:
- `_is_notification_enabled()` — guard check
- `_get_notification_context()` — type-narrowed context fetch
- `_get_notification_config_if_enabled()` — config access helper
- `_get_notification_publisher_if_enabled()` — publisher access helper
- `_fetch_current_state_for_notification()` — pre-projection state fetch
- `_publish_transition_notification()` — post-commit notification publish
- `_extract_state_from_values()` / `_extract_aggregate_id_from_values()` / `_extract_version_from_values()` — value extraction helpers

Multiple entry points are used at different stages of the projection lifecycle.
This is coordinated composition, not a single-responsibility dispatch boundary.

### Criterion 4: Testability Without Subclassing

**Score: NO**

Test fixtures must provide `_contract`, `_pool`, `_query_timeout`, `_notification_publisher`,
`_notification_config`, `projector_id`, and `aggregate_type`. This requires subclassing or
constructing a host class. No different from the SQL operations mixin.

### Criterion 5: Cross-Layer Leakage Risk

**Score: NO**

Used exclusively by `ProjectorShell`. No layer boundary violations. The Kafka publishing
capability remains owned by `ProjectorShell` through the injected publisher; the mixin does
not grant new I/O capabilities to any compute or orchestrator node.

**Total score: 0/5 — KEEP AS MIXIN**

---

## Key Insight: Class Decomposition vs I/O Boundary

The projector mixins are architectural decomposition aids, not I/O boundary units.

`ProjectorShell` was split into three composable units (as documented in its `versionchanged`
entries) to keep the main class under the method count limit and to separate concerns:

```
ProjectorShell
  ├── MixinProjectorSqlOperations      — SQL generation + execution helpers
  └── MixinProjectorNotificationPublishing — notification lifecycle helpers
```

This is the correct use of mixins: splitting a large class into focused units where each
unit genuinely belongs as part of the same object. The I/O ownership (pool, publisher) stays
with `ProjectorShell`. The mixins borrow access via inherited attributes.

Converting these to injected handlers would require:
1. `ProjectorShell.__init__` to accept `HandlerProjectorSql` and `HandlerProjectorNotification`
2. All mixin method calls replaced with `self._sql_handler.method()` and `self._notification_handler.method()`
3. Handler contracts written for both
4. Container wiring for both handlers

For **zero architectural benefit** — `ProjectorShell` already owns and manages the pool and publisher.
The mixin merely provides access to SQL-building and notification-publishing helpers, which are
tightly coupled to `ProjectorShell`'s state (contract, pool, timeout, notification config).

---

## Impact on Wave 2

**OMN-4009 finding does not gate any downstream tickets.** The dependency map from OMN-4004
Section 9 lists OMN-4009 as contingent on OMN-4005, not as a gate for further tickets.

OMN-4011 (`ServiceTopicCatalogPostgres`) is a separate evaluation that stands on its own.

---

## Abort Conditions — Not Triggered

All four abort conditions from OMN-4004 Section 7 were assessed against a hypothetical
conversion to verify this classification:

| Abort Condition | Assessment |
|----------------|-----------|
| Dispatch integration more complex than expected | YES — would require 2 injected handlers, contract YAMLs, container wiring, 9+ caller sites updated across `ProjectorShell` |
| Behavior preservation requires excessive shims | YES — `_pool`, `_contract`, `_query_timeout` access would require refactoring every mixin method call |
| Handler boundaries distort ownership | YES — handler extraction would expose `ProjectorShell`'s internal SQL-building logic as a public handler API |
| >10 test files require structural changes | LIKELY — projector test suite is extensive |

All four abort conditions would trigger. This confirms the KEEP AS MIXIN decision.

---

## References

- [Handler Classification Rules](HANDLER_CLASSIFICATION_RULES.md) — OMN-4004 rubric
- Handler Classification POC 3.1 — postgres mixins (OMN-4005, PR #706; doc pending merge)
- `src/omnibase_infra/runtime/mixins/mixin_projector_sql_operations.py`
- `src/omnibase_infra/runtime/mixins/mixin_projector_notification_publishing.py`
- `src/omnibase_infra/runtime/projector_shell.py`
- OMN-4014 — parent epic
