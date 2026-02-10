# FK Scan Report: omnibase_infra

**Ticket**: OMN-2066 (DB-SPLIT-03)
**Parent**: OMN-2054
**Date**: 2026-02-10
**Result**: PASS — all foreign keys are intra-service

---

## Summary

| Metric | Value |
|--------|-------|
| Forward migrations scanned | 16 (14 docker + 2 src) |
| Rollback migrations scanned | 16 (14 docker + 2 src) |
| Physical FK constraints found | 2 |
| Cross-service FK violations | **0** |
| Logical references (no FK) | 2 |

All `REFERENCES` / `FOREIGN KEY` constraints target tables owned by
omnibase_infra. No resolution plan is required.

---

## Physical Foreign Keys

Both constraints live in
`docker/migrations/forward/016_create_session_snapshots.sql`:

| # | Source Table | Source Column | Target Table | Target Column | On Delete | Intra-service? |
|---|-------------|---------------|--------------|---------------|-----------|----------------|
| 1 | `claude_session_prompts` | `snapshot_id` | `claude_session_snapshots` | `snapshot_id` | CASCADE | Yes |
| 2 | `claude_session_tools` | `snapshot_id` | `claude_session_snapshots` | `snapshot_id` | CASCADE | Yes |

These form a single aggregate: `claude_session_snapshots` is the parent,
with `prompts` and `tools` as child value objects sharing the same lifecycle.

---

## Logical References (No FK Constraint)

Found in `docker/migrations/forward/026_injection_effectiveness_tables.sql`:

| Child Table | Column | Logical Parent | Reason for No FK |
|-------------|--------|----------------|------------------|
| `latency_breakdowns` | `session_id` | `injection_effectiveness` | Async Kafka events may arrive out-of-order |
| `pattern_hit_rates` | `pattern_id` | (aggregated metric) | Eventual consistency for analytics |

Integrity is enforced at the application layer (see migration lines 128-135).

---

## Tables Owned by omnibase_infra

### docker/migrations/forward/

| Migration | Tables Created |
|-----------|---------------|
| 001 | `registration_projections` |
| 002 | *(indexes on 001)* |
| 003 | *(columns on 001)* |
| 004 | *(concurrent indexes on 001)* |
| 005 | `contracts`, `topics` |
| 016 | `claude_session_snapshots`, `claude_session_prompts`, `claude_session_tools`, `claude_session_event_idempotency` |
| 020 | `agent_actions` |
| 021 | `agent_routing_decisions` |
| 022 | `agent_transformation_events` |
| 023 | `router_performance_metrics` |
| 024 | `agent_detection_failures` |
| 025 | `agent_execution_logs` |
| 026 | `injection_effectiveness`, `latency_breakdowns`, `pattern_hit_rates` |
| 027 | `agent_status_events` |

### src/omnibase_infra/migrations/forward/

| Migration | Tables Created |
|-----------|---------------|
| 001 | `event_ledger` |
| 002 | `validation_event_ledger` |

**Total: 19 tables, all owned by omnibase_infra.**

---

## Methodology

1. `grep -rn 'REFERENCES\|FOREIGN KEY'` across both migration directories
2. Manual review of each forward migration file
3. Cross-referenced table names against other OmniNode repos (omniarchon,
   omniclaude, omninode_bridge, omnidash, omnibase_core) — no shared tables

---

## Conclusion

omnibase_infra has **zero cross-service foreign key violations**. The two
physical FK constraints form a well-scoped aggregate within a single
migration file. The intentional absence of FKs on async-event tables
(migration 026) is correctly documented in-migration and follows the
event-driven architecture pattern.

No remediation required.
