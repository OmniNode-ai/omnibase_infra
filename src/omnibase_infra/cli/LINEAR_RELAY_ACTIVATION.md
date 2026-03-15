# Linear Snapshot Relay — Activation Guide

## Overview

`onex-linear-relay` is a CLI tool that reads a Linear workspace snapshot JSON
file and publishes a `ModelLinearSnapshotEvent` to the
`onex.evt.linear.snapshot.v1` Kafka topic.

This is the **primary ingress path** for Linear data into the ONEX event bus.
The omnidash `/api/linear/snapshot` endpoint is debug-only and must NOT be used
as a primary data source.

## Usage

```bash
# Basic usage: emit a Linear snapshot to Kafka
onex-linear-relay emit --snapshot-file /tmp/linear-snapshot.json

# With custom bootstrap servers
onex-linear-relay --bootstrap-servers localhost:19092 emit --snapshot-file /tmp/snapshot.json

# With explicit snapshot ID (auto-generated if omitted)
onex-linear-relay emit --snapshot-file /tmp/snapshot.json --snapshot-id "my-uuid-here"
```

## Non-Blocking Design

If Kafka is unreachable within 2 seconds, the CLI:
1. Exits 0 (does NOT block callers)
2. Spools the event to `~/.onex/spool/linear-snapshots.jsonl` for deferred delivery

## Snapshot JSON Format

The snapshot file must be a JSON object. The `workstreams` key is extracted if
present:

```json
{
  "workstreams": [
    { "name": "Active Sprint", "status": "in_progress", "ticket_count": 12 },
    { "name": "Ready", "status": "planned", "ticket_count": 8 }
  ],
  "teams": [...],
  "projects": [...]
}
```

## Event Flow

```
onex-linear-relay CLI
  --> onex.evt.linear.snapshot.v1 (Kafka)
    --> omnidash StatusProjection.replaceWorkstreams()
      --> /status page (Workstreams section)
```

## Automation Options

### Option 1: Cron Schedule (recommended)

```bash
# Add to crontab: run daily at 8am
0 8 * * * /path/to/generate-linear-snapshot.sh | onex-linear-relay emit --snapshot-file /dev/stdin
```

### Option 2: Session-Start Hook

Add to omniclaude session-start hook to capture Linear state at the beginning
of each work session.

### Option 3: Manual Trigger

Run manually whenever a fresh snapshot is needed for the Status page.

## Omnidash Consumer

The `StatusProjection` singleton (in-memory) consumes
`onex.evt.linear.snapshot.v1` events. It uses single-snapshot semantics:
each new snapshot **replaces** the previous workstreams entirely.

The `/status` page displays:
- Workstream status breakdown
- Last snapshot timestamp
- Workstream counts per status

## Verification

1. Generate or obtain a Linear snapshot JSON file
2. Run: `onex-linear-relay emit --snapshot-file /path/to/snapshot.json`
3. Check Kafka: `kcat -C -b localhost:19092 -t onex.evt.linear.snapshot.v1 -c 1`
4. Check omnidash `/status` page shows workstream data
