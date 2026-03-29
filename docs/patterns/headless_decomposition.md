# Headless Pipeline Decomposition Pattern

> **Ticket**: OMN-6927 | **Epic**: OMN-6924
>
> Defines the decomposition pattern for replacing mega-session autonomous pipelines
> with scoped headless invocations that complete in bounded time.

## Problem

Long-running autonomous pipelines (48-hour close-outs, overnight cron autopilot)
fail due to:

1. **Context window exhaustion** — 9 `context_window_exceeded` events in 30 days
2. **Agent stalling** — sub-agents stop producing tool calls mid-pipeline
3. **Partial completion** — first cron pass partially succeeds, subsequent retries
   fail because context is polluted with prior state

## Solution: Scoped Headless Invocations

Replace a single mega-session with N parallel headless tasks, each scoped to
**one repo** and **one action**, designed to complete in **under 15 minutes**.

### Core Principles

1. **One task, one repo, one action** — each headless invocation does exactly one
   thing (e.g., "merge-sweep omnibase_core" or "release omniclaude")
2. **Bounded execution** — each task completes in <15 minutes or fails explicitly
3. **State handoff via files** — stages communicate through structured state files,
   not shared session context
4. **Parallel by default** — independent tasks run simultaneously via shell `&` + `wait`
5. **Idempotent** — re-running a stage with the same input produces the same result

## Invocation Pattern

### Single Headless Task

```bash
claude -p \
  --print \
  --permission-mode auto \
  --allowedTools "Bash(git:*) Read Glob Grep" \
  --working-directory /Volumes/PRO-G40/Code/omni_home/<repo> \
  "Merge-sweep <repo>: list open PRs with passing CI, merge them, report results." \
  > /tmp/headless-state/<run-id>/<repo>/merge-sweep.json 2>&1
```

### Key Flags

| Flag | Purpose |
|------|---------|
| `-p` / `--print` | Non-interactive mode, print response and exit |
| `--permission-mode auto` | Auto-approve tool use (headless, no human present) |
| `--allowedTools` | Constrain to only the tools needed for this task |
| `--working-directory` | Scope to a single repo |
| `--max-budget-usd` | Optional cost cap per invocation |

### Tool Scoping Examples

| Task Type | Allowed Tools |
|-----------|--------------|
| Merge sweep | `Bash(git:*,gh:*) Read Glob Grep` |
| Release | `Bash(git:*,gh:*,uv:*) Read Edit Write Glob Grep` |
| Ticket close | `Bash(gh:*) Read Grep mcp__linear-server__*` |
| Report generation | `Read Glob Grep Write` |

## State Handoff Model

### State File Schema

Each stage writes a JSON state file that the next stage reads:

```json
{
  "schema_version": "1.0",
  "run_id": "2026-03-28T22-00-00Z",
  "stage": "merge-sweep",
  "repo": "omnibase_core",
  "status": "success|partial|failed",
  "started_at": "2026-03-28T22:00:00Z",
  "completed_at": "2026-03-28T22:03:42Z",
  "results": {
    "prs_merged": ["#142", "#145"],
    "prs_skipped": ["#143"],
    "prs_failed": []
  },
  "errors": [],
  "next_stage_input": {
    "merged_versions": {
      "omnibase_core": "0.35.0"
    }
  }
}
```

### State Directory Layout

```
/tmp/headless-state/<run-id>/
├── manifest.json              # Run-level manifest (repos, stages, config)
├── omnibase_core/
│   ├── merge-sweep.json       # Stage 1 output
│   ├── release.json           # Stage 2 output (reads merge-sweep.json)
│   └── deploy-verify.json     # Stage 3 output (reads release.json)
├── omnibase_infra/
│   ├── merge-sweep.json
│   ├── release.json
│   └── deploy-verify.json
└── summary.json               # Aggregated run summary (written by coordinator)
```

### Stage Dependencies

```
Stage 1: merge-sweep (per repo, parallel)
    ↓ writes merge-sweep.json per repo
Stage 2: release (per repo, parallel, reads merge-sweep.json)
    ↓ writes release.json per repo
Stage 3: deploy-verify (per repo, parallel, reads release.json)
    ↓ writes deploy-verify.json per repo
Stage 4: summary (sequential, reads all per-repo outputs)
    ↓ writes summary.json
```

## Coordinator Shell Script Pattern

```bash
#!/usr/bin/env bash
set -euo pipefail

RUN_ID=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
STATE_DIR="/tmp/headless-state/${RUN_ID}"
mkdir -p "${STATE_DIR}"

REPOS=(omnibase_core omnibase_infra omnibase_spi omniclaude)
OMNI_HOME="/Volumes/PRO-G40/Code/omni_home"

# --- Stage 1: Merge sweep (parallel) ---
pids=()
for repo in "${REPOS[@]}"; do
  mkdir -p "${STATE_DIR}/${repo}"
  claude -p \
    --print \
    --permission-mode auto \
    --allowedTools "Bash(git:*,gh:*) Read Glob Grep" \
    --working-directory "${OMNI_HOME}/${repo}" \
    "Merge-sweep: list open PRs with passing CI, merge green PRs.
     Write results as JSON to stdout with schema:
     {status, prs_merged, prs_skipped, prs_failed, errors}" \
    > "${STATE_DIR}/${repo}/merge-sweep.json" 2>"${STATE_DIR}/${repo}/merge-sweep.log" &
  pids+=($!)
done

# Wait for all merge-sweeps
for pid in "${pids[@]}"; do
  wait "$pid" || echo "WARNING: merge-sweep PID $pid failed"
done

# --- Stage 2: Release (parallel, reads Stage 1) ---
pids=()
for repo in "${REPOS[@]}"; do
  merge_result="${STATE_DIR}/${repo}/merge-sweep.json"
  if [[ -f "$merge_result" ]] && jq -e '.status == "success"' "$merge_result" >/dev/null 2>&1; then
    claude -p \
      --print \
      --permission-mode auto \
      --allowedTools "Bash(git:*,gh:*,uv:*) Read Edit Write Glob Grep" \
      --working-directory "${OMNI_HOME}/${repo}" \
      "Release: bump version, create tag, push. Previous stage result:
       $(cat "$merge_result")
       Write release result as JSON to stdout." \
      > "${STATE_DIR}/${repo}/release.json" 2>"${STATE_DIR}/${repo}/release.log" &
    pids+=($!)
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid" || echo "WARNING: release PID $pid failed"
done

echo "Run complete: ${STATE_DIR}"
```

## Failure Handling

### Per-Task Failures

Each headless task either succeeds or fails independently. Failures do NOT
block other parallel tasks. The coordinator checks exit codes and state files
after each stage.

### Retry Policy

- **Automatic retry**: If a task fails with a transient error (network, rate limit),
  retry once with the same parameters
- **No retry**: If a task fails with a logic error (merge conflict, test failure),
  log it and continue with other tasks
- **Escalation**: If >50% of tasks in a stage fail, halt the pipeline and write
  a diagnosis document per the Two-Strike Protocol

### Timeout

Each headless invocation should complete in <15 minutes. Use `timeout` as a guard:

```bash
timeout 900 claude -p ... || echo "TIMEOUT: ${repo}/${stage}" >> "${STATE_DIR}/failures.log"
```

## Migration from Mega-Sessions

| Before (mega-session) | After (decomposed) |
|------------------------|-------------------|
| Single 48-hour session | N parallel 10-min tasks |
| Shared context across repos | Isolated context per repo |
| Context overflow on retry | Clean state per invocation |
| Manual stall diagnosis | Automatic timeout + retry |
| All-or-nothing completion | Per-repo partial success |

## Related

- **Task 11** (OMN-6935): Concrete implementation for close-out/autopilot
- **Task 12** (OMN-6936): Extended headless templates for all pipeline stages
- **Agent Orchestration rules**: `~/.claude/CLAUDE.md` section on stall detection
