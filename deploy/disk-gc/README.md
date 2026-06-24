<!-- SPDX-FileCopyrightText: 2025 OmniNode.ai Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# `<onex-host>` disk-GC / worktree-reaper units (OMN-13008)

Two coexisting systemd USER timers implement the **two-layer merge-triggered
worktree GC** model. Neither replaces the other — both are retained.

| Unit pair | Layer | Cadence | Install |
|-----------|-------|---------|---------|
| `onex-worktree-reaper.{timer,service}` | 1 — event-first reaper (reap-on-merge) | every 2 min | `bash install-worktree-reaper.sh` |
| `onex-disk-gc.{timer,service}` | 2 — hourly backstop (worktree GC + docker GC + watermark) | hourly | `bash install-disk-gc.sh` |

- **Layer 1** reaps each newly-merged PR's worktree within ~one poll interval by
  reading the `onex.evt.github.pr-merged.v1` projection `?since=<cursor>`.
- **Layer 2** is the cursor-INDEPENDENT backstop that reconciles events missed
  during downtime / dropped before the cursor advanced. **`onex-disk-gc.timer`
  must stay installed** (T6 / OMN-13230 retention rule).

Both layers drive the same safety core (`omniclaude/scripts/prune-worktrees.sh`:
merged + clean + pushed-only; dirty → SKIP). The Mac equivalent of both layers is
the single `worktree_reaper.py --loop` KeepAlive daemon (Layer 1 fast poll +
Layer 2 catch-up sweep on start and hourly).

Full model, convergence proof, and verification commands:
[`docs/runbooks/worktree-reaper-two-layer-gc.md`](../../docs/runbooks/worktree-reaper-two-layer-gc.md).
