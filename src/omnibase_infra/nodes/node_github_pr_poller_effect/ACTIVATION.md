# GitHub PR Poller Effect — Activation Guide

## Overview

`NodeGitHubPRPollerEffect` polls the GitHub REST API for open PR triage state
across configured repositories and publishes `ModelGitHubPRStatusEvent` events
to `onex.evt.github.pr-status.v1`.

## Prerequisites

1. **GITHUB_TOKEN** must be set in `~/.omnibase/.env` (or Infisical)
2. Runtime services must be running (`infra-up-runtime`)
3. Kafka/Redpanda must be available on the configured bootstrap servers

## Configuration

The node is configured via `contract.yaml`:

- `poll_interval_seconds`: Minimum seconds between polls (default: 60)
- `repos`: List of `{owner}/{name}` repositories to poll (default: empty)
- `stale_threshold_hours`: Hours before a PR is classified as stale (default: 48)
- `github_token_env_var`: Env var containing the token (default: `GITHUB_TOKEN`)

## Event Flow

```
Runtime tick (onex.evt.runtime.tick.v1)
  --> NodeGitHubPRPollerEffect
    --> HandlerGitHubApiPoll (polls GitHub REST API)
      --> onex.evt.github.pr-status.v1 (one event per open PR)
        --> omnidash StatusProjection.upsertPR()
          --> /status page (PR triage dashboard)
```

## Omnidash Consumer

The omnidash `StatusProjection` (in-memory, singleton) consumes
`onex.evt.github.pr-status.v1` events via its `upsertPR()` method.
The `/status` page renders PR data grouped by triage state.

## Triage States

PRs are classified into 8 triage states:
- `open`, `ci_running`, `ci_failed`, `approved_pending_ci`
- `approved`, `changes_requested`, `merged`, `closed`

## Verification

After starting the runtime:
1. Check logs for `[node_github_pr_poller_effect]` entries
2. Verify events on `onex.evt.github.pr-status.v1` topic
3. Check omnidash `/status` page shows PR data
