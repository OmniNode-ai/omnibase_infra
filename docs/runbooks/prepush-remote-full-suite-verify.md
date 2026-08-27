# Pre-push remote full-suite verification (OMN-16688)

Third execution target for the governed pre-push hook's **heavy (full-suite)
escalation**, alongside `.200` (OMN-15059) and the `.201` gate-runner container
(OMN-16295): a **GitHub-hosted CI run of the full suite, pinned to the exact
HEAD sha being pushed**.

- Hook: `scripts/hooks/prepush_smart_tests.sh` (`guard_full_suite_host`)
- Checker: `scripts/hooks/prepush_remote_verify.py`
- Proof: `tests/unit/scripts/test_prepush_remote_verify.py`

## Why it exists

Both local heavy targets can be over the load threshold at the same time, and on
2026-08-26 both were. `.201` sat at load **74.25 against 32 cores (2.32x)** with
**50 of 53** self-hosted runners busy, so every heavy escalation refused and the
pre-push queue stalled outright — not slow, *stalled*, because `host_is_fit` is a
**hook** constraint, so no amount of queue depth can satisfy it.

Meanwhile every OmniNode repo is public, GitHub-hosted minutes are free and
unmetered, and `.github/workflows/ci.yml` already runs this same sharded full
suite on `ubuntu-latest`. The local heavy escalation was a slow, contended
duplicate of a fast, free gate that runs anyway.

## When it fires

Only on the paths that would otherwise refuse the push or fall back to a
degraded-evidence override grant:

| Situation | Before | Now |
|---|---|---|
| Designated host, fit | run locally | run locally (**unchanged**) |
| Designated host, over threshold | grant, else refuse | **remote check first**, then grant, then refuse |
| Undesignated host | grant, else refuse | **remote check first**, then grant, then refuse |

Every currently-passing path behaves identically. The remote check is tried
*before* the override grant because it is strictly **stronger** evidence: an
uncontended full run on this exact tree, versus a contended local one.

## The three bindings

`check` returns 0 only if all three hold:

1. **SHA-pinned** — `run.head_sha` equals the exact 40-character HEAD sha. Never
   a prefix, never a branch name. A git sha is content-addressed, so a run on
   that sha is a run on that exact tree.
2. **Green** — `status == completed` and `conclusion == success`.
3. **Full-suite shape** — every shard `Tests (Split i/N)` for `i` in `1..N` is
   present and succeeded, where `N` is `_FULL_SUITE_SPLIT_COUNT` **imported from
   the selector itself**, never re-typed.

Binding 3 is what keeps this from being a downgrade. The selector emits
`split_count == _FULL_SUITE_SPLIT_COUNT` **only** when `is_full_suite=True`;
narrowed selections are capped at 5 by `_split_count_for`, well below 15. So the
shard denominator visible in the job names is a faithful, forge-resistant witness
of `is_full_suite` — produced by CI from the pushed tree, not supplied by the
caller. `test_narrowed_ceiling_cannot_reach_full_suite_count` fails loudly if a
future change ever closes that gap.

## Why this is not a bypass

- **No env override.** It adds no `PREPUSH_*` variable. The selector contract is
  untouched; escalation is never skipped.
- **No local artifact.** Nothing is written to disk, so there is nothing to
  forge — the answer is re-derived live from the GitHub API each time.
- **It cannot accept less work.** A selector-narrowed run is refused exactly as a
  failing one is.
- **Unresolvable is not a pass.** Exit `2` (`gh` missing, auth failure, API
  timeout) is treated as *no evidence* and falls through to the existing
  refusal — the same fail-closed posture as the load probe.

Exit codes: `0` pass · `1` resolved, no qualifying run · `2` could not resolve.

## Operator usage

Check by hand:

```bash
uv run python scripts/hooks/prepush_remote_verify.py check --head-sha "$(git rev-parse HEAD)"
uv run python scripts/hooks/prepush_remote_verify.py check --head-sha "$(git rev-parse HEAD)" --json
```

Normal flow when both hosts are loaded: push the branch / open the PR so CI runs
the full suite on that sha, let it go green, then re-push — the hook picks the
run up automatically.

**Credentials:** all reads go through `gh api`, reusing the operator's existing
`gh auth` session. No token is read, written, minted, or persisted, and no new
secret is introduced.

## Related: trusted-CI runner routing

The same ticket flipped `OMNI_TRUSTED_CI_RUNS_ON_JSON` from
`["self-hosted","omnibase-ci"]` to `["ubuntu-latest"]` at org scope **and** at
each of the five repo scopes that override it (`omnibase_core`, `omnibase_infra`,
`omnimarket`, `omniclaude`, `onex_change_control`) — a repo-scoped Actions
variable overrides the org one, so flipping org alone drains nothing.

Effect, measured within ~6 minutes: `.201` load **74.25 → 18.23** (0.57x, back
under the 1.0 fit threshold) and the fleet went from 50-busy/3-idle to
5-busy/52-idle. Rollback is one `gh api -X PATCH` per scope.
