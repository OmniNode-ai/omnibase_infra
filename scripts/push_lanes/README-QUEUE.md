# .201 heavy-prepush queue (added 2026-08-26, lane prepush-offload-201)

Offloads the governed pre-push selector (`scripts/hooks/prepush_smart_tests.sh`)
from the .200 Mac onto this host, **one heavy suite at a time**.

## Files

| path | purpose |
|---|---|
| `QUEUE` | one lane name per line, FIFO |
| `lanes/<LANE>.env` | lane spec: `LANE`, `WORKTREE`, `BRANCH` |
| `queue-runner.sh` | flock-guarded sequential runner |
| `queue-runner.log` | runner decisions (WAITING / SLOT ACQUIRED / rc) |
| `<WORKTREE>/.onex_state/push.log` | per-lane governed push transcript |

## Gates before a lane starts

1. no other `prepush_smart_tests.sh` running host-wide (covers foreign runs not
   launched through this queue)
2. `load1/nproc <= 1.0` — the same criterion the hook's own `host_is_fit` uses,
   so we never hand the hook a host it will refuse

## The hook is NEVER bypassed

The only env var set is `PREPUSH_201_GATE_RUNNER_HOSTNAME=omninode-pc` — the
hook's own sanctioned host-identity override, because this box's real
`hostname -s` is `omninode-pc`, not the hook default `gate-runner-201`.

NOT used: `--no-verify`, `PREPUSH_FULL_SUITE`, `ENABLE_SMART_TESTS`,
any `PREPUSH_ALLOW_*` degraded-capacity override, hook edits, `core.hooksPath`.

## Enqueue a lane

    cat > lanes/OMN-XXXX.env <<'SPEC'
    LANE=OMN-XXXX
    WORKTREE=/data/omninode/omni_home/omni_worktrees/OMN-XXXX/omnibase_core
    BRANCH=jonah/omn-xxxx-something
    SPEC
    echo OMN-XXXX >> QUEUE
    nohup setsid ./queue-runner.sh >> queue-runner.nohup.out 2>&1 < /dev/null &

## Recovery

The runner pops a lane off `QUEUE` when it dequeues it, so a waiting lane lives
only in the runner's memory. If the runner dies while a lane is waiting, the lane
spec still exists in `lanes/` — re-append the lane name to `QUEUE` and relaunch
the runner. `flock` makes a double-launch a safe no-op.

## Branch transfer

Always via `git bundle` + `scp` from the Mac, then:

    git -C /data/omninode/omni_home/omnibase_core fetch <bundle> <ref>:<ref>
    git -C /data/omninode/omni_home/omnibase_core worktree add <path> <branch>

NEVER a `git push` from the Mac — that triggers the very hook being offloaded.

---

## Wave 2 (2026-08-26, lane prepush-offload-201)

Four more lanes transplanted off the .200 Mac after the OMN-16589 canary.
Queue order (FIFO): **OMN-16589** (in runner memory, dequeued pre-wave-2) →
OMN-16625 → OMN-16581 → OMN-16507 → OMN-16346.

| lane | .201 worktree | transfer |
|---|---|---|
| OMN-16625 | `omni_worktrees/OMN-16625/omnibase_core` | bundle `omn16625-3836f652.bundle` md5 `bc76fdf7976d0b50729131c2a1e0213b` |
| OMN-16581 | `omni_worktrees/OMN-16581/omnibase_core` | bundle `omn16581-c20a13a6.bundle` md5 `b3b91e21a2ac94091c688eae9d3c5317` |
| OMN-16507 | `omni_worktrees/OMN-16507/omnibase_core` | bundle `omn16507-683f7dbe.bundle` md5 `2618066865794962afd5218e522c554f` |
| OMN-16346 | `omni_worktrees/OMN-16346/omnibase_core` | **reused a pre-existing worktree** — see below |

### OMN-16346 is a reused worktree, not a fresh transplant

.201 already carried this branch at `c8aac4fcd`, from an earlier abandoned
staging attempt. The Mac carried `9866c8f6`. They are DIFFERENT SHAs but the
**same tree** (`01dbbb091529159f01dd423c6f415dfdff99b465`) on the **same parent**
(`c5d6bfea2`) — two independent rebases of one commit, differing only in author
metadata. Content-equivalent, so the .201 copy was kept and nothing was
clobbered. `c8aac4fcd` is what will reach `origin`.

### Why no lane was rebased

The recon plan recommended rebasing OMN-16507/OMN-16346 (both "behind 7") on the
theory that divergence was widening the selector. That premise is **false**.
`prepush_smart_tests.sh` line 346 computes
`BASE_SHA=$(git merge-base "$BASE_REF" HEAD)` and line 357 diffs
`git diff --name-only "$BASE_SHA" HEAD` — a merge-base diff, so how far *behind*
a branch is cannot change the changed-file set. The hook also re-fetches the
base itself (line 340). Every lane on the Mac was selecting the identical
near-full path list regardless of divergence, which confirms it. Rebasing would
have rewritten another session's commit SHAs for zero benefit.

### The load gate is a HOOK constraint, not a queue-runner preference

`guard_full_suite_host()` calls `host_is_fit ""` against the **local** host and
`die`s if `load1/nproc > PREPUSH_LOAD_THRESHOLD` (default `1.0`). So a lane
started while .201 is over 1.0x does not run slowly — it **fails immediately**
with the same refusal seen on the Mac. The runner's identical gate exists so it
never hands the hook a host the hook will reject. Raising
`PREPUSH_LOAD_THRESHOLD` or setting `PREPUSH_ALLOW_LOCAL_FULL_SUITE=1` would be
a bypass producing explicitly degraded evidence — do not do it to drain a
backlog. The real lever is .201 CI-runner-fleet concurrency (68 listeners /
up to 57 busy workers observed 2026-08-26), which is what holds this box above
1.0x.

---

## Wave 3 addition (2026-08-26): OMN-16677

| lane | .201 worktree | transfer |
|---|---|---|
| OMN-16677 | `omni_worktrees/OMN-16677/omnibase_core` | bundle `omn16677-8d736f3b73.bundle` md5 `5a63118bf376349947ebada8bea25b6e` |

Branch `jonah/omn-16677-onex-run-harness`, commit `8d736f3b7`
(`feat(OMN-16677): expose the tier-0 local runtime harness as onex run`).

Transplanted because the Mac-side hook refused: the selector escalated to the
full suite (`is_full_suite=True reason=shared_module`, because the change edits
`src/omnibase_core/cli/cli_commands.py`) and BOTH `.200` and the `.201`
gate-runner were at/over the 1.0x-core load threshold at 2026-08-26T20:36Z
(Mac load1 31.3/24 cores; .201 load1 142/32 cores). No bypass was used --
`PREPUSH_ALLOW_LOCAL_FULL_SUITE` was explicitly NOT set.

Bundle base is `982dd1c322` (the .201 clone tip at transfer time), so the
bundle also carries `23035a787e`, the dev commit the branch was cut from.

Lane venv built via `uv sync --all-extras` before enqueue
(`setup-venv-OMN-16677.log`, rc=0).

Queue position at enqueue: 5th (behind OMN-16589 running, then OMN-16581,
OMN-16507, OMN-16346, OMN-16663-omnimarket).

**Still to do after the queued push lands the branch on origin:** open the PR
(`feat(OMN-16677): ...`), which does NOT yet exist. The change touches a runtime
path (`src/omnibase_core/runtime/harness/harness_cli.py`), so deploy-gate will
apply and an OCC evidence companion must merge FIRST.

---

## OMN-16663-omnimarket: base is OFF-LINEAGE — rebase before opening the PR

_Appended 2026-08-26T21:45Z, lane `brand-dup-fix`. The queue was deliberately **not** touched — this is a note for the eventual PR-opener, not a queue change._

`omnimarket#2152` (the OMN-16639 market badge row) **merged 2026-08-26T20:30:38Z**
as squash `fecf14bb41599c0bb2fad76095a026c4fb0fe5b6`, and its head branch —
`jonah/omn-16639-market-badge-row`, **the same name this lane's `BRANCH=` points at** —
was deleted on origin.

Because the landing was a squash, none of the branch commits are ancestors of `dev`.
This lane's base `dff2ddb9127d145328c1ee59ad4798b056f485ef` is now diverged:

    gh api repos/OmniNode-ai/omnimarket/compare/dff2ddb9...dev
    -> {"status":"diverged","ahead":3,"behind":1}

### After the push lane completes

1. **Rebase onto current `origin/dev`** before opening the PR. Opening from the
   transplanted base as-is would re-propose the badge-row changes `#2152` already
   landed, and conflict on `README.md`.
2. **Verify the net diff is banner-only.** Post-rebase, the diff vs `dev` must contain
   *only* the `README.md` `<picture>` banner block plus the brand assets under
   `docs/assets/brand/` (12 files on this repo — `BRAND.md` is correctly **absent**
   here per the omnimarket kb-doc-gate). A surviving badge-row hunk means the rebase
   kept the wrong side — redo it.
3. Expect to push to a **new branch name**; the old one no longer exists on origin.

The same note is appended as `#` comments to `lanes/OMN-16663-omnimarket.env`
(verified still shell-sourceable — `LANE`/`WORKTREE`/`BRANCH` all resolve).

Context: OMN-16663 fleet brand-banner rollout; omnimarket is one of 5 repos still
lacking the banner on its default branch as of 21:40Z.

---

## The cloud runner exists — and CANNOT take omnibase_core lanes yet (2026-08-26, lane `cloud-prepush-prove`)

_Appended 2026-08-26T23:40Z. The QUEUE was deliberately **not** touched: OMN-16677 stays
queued as the tail entry, exactly where it was. Nothing here changes queue order._

### What now exists

Two things landed/opened under **OMN-16688**:

1. **LIVE, zero-code:** `OMNI_TRUSTED_CI_RUNS_ON_JSON` flipped from
   `["self-hosted","omnibase-ci"]` to `["ubuntu-latest"]` at org scope **and** at the five
   repo scopes that override it. This is why this box drained: `.201` went
   **74.25/32 = 2.32x → 9.20/32 = 0.29x**, fleet 50-busy/3-idle → 63-idle, and the queue
   started moving again (OMN-16581 acquired a slot 22:51:51Z after 41m of `WAITING`).
2. **PR OPEN, not merged:** `omnibase_infra#2917` adds `scripts/hooks/prepush_remote_verify.py`
   — a **third heavy-escalation execution target**: a GitHub-hosted full-suite CI run pinned
   to the exact HEAD sha, consulted where the hook would otherwise refuse.

### Routing a lane to the cloud runner — the intended recipe

    uv run python scripts/hooks/prepush_remote_verify.py check --head-sha "$(git rev-parse HEAD)"
    # exit 0 = sha-pinned + green + full-suite shaped -> hook accepts it in place of the local run
    # exit 1 = resolved, no qualifying run     exit 2 = could not resolve (counts as NO evidence)

Then `git push` normally. No env var is set, none exists — see `omnibase_infra`
`docs/runbooks/prepush-remote-full-suite-verify.md`.

### Why NO lane in this queue can use it today — measured, not assumed

**Blocker 1 — the target does not exist in `omnibase_core`.** `omnibase_infra#2917` is open
and touches only `omnibase_infra`. A real governed push attempt of the OMN-16677 lane on
`.200` at 2026-08-26T23:32Z (`Stickybeatz-Studio`, 24 cores, load1 42.91 = 1.79x, and
`env | grep -Ei 'PREPUSH|ENABLE_SMART_TESTS'` → **none present**) produced:

    [prepush-smart-tests] selection: is_full_suite=True reason=shared_module paths=[ tests/ ] (feature-flag=on)
    [prepush-smart-tests] ERROR: heavy fail-closed full-suite escalation triggered on
      'Stickybeatz-Studio' (the designated host by identity), but its load is at/over the
      1.0x-core threshold
    [prepush-smart-tests] REMEDIATION: the .201 gate-runner (gate-runner-201) currently HAS
      capacity -- route there instead. ... or set PREPUSH_ALLOW_LOCAL_FULL_SUITE=1 ...

Exactly two options offered — `.201`, or the degraded-evidence bypass. **No cloud target.**
That is the pre-OMN-16688 wording, i.e. `omnibase_core`'s vendored hook copy has no remote path.

**Blocker 2 — bootstrap gap: the target is a CONSUMER, and there is nothing to consume.**
`check` requires a `ci.yml` run pinned to the sha. For OMN-16677's commit
`8d736f3b734960bea735553c579af48ab686c09a`:

    gh api repos/OmniNode-ai/omnibase_core/commits/8d736f3b73...
      -> HTTP 422 "No commit found for SHA"
    gh api ".../actions/runs?head_sha=8d736f3b73..." --jq .total_count   -> 0
    git ls-remote origin | grep -c 8d736f3b                             -> 0

    prepush_remote_verify.py check --head-sha 8d736f3b73... --json
      -> {"ok": false, "reason": "no ci.yml run exists for 8d736f3b7349 on OmniNode-ai/omnibase_core yet"}

CI only runs on a sha GitHub already has; GitHub only has a sha that was pushed; the push is
the thing being gated. **Every lane in this queue is a never-pushed branch**, so none of them
can present remote evidence. The mechanism helps a *re-push* of an already-published sha; it
cannot land a new branch. Getting the objects onto origin any other way (a hookless clone, a
server-side ref via the Git Data API) is a bypass — **do not**.

**Blocker 3 — binding 3 does not hold in `omnibase_core`; porting it as-is would OPEN a bypass.**
The design's safety rests on the shard denominator being a forge-resistant witness of
`is_full_suite`, because narrowed runs are "capped far below" the full-suite count. In
`omnibase_infra` that is true: `_FULL_SUITE_SPLIT_COUNT = 15` vs a narrowed ceiling of 5.
In `omnibase_core` it is **false**:

| | `omnibase_infra` | `omnibase_core` |
|---|---|---|
| full-suite split count | `_FULL_SUITE_SPLIT_COUNT = 15` (line 189) | `split_count=40` in `_full_suite()` (line 347) |
| narrowed ceiling | `_split_count_for` returns ≤ **5** | `min(..., VOLUME_MAX_SPLITS)`, `VOLUME_MAX_SPLITS = **40**` (line 107) |
| gap | 5 → 15, safe | **40 → 40, none** |

`omnibase_core` also has no `_FULL_SUITE_SPLIT_COUNT` symbol at all, so the checker's
"imported from the selector, never re-typed" import does not even resolve there.

Live proof the gap is real, not theoretical: run **33019144356** (omnibase_core, PR event,
`ubuntu-latest`, 22:17:55Z→23:14:05Z) ran shards named **`Tests (Split i/39)`** — a
*selector-narrowed* run **one shard below** the full-suite denominator of 40. A narrowed
selection covering ≥1560 test files would emit exactly 40 and be **indistinguishable from
the full suite** by the only signal binding 3 inspects.

### Measured cost/wall-clock, for when this is unblocked

| | measured |
|---|---|
| `.201` queue, one successful lane (OMN-16589) | dequeued 18:27:06Z → slot 18:28:06Z → finished 22:10:42Z rc=0 = **3h42m36s executing**, serialized one-at-a-time |
| `.201` observed queue wait | 26m34s (OMN-16589), 41m09s (OMN-16581) |
| GitHub-hosted sharded suite (run 33019144356, 39 shards, `ubuntu-latest`) | **56m10s** wall; shards ~21-24m each, in parallel |
| GitHub-hosted marginal cost | **$0** — every OmniNode repo is public, so GitHub-hosted minutes are free and unmetered |
| `.201` marginal cost | ~14 core-hours of contention per lane, on the box whose saturation created this queue |

OMN-16677 sits 4th (behind OMN-16581 running, then OMN-16507, OMN-16346,
OMN-16663-omnimarket). At the measured ~3h43m/lane plus waits that is a **projection** of
roughly 15h to reach it. Cloud is ~4x faster on execution alone and ~16x on
queue-position-to-green — which is why closing the blockers above is worth doing.

### What would actually unblock it

Not a port. The gate has to be able to **initiate** its own remote run — on the path where it
would `die`, push HEAD to a dedicated non-proposal CI-probe ref *from inside the gate*
(recursion-guarded), wait for the sha-pinned run to go green, then permit the real push. That
is an architecture change to a governed gate across two repos and needs an explicit operator
decision; it was NOT taken unilaterally here. Blocker 3 must be fixed in the same change or
the port is a net loss of gate strength.

**Until then: `omnibase_core` heavy lanes route to `.201`, exactly as they do now.**
