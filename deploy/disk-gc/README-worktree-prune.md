# .201 worktree prune — disposition of the two passes it replaces (OMN-18688)

This file is the reason of record for turning two live systemd passes off. It exists
because the ticket's own words were *"do not silently leave two competing,
differently-safe pruners live on the same host"* — so the decision has to be citable
after the session that made it is gone.

Every figure below was read live from `192.168.86.201` on **2026-09-18**, read-only,
before anything was changed.

---

## Disposition

| Unit / line | Disposition | Why |
|---|---|---|
| `onex-worktree-reaper.timer` + `.service` (OMN-13228) | **RETIRED** — stopped, disabled, unit files removed | drives a predicate with no ledger-CLAIM gate and `--force`; has reaped nothing for its whole recent history; points at a root that does not hold the worktrees |
| the `worktree-gc.sh` `ExecStart` line in `onex-disk-gc.service` | **RETIRED** — line deleted from the unit | same predicate, same missing gate, same wrong root |
| `disk-gc.sh`, `docker-volume-gc.sh`, `disk-watermark-check.sh` lines | **KEPT, untouched** | these do real, working image/builder/volume reclaim and are out of scope |
| `onex-worktree-prune.timer` + `.service` | **NEW** | ledger-CLAIM-aware, no `--force`, correct root |

The `--uninstall` path of the installer deliberately does **not** put the reaper back.
It was retired on safety grounds, not to make room.

---

## The three findings behind it

### 1. Both retired passes drive the same predicate, and it has no ledger-claim gate

Neither `worktree_reaper.py` nor `worktree-gc.sh` implements a removal rule of its own.
Both are thin drivers over `omniclaude/scripts/prune-worktrees.sh`:

- `worktree-gc.sh` sets `PRUNE_SCRIPT="${OMNI_HOME}/omniclaude/scripts/prune-worktrees.sh"` and execs it;
- `worktree_reaper.py`'s `reap_row()` builds `["bash", prune_script, "--worktrees-root", root]` and appends `--execute`. It only decides *when* to invoke, gated on a `pr-merged` projection row.

That shared predicate removes a worktree when the remote branch is gone **or** a merged
PR exists, **and** the tree is clean, **and** everything is pushed — with
`git worktree remove --force`. `grep -n "LEDGER\|ledger\|CLAIM"` returns **zero matches
in all three files**. None of them reads `docs/tracking/ROLLING_WORK_LEDGER.md` in any form.

**Clean, pushed and merged is precisely the state a live lane occupies between its push
and its post-merge verification.** That is the hazard OMN-15551 measured, and it is why
the replacement additionally requires no open `CLAIM` row — re-verified live immediately
before each removal, because a registry-scale classification pass runs for tens of minutes
while peer lanes keep working.

Stated plainly, because the disposition turns on it:

- **Would either retired pass take a dirty tree?** No. The dirty-tree check skips it. All
  20 of the census's `.201` STALE-DIRTY rows are dirty by definition of that class, so
  none was ever at risk from these two.
- **Would either take a clean, merged tree carrying an open ledger `CLAIM`?** **Yes.** The
  predicate cannot see the ledger. This is the whole reason for the replacement.

### 2. Neither retired pass has ever been able to see a real worktree on this host

Both `ExecStart` lines pass `/data/omninode/omni_worktrees`. That path is missing the
`omni_home` segment.

| path | ticket dirs |
|---|---|
| `/data/omninode/omni_worktrees` (what the units passed) | **1** |
| `/data/omninode/omni_home/omni_worktrees` (the real root, `$OMNI_HOME/omni_worktrees`) | **163** |

`systemctl --user show-environment` confirms `OMNI_HOME=/data/omninode/omni_home`, and
`~/Code/omni_home` is a symlink to it. The journal shows the consequence directly — every
recent `onex-disk-gc` run logs `No worktrees found under /data/omninode/omni_worktrees`
while `disk-gc.sh` and `docker-volume-gc.sh` in the same unit report real reclaim.

This was a literal path defect in the repository templates, not host drift: the installed
unit files diffed byte-identical against `deploy/disk-gc/`. It is fixed in the new unit by
using `%h/Code/omni_home/omni_worktrees` rather than a hand-written absolute path.

**Consequence for risk:** retiring these two reclaims nothing that was working and
regresses nothing, because neither has ever removed a real worktree here.

### 3. The reaper has a second, independent reason it does nothing

Its projection fetch fails on every run:

```
[worktree-reaper] projection fetch failed (cursor un-advanced): HTTP Error 503: Service Unavailable
[worktree-reaper] pass done rows_seen=0 reaped_ok=0 cursor 0->0 advanced=False
```

Ten consecutive runs on a 2-minute cadence, `rows_seen=0` every time, cursor never
advancing. Its timer has been `active (waiting)` since 2026-08-23 doing nothing but
failing. A 2-minute timer that cannot reach its dependency is not a safety net.

---

## Which clone the new unit runs from, and why it is not the infra one

`onex-worktree-prune.service` resolves its script through
`%h/Code/omni_home/omniclaude`, **not** through `omnibase_infra`.

Every other `ExecStart` on this host runs out of `/data/omninode/omni_home/omnibase_infra`,
which is **detached** (`## HEAD (no branch)`) and pinned at `9add32a9`. Advancing that pin
is a separate change with its own approval, and this unit must not require it. The
omniclaude clone is on `dev`, clean, tracking `origin/dev` with zero commits behind, and it
carries the classifier and its pure predicate together — which is what the script needs on
its import path anyway.

*(Correction to the brief this work started from: that pin was described as `e49dea8f`.
`git rev-parse HEAD` returns `9add32a9`; `e49dea8f` is an ancestor commit, not the pin.)*

---

## What the preflight actually found on .201 (2026-09-18)

`install-worktree-prune.sh --preflight` is read-only and changes nothing. Run against
the host it reported:

```
  OK    worktrees root exists (163 ticket dirs)
  OK    classifier present
  FAIL  ledger missing: /home/jonah/Code/omni_home/docs/tracking/ROLLING_WORK_LEDGER.md
  OK    interpreter: /home/jonah/Code/omni_home/omniclaude/.venv/bin/python
```

163 ticket dirs matches the census exactly.

### The interpreter is fine — an earlier reading of this was wrong

An earlier survey of this host concluded that no interpreter could import
`omniclaude.hooks.lib.worktree_prune_policy` (no `uv`, `pydantic_settings` absent). That is
**false**, and it is corrected here rather than left standing:

```
$ PYTHONPATH=.../omniclaude/src .../omniclaude/.venv/bin/python     -c 'import pydantic_settings, omniclaude.hooks.lib.worktree_prune_policy'
import OK pydantic_settings
```

The venv's `bin/python` is a symlink to the system `python3.12`, which reads as "empty" at a
glance, but its site-packages carry the dependency. The preflight proves each candidate by
**running the import**, never by inspecting the path — which is why it got the right answer
where an eyeball did not.

### The ledger is the real blocker, and it blocks by design

`.201`'s `omni_home` clone is pinned at `0e0ba40f` ("morning update", **2026-05-24**) and has
no `docs/tracking/ROLLING_WORK_LEDGER.md`. A `docs/tracking/` directory exists, holding files
from March; the ledger was never in this clone.

The classifier **refuses to run without a readable ledger**, and that refusal is correct: a
prune with no claim-awareness is exactly the OMN-15551 hazard, and it is the entire safety
difference between this pass and the two it retires. Failing closed here is the design
working.

So the timer is **not armed on `.201` yet**. Remaining work, as a follow-up rather than a
silent gap:

1. give that host a current ledger — either advance the `omni_home` clone (a change to a
   surface outside `omni_worktrees`, so it takes its own approval), or have the unit fetch
   just that file before each run;
2. re-run `--preflight` until it passes;
3. run `install-worktree-prune.sh`.

Until then `onex-worktree-reaper.timer` and the `worktree-gc.sh` line remain live on the
host — but, per finding 2 above, both are pointed at a root holding one stale entry, so
neither is doing anything. The host is not losing protection it had.

---

## Verifying the disposition on the host

```bash
# the new timer exists and its unit invokes the ledger-aware classifier
systemctl --user list-timers --all | grep onex-worktree-prune
systemctl --user cat onex-worktree-prune.service | grep -A6 '^ExecStart'

# the reaper is gone, not merely stopped
systemctl --user list-unit-files | grep onex-worktree-reaper || echo "retired (absent)"

# the disk-gc unit no longer runs a worktree pass
systemctl --user cat onex-disk-gc.service | grep -c 'worktree-gc.sh'   # expect 0 ExecStart hits
```
