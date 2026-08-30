# Lab pre-push host table + distribution (OMN-16991)

Replaces `docs/runbooks/200-build-lane-execution-pattern.md`, which the pre-push
hook's refusal text cited for months and which **has never existed in this
repo** (OMN-16446, OMN-16968 "adjacent runbook gap"). If you followed a `die()`
message here, this is the page it meant.

## What changed

Before OMN-16991 the heavy (full-suite) pre-push branch tested exactly two
hostnames:

```bash
if [ "$lc_host" = "$lc_target" ] || [ "$lc_host" = "$lc_201" ]; then
```

That literal `||` — not policy — was the entire reason `.101` and `.105` could
not run a pre-push suite. The hook also *probed* the other host and then
interpolated the answer into a string in the refusal message; it never executed
anything remotely. `.201` was reachable only by a human reading the `die()` text
and hand-driving a bundle recipe.

Now: host identity comes from a committed table, placement picks the
least-loaded host that has proven a **free slot**, and the hook actually ships
the work.

## The table — `scripts/hooks/prepush_hosts.tsv`

Read from `git show HEAD:scripts/hooks/prepush_hosts.tsv`, **never** the working
tree. If the working copy differs from HEAD the hook refuses. An uncommitted row
whose `hostname` matched your laptop would otherwise self-designate it as an
authorizing gate host with no review, no receipt and no guard tripped — the
forgeable-on-disk-artifact surface OMN-16688 deliberately avoided.

| column | meaning |
|---|---|
| `label` | row id; also names `PREPUSH_HOST_OVERRIDE_<LABEL>` |
| `role` | `capacity` = placement target + identity · `identity` = identity only |
| `hostname` | `hostname -s`, lowercased |
| `ssh_target` | probe/execution target |
| `uv_abs_path` | **absolute** uv path — uv is on no host's non-interactive PATH |
| `uv_min_version` | floor; below it the host is skipped, never assumed fit |
| `workroot` | remote scratch root; also where that host's `LOCK` lives |
| `slot_mode` | `queue` (.201) · `lockdir` · `none` |
| `repos_denied` | repos this host must not run |
| `mode` | `authorizing` · `shadow` (runs + receipts, never authorizes) · `disabled` |

### Current fleet (probed live 2026-08-30, non-interactive `ssh <host> '<cmd>'`)

| label | host | cores | load1 | uv | mode | why |
|---|---|---:|---:|---|---|---|
| `h200` | `stickybeatz-studio` (.200) | 24 | varies, often >1.0x | 0.11.32 | authorizing | rule-11a default gate host |
| `h201` | `omninode-pc` (.201) | 32 | 14.08 → 0.44x | 0.11.5 | authorizing | **denied for `omnibase_infra`** until OMN-16989 closes |
| `h201c` | `gate-runner-201` (container) | 32 | — | — | authorizing | identity only; the container has no sshd (OMN-16446) |
| `h101` | `stickybeatz` (.101) | 12 | 2.78 → 0.23x | 0.12.7 | **authorizing** | promoted OMN-17161 — see below |
| `h105` | `omnibook` (.105) | 10 | 1.76 → 0.18x | 0.11.8 | **authorizing** | net-new capacity; promoted out of shadow — see below |

> `h101`'s `hostname -s` prints **`Stickybeatz`**, not `stickybeatz.local`; the
> table's `hostname` column carries the correct lowercase `stickybeatz`
> (identity is matched case-insensitively). Re-probed 2026-08-30 after
> `uv self update`: uv moved 0.8.3 → 0.12.7, above the 0.11.0 floor, load1
> 2.78/12 = 0.23x, ~89GB free on `/System/Volumes/Data`. Promoted OMN-17161,
> same reasoning as `h105` below — a shadow row can never authorize, so the
> promotion is proven by a real full-suite dispatch to `h101` rather than a
> preceding shadow day.

### Why `h105` was promoted rather than run as a shadow

A shadow host **cannot** complete a shadow day: the transplanted tree carries
this repo's own `conftest.py` → `scripts/hooks/pytest_full_suite_host_guard.py`,
which refuses a full-suite target on any host outside the **authorizing** set.
Every heavy dispatch to a shadow host therefore exits nonzero at
`pytest_configure` and writes a receipt whose `pytest_exit != 0` is
indistinguishable from a genuine red. "Run in shadow until it records zero
verdict mismatches" was unreachable by construction — the shadow host could
never record a green.

`h105` is the only net-new host, so leaving it in shadow meant this whole
mechanism added **zero** pre-push capacity. It was probed fit (uv 0.11.8, load
0.18x, 153 GB free) and promoted under the operator's lab-wide-distribution
directive. Both guards read the same committed table, so the promotion takes
effect on the bash side and the pytest side at once.

`.101`'s workroot is `/Users/Shared/onex-prepush`, **outside** the TCC-protected
tree — verified writable over ssh while `ls ~/Code` still returns `Operation not
permitted`. The bundle design never needs `~/Code` on a remote host, so no
System-Settings Full-Disk-Access step is required. Only the uv upgrade gates it.

## Why `.201` reports 0.44x and is still skipped

Measured 2026-08-30: `.201` had the **fittest load ratio in the lab** (14.08/32)
while running **three concurrent pre-push suites** behind a **10-deep queue**.
`load1` is a CPU-time proxy; the scarce resource is an exclusive heavy-suite
slot. So placement reads slot state **first** and load only as a tiebreaker
among hosts already proven free. A host with a held lock, a non-empty queue, or
a live foreign `prepush_smart_tests.sh` is **unfit (rc 3)**, not low-ranked.

## The slot, and why it is one mutex and not two

`.201` already serializes via `~/push-lanes/QUEUE` + `queue-runner.sh`
(OMN-16295). Its gate 1 is *"no other `prepush_smart_tests.sh` running
host-wide — covers foreign runs not launched through this queue"*.

The remote wrapper this hook ships is therefore **named
`prepush_smart_tests.sh`**, so a distributed run is visible to that existing
gate. The queue and the hook share one mutex rather than the hook becoming
another foreign detached run — the exact defect class OMN-16968 is open against.

Locking is `mkdir(2)`, on every host, deliberately: `flock(1)` is **absent on
both Macs** (probed live) and its fd-holding idiom needs `exec {fd}<>`, which
macOS system bash 3.2 cannot parse. One implementation beats a Linux path and a
Mac path that drift. What `mkdir` lacks versus `flock` is auto-release on death,
so the holder pid is recorded and a lock whose holder is provably gone **on the
same machine** is reclaimed — otherwise one externally-SIGTERMed run (OMN-16713)
would wedge a host forever.

The **local** heavy path now takes this same lock. It never took one before,
which is why five concurrent full suites once ran on one host with one taking
97+ minutes (OMN-16174). It was the busiest path in the hook and the only
unserialized one.

## Precedence — strongest evidence first

1. **GitHub-hosted sha-pinned FULL-suite pass** (OMN-16688). Uncontended,
   full-suite shaped, re-derived live from the API, no file on disk to forge.
   It stays first: routing the lab leg ahead of it would silently demote the
   strongest evidence the hook has.
2. **A designated lab host running this exact tree** (OMN-16991). Weaker than
   (1) — the tree is materialized elsewhere — but far stronger than (3).
3. **Single-use receipted degraded-capacity grant.**
4. **`die()`.**

A **remote RED refuses the push immediately** and never falls through to (3). A
red suite satisfied by minting a grant would be a bypass wearing the word
"fallback".

A **shadow** row is never a placement candidate for a verdict-bearing run at
all. Placement filters on `mode` **before** it probes: ranking on load alone let
the idlest host win regardless of mode, and a shadow verdict cannot satisfy the
escalation, so the run cost a bundle + `scp` + `uv sync` + a full suite and was
then discarded — while the authorizing host that could have answered went
unprobed. `pick_capacity_host` takes the eligible mode as a parameter
(`authorizing` at the verdict-bearing call site), and `prepush_remote_run` keeps
its shadow refusal as a second line of defence.

Placement returns a **ranked list**, not a single winner. A candidate that fails
to produce a verdict — unreachable on arrival, no completion marker, or its slot
taken between the probe and the run (wrapper exit 94) — advances to the
next-best host. Only a verdict, green or red, ends the walk; a remote RED still
refuses immediately and never shops for a greener host.

## Verdict readback: a marker, not the ssh exit code

`ssh` returns 255 on transport failure (indistinguishable from a test failure),
and any backgrounding/`nohup`/`tee` wrapper returns 0 with nothing having run —
a fail-**open** shape. The remote leg writes a `MARKER` carrying
`{head_sha, argv_sha, exit, collected, log_sha256}`; the hook reads it back and
requires it to bind to this tree **and** this argv. Absence or mismatch is **no
evidence** and falls through to refusal.

The argv is transferred verbatim per call site. The two local sites differ: the
heavy site runs `tests/unit/` **plus** the allowlisted service-free integration
paths (OMN-16825 — an escalation must never run fewer of the impacted tests than
the narrowing it replaces), the whole-suite-equivalent site runs the selection.
Shipping only `tests/unit/` would silently drop `tests/integration/chains/`.

## Recursion: the remote command re-arms both guards

`ssh` forwards neither `ONEX_PREPUSH_HOOK_ACTIVE` nor the env scrub. Without
re-arming, the remote repo's own suite — which subprocesses this hook from
`tests/ci/test_prepush_hook_host_identity_guard.py` and siblings — would take
**first-entry** behavior on the remote host, resolve the selector, pick a host,
and ship another bundle: an unbounded *distributed* variant of the
OMN-16425/OMN-16489 F-01 recursion (~9h03m, 44,064 tests). The wrapper therefore
unsets every `PREPUSH_*` name and `ENABLE_SMART_TESTS` and exports
`ONEX_PREPUSH_HOOK_ACTIVE` naming the originating host and pid.

## The `.201` identity fix (and what it was really about)

The landing lane recorded this as *"`scrub_prepush_override_env()` unsets
`PREPUSH_*` before `exec uv run pytest`, so `PREPUSH_201_GATE_RUNNER_HOSTNAME`
doesn't reach the inner host guard."*

The **inner** guard is `scripts/hooks/pytest_full_suite_host_guard.py`, a pytest
plugin that re-checks host identity inside the suite the hook spawns. It read
`.201`'s identity straight off the env, and `.201`'s real `hostname -s` is
`omninode-pc`, not the container's `gate-runner-201` — so `~/push-lanes`'s
runner exported the override, the bash guard passed, and the push was then
refused by its own pytest child.

**The scrub is not the bug and is not weakened.** An inheritable `PREPUSH_*`
override crossing into the pytest tree is precisely what turned one sanctioned
grant into a recursive 44,064-test launcher. The bug was sourcing host
*identity* from an environment variable that must not cross a process boundary.
Both guards now read the committed table, which needs no inheritance at all, and
`omninode-pc` is a first-class row. The `~/push-lanes` runner's
`PREPUSH_201_GATE_RUNNER_HOSTNAME` export is now redundant (still honored).

## Overrides REPLACE a row, they never add a name

`PREPUSH_200_HOSTNAME` replaces row `h200`; `PREPUSH_201_GATE_RUNNER_HOSTNAME`
replaces row `h201c`; `PREPUSH_HOST_OVERRIDE_<LABEL>` replaces any row. This is
load-bearing: an override that merely *appended* a hostname could no longer
**de-designate** the local machine, silently inverting the OMN-15059 guard that
`test_guard_refuses_full_suite_escalation_on_non_200_host` proves by forcing a
nonsense hostname.

## The exclusive heavy-suite slot, on BOTH sides

`mkdir(2)` at `<workroot>/LOCK` is the lock primitive on every host — `flock(1)`
is absent on both Macs and its fd idiom needs `exec {fd}<>`, which bash 3.2
cannot parse. What `mkdir` lacks is auto-release on death, so the holder's pid
and machine name are recorded and a lock whose holder is provably gone **on that
same machine** is reclaimed; a holder record from anywhere else is never reaped.

Both legs take it:

* the **local** heavy path, which took no lock of any kind before OMN-16991
  (OMN-16174: five concurrent full suites on one host, one of them 97+ minutes);
* the **remote** wrapper, on the target host, acquired before the clone and
  `uv sync` and released by an `EXIT` trap. Acquiring it on the target is what
  closes the local/remote overlap — a local push on `.200`/`.201` could
  otherwise start while a transplanted suite was mid-run there. If the slot is
  already held the wrapper exits **94** without running anything, and the
  dispatcher treats that as a placement miss and tries the next ranked host.

The remote leg also reclaims its transplanted tree (`runs/<id>/tree`, roughly
0.5 GB per run once `uv sync --all-extras` has run) and prunes run directories
older than three days. The small artifacts — `MARKER`, `suite.log`, `sync.log` —
are kept as the audit trail behind the receipt. On a remote RED the last 200
lines of that host's `suite.log` are fetched and streamed back prefixed
`[<label>]`, because the refusal tells the developer to read exactly that.

## Host prerequisites, and why PATH is one of them

A non-interactive `ssh <host> '<cmd>'` session gets a **minimal** PATH — measured
on `omnibook` it is literally `/usr/bin:/bin:/usr/sbin:/sbin`, with neither the
Homebrew prefix nor `~/.local/bin` on it. This suite shells out to tools by
**bare name** (`uv` in `tests/unit/infra/test_catalog_cli.py`, `shellcheck` in
the shell-hygiene gate tests), so a transplanted run fails in ways the same tree
never fails locally.

Measured: the first full-suite dispatch to `omnibook` collected **24,872** tests
and returned **8 failures**, every one a `FileNotFoundError` for a tool that
*was* installed on that host and simply absent from the ssh PATH. The remote
wrapper therefore prepends the uv directory, `/opt/homebrew/bin`,
`/usr/local/bin` and `~/.local/bin` before it runs anything. A false red here
hard-blocks a push, so PATH parity is part of the verdict meaning something.

Before promoting a host, confirm over a **non-interactive** ssh that
`shellcheck`, `git` and the row's `uv_abs_path` all resolve once that prefix is
applied. A tool that is missing outright (not merely off PATH) will produce a
false red; deny the affected repo on that row rather than accept it.

## Receipts

One JSONL line per remote run to `.onex_state/prepush_distribution/receipts.jsonl`:
`{ts, repo, head_sha, chosen_host, chosen_label, host_mode, host_load_ratio,
all_probed_ratios, selection_paths, pytest_exit, collected, duration_s,
suite_log_sha256}`. `all_probed_ratios` puts **every** probed host on the record,
so a refusal can be audited instead of believed.

## Adding or promoting a host

1. Fix whatever the table's `note` names (e.g. `.101`: `uv self update` past the
   floor), then re-probe **non-interactively** (`ssh <host> '<cmd>'`, never a
   login shell) and record the numbers in the row's `note`.
2. Confirm `hostname -s` on that host matches the `hostname` column exactly,
   lowercased. A dotted or stale value fails identity silently.
3. Run the full suite there once over the real leg and read the receipt. Expect
   first-run triage: a suite that has only ever run on 24- and 32-core hosts can
   meet timing and `nproc`-sensitive failures on a 10-core Air (cf. OMN-16297,
   OMN-15609). If a repo produces host-coupled failures there, add it to that
   row's `repos_denied` rather than accepting a false red.
4. Set `mode` to `authorizing`, commit, and update
   `tests/unit/scripts/test_prepush_host_table.py` — the table contents are
   asserted, so promotion requires a reviewed commit **and** a deliberate test
   change.

`shadow` remains a supported mode for a row you want probed and receipted by an
explicit operator dispatch (`pick_capacity_host <host> <repo> shadow`), but it
is not a step on the promotion path: see "Why `h105` was promoted" above.

## Scope note

This ships in `omnibase_infra` only. `omnibase_core` (549 L) and `omnimarket`
(559 L) carry seam copies that contain **zero** occurrences of
`reject_inherited_env_overrides` (OMN-16480), `consume_override_grant`,
`remote_full_suite_verified` (OMN-16688) or `filter_prepush_runnable_paths`
(OMN-16825) — in those two repos the entire fallback after an unfit host is
`die()`. Adding a "remote host is fit -> run there -> allow push" branch there
first would insert a brand-new **PASS** path into repos that today only refuse,
with no entry rejection behind it. Port OMN-16480 and the OMN-16688 verify path
into core/market first; the host table is a shared data file, so only the picker
block duplicates.

## This is a stopgap

The ONEX-native target is owned by
`docs/plans/2026-08-27-distributed-compute-consolidated-plan.md` and its children
(OMN-16737 / OMN-16741 / OMN-16739), whose execution gate is **CLOSED**. This
change deliberately does **not** delete the local heavy path or the `die()` —
that is OMN-16523 rung R4 and is not authorized.
