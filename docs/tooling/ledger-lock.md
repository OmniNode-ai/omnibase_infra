# ledger_lock.py — shared ledger mutex

`scripts/ledger_lock.py` is a small, dependency-free CLI that serializes writes to a
shared, append-only ledger file. It exists for one problem: several agents or processes
(human or automated) working against the *same* ledger file at the same time, each
wanting to "claim a piece of work, then write to the ledger recording the claim" without
two writers interleaving mid-write or clobbering each other's rows.

Any team can point it at their own ledger file — it takes the ledger path as an
argument and has no built-in notion of which team or project owns it.

## What it does

1. Acquires a mutex scoped to the ledger's own path (an atomically-created lock
   directory — this works on macOS without `flock(1)` and needs no third-party
   dependencies).
2. While holding the lock, performs exactly one of:
   - `--append TEXT` / `--append-file PATH` (`-` for stdin): appends the text durably
     (the write is `fsync`'d before the lock is released).
   - `-- COMMAND ...`: runs an arbitrary command (e.g. an editor) while holding the
     lock, for an edit that isn't a pure append.
3. Releases the lock.

## Row convention (a convention, not something this script enforces)

Ledgers of this kind typically carry three row shapes, so concurrent writers can grep
the file to see who is doing what before claiming new work themselves:

- **CLAIM** — "I am about to do X." Written before starting work, so a second writer
  sees the claim and picks something else instead of duplicating effort.
- **PROGRESS** — zero or more rows recording what happened while the work was in
  flight (useful for a long-running or multi-step piece of work).
- **TERMINAL** — "X is done," with evidence (a PR link, a commit SHA, a test run) —
  or a note that the claim was abandoned/superseded.

This script is agnostic to row shape. It serializes and durably persists whatever text
a caller appends; the claim/progress/terminal convention above is something callers
adopt, not something `ledger_lock.py` parses or validates.

## The exit-75 retry contract

| Exit code | Meaning |
|-----------|---------|
| `0`   | Success. Includes an `--append` that was skipped because it duplicated the last `--dedup-window` lines already on disk (see below) — that is a normal, successful no-op, not an error. |
| `2`   | Usage error (e.g. neither `--append`/`--append-file` nor `-- COMMAND` given, or both given). |
| `75`  | Timed out waiting for the lock (`EX_TEMPFAIL` in `sysexits(3)`). Someone else holds it. **This is the signal to retry** — the same command, unmodified, some time later. |
| `127` | The `-- COMMAND` could not be started at all (e.g. binary not found). |
| *n*   | Whatever `-- COMMAND` itself exited with, when it started and ran. |

The retry-on-75 contract only works safely because of the **dedup-window** check: before
writing, the payload (with its bullet and leading UTC timestamp stripped, so a few
seconds of clock drift between attempts doesn't matter) is compared against the last
`--dedup-window` lines (default 20) already on the ledger. A caller that retries an
identical `--append '<same text>'` after an exit-75 — not knowing whether the *first*
attempt's write actually landed before the process was interrupted — gets a clean `0`
either way: either the write happens for the first time, or it is recognized as a
duplicate of what's already there and skipped. Callers should therefore treat exit 75 as
"retry the same command," never as "something is wrong."

Stale locks are also broken automatically, so a crashed writer can't wedge the ledger
forever:

- A lock written by a process on the **same host** that is no longer running is broken
  unconditionally, the moment another writer notices it.
- A lock **older than `--stale-after`** (if you pass that flag) is broken regardless of
  which host wrote it, since liveness can't be checked across hosts.

## Usage

```bash
# Append a row, waiting up to the default 5 minutes for the lock.
scripts/ledger_lock.py path/to/ledger.md --append '- 2026-...: claimed X'

# Same, but from a file or stdin.
git diff -- path/to/ledger.md | scripts/ledger_lock.py path/to/ledger.md --append-file -

# Open the ledger in an editor while holding the lock, instead of a pure append.
scripts/ledger_lock.py path/to/ledger.md -- "${EDITOR:-vi}" path/to/ledger.md

# Tune the wait and the dedup window.
scripts/ledger_lock.py path/to/ledger.md --timeout 30s --dedup-window 50 --append '...'
```

By default, lock directories are created next to the ledger file itself (a
`.ledger_locks/` sibling directory), so no shared location needs to be agreed on
up-front. Set `LEDGER_LOCK_ROOT` to point every writer at one shared directory instead
(only necessary if the ledger's own parent directory isn't writable by every writer).

## Scope — what this script deliberately does *not* do

This is the generic mutex/append/dedup/retry mechanism only. It does not validate row
shape, enforce a cost/estimate on claim rows, or otherwise inspect the *content* being
appended beyond the dedup comparison above. If your team wants stronger row-shape
enforcement (e.g. every claim row must name an estimate, or must cite a tracking ticket),
build that as a wrapper around this script rather than expecting `ledger_lock.py` to
grow team-specific validation — that keeps the shared tool generic and keeps your
team's own conventions in your own repo.

## Origin

This script originated as `omni_home/scripts/ledger_lock.py`, an internal multi-agent
work-ledger tool with team-specific claim-row validation layered on top of the same
core mechanism. That internal copy is intentionally left as-is rather than migrated to
delegate here — it's cited by absolute path in a large number of live, in-flight
prompts, and switching it out from under them is a needless risk for no immediate
benefit. This `omnibase_infra` copy is the generic core only, with the team-specific
validation left behind. Consolidating to a single copy (e.g. having the internal tool
delegate to this one) is a reasonable future cleanup once nothing is depending on the
internal copy's exact behavior mid-flight — not something this change needed to force.
