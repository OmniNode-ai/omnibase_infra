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

## Section caps and the archive roll (OMN-17023)

An append-only section grows forever unless something bounds it. The rolling work
ledger this script serves reached **21,752 lines** before a human noticed and split it
by hand — and that hand-split then invalidated every line-number watermark pointing
into the file. Both flags below exist so that stopgap is never needed again.

The **capped section is the tail of the file**: from `--section-heading` to EOF. That
is the shape an append-only section has, and the only shape where "move the oldest
rows out" is well defined. The heading must occur exactly once; zero or two
occurrences fails closed (exit 2) rather than capping the wrong bytes. A **row** is a
markdown heading (`##` … `######`) and everything under it up to the next one. Rows are
the unit a roll moves; lines and bytes are the units a cap counts.

| Flag | Meaning |
| -- | -- |
| `--section-heading HEADING` | the exact heading line opening the capped section |
| `--max-section-rows N` | refuse or roll when the section would exceed N lines |
| `--max-section-bytes N` | refuse or roll when the section would exceed N bytes |
| `--max-append-bytes N` | refuse a single append larger than N bytes |
| `--on-cap {roll,block}` | required with any cap: archive the oldest rows first, or refuse |
| `--archive-dir DIR` | where rolled rows are written (required with `--on-cap roll`) |
| `--roll-keep-entries N` | how many newest rows stay live after a roll |
| `--roll-section` | action: roll now instead of appending (add `--force-roll` to roll while under the caps) |

```bash
# Append under a cap, rolling the oldest rows out when it would be crossed.
scripts/ledger_lock.py LEDGER --append-file - \
  --section-heading '## §5 Action Log (append-only)' \
  --max-section-rows 4000 --max-section-bytes 2000000 \
  --on-cap roll --archive-dir docs/tracking/archive --roll-keep-entries 40

# Roll on its own (what a scheduled or merge-triggered job calls).
scripts/ledger_lock.py LEDGER --roll-section \
  --section-heading '## §5 Action Log (append-only)' \
  --max-section-rows 4000 --on-cap roll \
  --archive-dir docs/tracking/archive --roll-keep-entries 40
```

**A refusal writes nothing at all** — not the row, and not a roll. Exit code
`74` means the cap held. That includes the case where a roll *would* fire but keeping
`--roll-keep-entries` rows still crosses the cap: planning and writing are separate
steps precisely so the tool can decline without leaving a split file and a lost row.
`--max-append-bytes` is refused outright rather than rolled for, because rolling
cannot make a single row smaller.

A roll prints a machine-readable receipt on stdout:

```
ledger_lock: ROLL {"archive": "...", "entries_rolled": 5, "entries_kept": 3,
                   "first_kept_heading": "## ...", "last_rolled_heading": "## ...", ...}
```

`first_kept_heading` is the boundary a reader re-anchors against. The live section
keeps exactly one pointer block (`<!-- ledger-roll: … -->`), replaced on each roll
rather than stacked, naming the archive file the older rows moved to.

## Reading a rolled ledger — `ledger_watermark.py`

A reader that stores its position as a **line number** cannot survive a roll: after a
split the same line number addresses different content, and there is nothing for the
reader to compare against, so it skips rows silently. `scripts/ledger_watermark.py`
stores the identity of the last row read instead — its heading plus a 12-hex digest
over its normalized body — and resolves that against the live file *and* the archive
directory.

```bash
scripts/ledger_watermark.py LEDGER --state STATE.json --source NAME --resolve   # report unread rows
scripts/ledger_watermark.py LEDGER --state STATE.json --source NAME --advance   # then move the anchor
scripts/ledger_watermark.py LEDGER --state STATE.json --source NAME --migrate   # convert a line-number mark, once
```

A roll moves the OLDEST rows out and does not ask whether the reader had read them,
so an anchor that lands in an archive can have unread rows on BOTH sides of the split.
The resolver reports those as `unread_archived_entries` / `unread_archived_headings`
alongside the live `resume_line`; dropping them would be the same silent skip the line
number produced, reached by a different route.

The state file declares `watermark_schema_version` (currently `2`); a state file
without one is the pre-OMN-17023 line-number schema and is **refused** (exit 4) rather
than reinterpreted. Exit 3 is `UNRESOLVED` — the anchor row is gone, or its digest
changed because the row was edited after being read — and nothing is advanced. Both
are fail-closed on purpose: a reader that cannot prove where it stopped must not guess.

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
