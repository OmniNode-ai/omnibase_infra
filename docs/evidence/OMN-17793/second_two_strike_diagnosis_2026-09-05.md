# OMN-17793 second two-strike diagnosis — 2026-09-05

## Status

**HOLD — no third source correction is authorized.** This fresh reconstruction
worktree is intentionally left dirty and uncommitted for an operator-reviewed
whole-boundary redesign. No source is staged; no commit, push, pull request,
OCC artifact, or runtime action was made.

* Branch: `codex/omn-17793-third-attempt-20260905`
* Fresh-worktree base/HEAD: `3da9e60f4fe4ccdbec78a0d4ae219349d7ac803e`
* Preserved remote reference: `b179d1d3` (immutable; no new remote action)

## The two rejected corrections

1. The first correction introduced fixed `env`/`git` selection for candidate
   identity, but automatic linked-worktree discovery still used caller-PATH
   `command -v git` and raw `git rev-parse`; its TOML parser also used
   caller-PATH `awk`. The test covered only an explicit candidate with a fake
   `git`, not automatic pathless sibling discovery.
2. The second correction moved discovery to `core_checkout_git`, fixed the
   parser to `/usr/bin/awk`, and added a pathless linked-worktree fake-`git`
   regression. Re-review found caller-PATH `basename`, `dirname`, and
   `python3` remained on the candidate-resolution path. More broadly, the
   checker and wrapper retain caller-PATH command resolution described below.

## Complete caller-PATH command inventory

The following inventory is from static inspection of the current draft. Shell
builtins (`cd`, `pwd`, `printf`, `echo`, `read`, `return`, `exit`, `true`) do
not use `PATH`; all non-builtin command names below do unless made absolute.

| Surface | Caller-PATH command(s) | Current locations | Why this is material |
| --- | --- | --- | --- |
| Script interpreter | `bash` via `#!/usr/bin/env bash` | `scripts/check_architecture.sh:1` | The absolute `env` binary resolves `bash` using the inherited `PATH`. |
| Candidate structure and sibling selection | `basename`, `dirname` | `scripts/check_architecture.sh:641-642`, `659`, `661` | A forged result can alter source-layout validation or synthesized sibling path. |
| Installed-package fallback | `python3` | `scripts/check_architecture.sh:713` | A forged interpreter can supply a deceptive package path; this fallback conflicts with the fail-closed source-only requirement. |
| Architecture-check setup | `sed`, `grep`, `sort`, `diff`, `cat` | `scripts/check_architecture.sh:185-204`, `327`, `769`, `802`, `1020` | A forged parser/search/output utility can alter the reported invariant result. |
| Architecture-check scan/count | `find`, `wc`, `tr` | `scripts/check_architecture.sh:831` | A forged utility can report an empty or deceptive scan set. |
| Python wrapper interpreter | `python3` via `#!/usr/bin/env python3` | `scripts/validate.py:1` | Direct execution resolves Python through inherited `PATH`. |
| Python wrapper subprocess | `bash` | `scripts/validate.py:115` | `subprocess.run(["bash", ...])` selects the executable through inherited `PATH`. |

`/usr/bin/env`, `/usr/bin/git`, and `/usr/bin/awk` in the draft are already
absolute for the candidate Git identity helper and are **not** caller-PATH
lookups. That partial trust boundary is insufficient because control reaches
the inventory above before, during, or after candidate selection.

## Root cause

The effort hardened individual commands only after each review observation.
That piecemeal approach did not first define an end-to-end trusted execution
boundary, so each new fake-PATH test exposed the next unresolved command.
Treating the installed-package import as a fallback also conflicts with the
required source-only, fail-loud behavior.

## Correct approach for a later, separately approved attempt

Do not resume from this draft with another local substitution. First choose
and review one whole-boundary design:

1. Establish a fixed, verified tool surface before any work: invoke the
   checker through a fixed shell, use a fixed safe `PATH` for every child
   command, and use only vetted absolute executables for all external tools;
   the Python wrapper must invoke that same fixed shell.
2. Preferably remove external parsing and the installed-package import path
   where shell-only, fail-closed logic suffices. In particular, remove the
   `python3` package-import fallback rather than hardening it. Candidate
   discovery must have an explicit source-layout rule and fail when no valid
   source checkout exists.
3. Treat all identity, discovery, validation, and wrapper invocation as one
   boundary; do not allow a caller-controlled executable or `PATH` value at
   any point in it. Validate the fixed tool paths as present/executable and
   fail loudly if they are unavailable.

This is a design choice to be approved before any further edit, not an
authorized third correction in this attempt.

## Required regression matrix for that later design

With an attacker-controlled `PATH` containing marker-writing decoys for
`env`, `bash`, `git`, `awk`, `basename`, `dirname`, `python3`, `sed`, `grep`,
`sort`, `diff`, `cat`, `find`, `wc`, and `tr`:

1. A normal canonical source checkout and a pathless linked-worktree sibling
   pass; no decoy marker is written and the real sibling is selected.
2. An explicit fake target is rejected; no decoy is invoked.
3. No valid local source checkout fails with exit 2 and cannot become valid
   through a fake `python3` package location.
4. `GIT_CONFIG_COUNT`, every numbered `GIT_CONFIG_KEY`/`GIT_CONFIG_VALUE`,
   `GIT_CONFIG_PARAMETERS`, global URL rewriting, and repository-selector or
   object environment spoofing cannot alter candidate identity.
5. Missing, multiple, or noncanonical local `remote.origin.url` values;
   unborn HEAD; untracked source; nested checkout; symlink; and venv or
   site-package candidates all fail closed.
6. The Python wrapper cannot select a fake `bash`, converts exit 2 to failure,
   and fails if its script is missing.

Run the focused suite, static checks, the real wrapper, strict typing, the full
repository suite, and `pre-commit run --all-files` only after this whole-boundary
design passes independent review.

## Preserved state and evidence

The fresh draft currently changes only the four requested implementation/test
paths plus this diagnosis. It remains unstaged and uncommitted. The prior
source worktree (`jonah/omn-17793-arch-layers-recovery-20260904`, `c8ef2ec`)
and rejected reconstruction worktree (`codex/omn-17793-reconstruction-20260905`,
`b179d1d3`) remain preserved. The ownerless OMN-17888 per-worktree
`index.lock` was observed but never touched.

The last focused checkpoint passed: 25 tests in the two new focused files.
`bash -n`, Ruff formatting/checking, and `git diff --check` passed before the
hold. The real wrapper separately timed out after 120 seconds in the current
host contention condition; strict typing was interrupted when review required
source changes; full and pre-commit gates were not started. No further gate is
authorized under this hold.
