# The `onex` CLI entry point — OMN-17190

**Use `$OMNI_HOME/omnibase_infra/scripts/onex`.** That is the invocation
contract. Everything else on this page explains why the alternatives are not.

```bash
# alias (interactive shells)
alias onex='$OMNI_HOME/omnibase_infra/scripts/onex'

# or put it on PATH ahead of any sibling install
ln -sf "$OMNI_HOME/omnibase_infra/scripts/onex" ~/bin/onex
```

## What the wrapper does

It resolves exactly one interpreter and `exec`s it:

```
$OMNI_HOME/omnibase_infra/.venv/bin/onex
```

No PATH lookup, no `uv run` (so no implicit sync at all), no fallback. If the
workspace venv's entry point is missing it runs
`scripts/reconcile-workspace-venvs.sh` once to build it and then execs that —
bootstrap, not fallback. If the venv cannot be produced it refuses and names
the command to run by hand (`EX_UNAVAILABLE`, 69); with `OMNI_HOME` unset it
refuses without guessing a root (`EX_CONFIG`, 78).

## Why an alias was not enough

`alias onex='uv run --project $OMNI_HOME/omnibase_infra onex'` only exists in
an **interactive** shell. Scripts, hooks, Makefiles, CI steps and agent tool
calls run non-interactively, so `onex` there resolves through `PATH` — and on
a development machine `PATH` is not empty. Measured 2026-08-30 on the author's
Mac, **with the workspace venv confirmed `IN_SYNC`**:

| Resolved binary | What it actually is | Result |
| --- | --- | --- |
| `~/.local/bin/onex` | a `uv tool` shim (omnibase-core env): omnibase_infra **0.38.11** (pre-self-heal) + omnimarket **0.4.10 from PyPI** | verbatim pre-OMN-17190 hard refusal, **no reconcile attempted** — it is old code |
| `/opt/homebrew/bin/onex` | brew python3.13 global site-packages: omnibase_infra + omnimarket installed **editable** from the canonical clones | self-heal ran, repaired `.venv`, re-probed **itself**, refused again — structurally non-convergent |
| `.venv/bin/onex` | the workspace CLI | self-heals and dispatches, 100% |

`bash -lc 'onex skill …'` resolved the first row. That — not `uv run` — is why
the OMN-17190 verification lane saw the self-heal succeed only 5 times in 15.

**The `uv run` hypothesis is falsified.** On uv 0.11.32 the implicit `uv run`
project sync is **inexact**: `uv run -v` logs
`Unnecessary package: omnimarket==0.4.11 (from git+…)` and *keeps* it, while
still repairing lock-layer pins in the same pass (observed: `idna 3.19 → 3.15`
with omnimarket untouched). `uv run` never stripped the co-install.

## Two guard fixes that came with the wrapper

1. **An editable omnimarket installed from the canonical clone is not drift.**
   An editable install records `dir_info` and no `vcs_info`, so the commit
   probe returned `None` — the same answer it gives for "absent" and for "a
   PyPI wheel". But such an interpreter imports `$OMNI_HOME/omnimarket`'s
   working tree directly and is therefore at clone HEAD by construction. It is
   now recognised instead of refused.
2. **A foreign interpreter refuses instead of reconciling.** The reconciler
   repairs `$OMNI_HOME/omnibase_infra/.venv` and only that. When the running
   interpreter is something else, the guard now raises a typed refusal naming
   the running prefix, the venv that would have been repaired, and this
   wrapper — rather than mutating a venv the operator never named and then
   failing anyway.

Interim by design: the successor named on OMN-17190 is a `NodeCompute`
drift-detect handler behind a `NodeEffect` reconcile publisher. The interpreter
identity is one more pure input to that handler.
