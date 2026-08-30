# ONEX CLI invocation contract (OMN-17190)

**Use `$OMNI_HOME/omnibase_infra/scripts/onex`. Do not invoke the CLI through
`uv run`, and do not rely on a bare `onex` resolving through `PATH`.**

```bash
# ~/.zshrc
alias onex='$OMNI_HOME/omnibase_infra/scripts/onex'
```

## Why the previous alias was unsafe

The documented invocation used to be:

```bash
alias onex='uv run --project $OMNI_HOME/omnibase_infra onex'   # SUPERSEDED
```

`uv run` does not pin the command to the project environment. It syncs the
project, prepends that environment's `bin/` to `PATH`, and then resolves the
command name normally — so whenever `<project>/.venv/bin/onex` is not
resolvable, uv **silently executes the first `onex` on the inherited `PATH`**.
No warning, no non-zero exit, no mention in the output.

Reproduced 2026-08-30 on the primary Mac: with `<project>/.venv/bin/onex`
hidden, `uv run --project <project> onex node <name>` ran `~/.local/bin/onex`
— a `uv tool` environment on Python 3.13 carrying:

| what | value | consequence |
| -- | -- | -- |
| `omnibase_infra` | `0.38.11` | `check_omnimarket_drift()` there is `(omni_home, *, allow_drift)` — **no `reconcile` parameter exists**, so the OMN-17190 self-heal cannot run |
| `omnimarket` | `0.4.10`, from PyPI | no `direct_url.json`, so `installed_omnimarket_commit()` returns `None` and the guard reports `omnimarket is NOT INSTALLED from git in this interpreter` |
| interpreter | `~/.local/share/uv/tools/omnibase-core/bin/python` | knows nothing about the real CLI venv, so its verdict is the same whether that venv is drifted or `IN_SYNC` |

That is the whole of the OMN-17190 verification failure — the verbatim
pre-OMN-17190 refusal, no evidence a reconcile was attempted, and a failure
against a venv confirmed `IN_SYNC` moments earlier. It was never a drift bug.

It was **not** an implicit-sync bug either, and that hypothesis is recorded here
because it is the intuitive one: `uv run`'s sync is *inexact*, so it does not
remove the composed `omnimarket` layer. `uv run -v` logs
`DEBUG Unnecessary package: omnimarket==0.4.11 (from git+…)` and leaves it
installed; 14/14 instrumented drift→`uv run` dispatches self-healed correctly
when the project entrypoint was present. The failing variable is interpreter
identity, not sync mode, so `--no-sync` would not have fixed it.

## What the wrapper guarantees

`scripts/onex` execs `$OMNI_HOME/omnibase_infra/.venv/bin/onex` **by absolute
path**. There is no `uv` in the invocation path at all, therefore no implicit
sync, no environment selection, and no fallback.

* entrypoint present → `exec` it, always.
* entrypoint missing → run `scripts/reconcile-workspace-venvs.sh` **once**, then
  re-check and `exec`. Same self-heal-then-proceed policy as the in-CLI drift
  guard, and the same single owner of repair policy.
* still missing → refuse (exit `2`) with a message naming the missing path, the
  reconcile command, and the `PATH` `onex` it deliberately did **not** run.

`OMNI_HOME` is honoured when set and otherwise derived from the wrapper's own
location — exact, not guessed, because the file lives at
`<omni_home>/omnibase_infra/scripts/onex`. That keeps it usable from cron,
launchd, and hooks that never export the variable.

## Shadowing `onex` installs

The wrapper warns on stderr, on every dispatch, when `PATH` resolves `onex` to
something other than the CLI venv's entrypoint. This is deliberate noise: a bare
`onex` in a script, hook, or Makefile — or in any non-interactive shell that
never read the alias — still goes through `PATH` and still runs the other build.
The warning ends when the stale install is removed:

```bash
uv tool uninstall omnibase-core          # ~/.local/bin/onex
command -v -a onex                        # confirm nothing else shadows it
```

A Homebrew-installed `onex` (`/opt/homebrew/bin/onex`) is the same hazard and is
removed the same way, through whatever installed it.

## Related

* `scripts/reconcile-workspace-venvs.sh` — the two-layer venv reconciler
  (lock-governed layer + composed `omnimarket` provider layer) and the single
  owner of repair policy.
* `src/omnibase_infra/cli/omnimarket_drift_guard.py` — the in-CLI detect-then-heal
  guard. It works; it just has to be the code that runs.
* OMN-14060 (drift guard), OMN-16366 (reversed drift), OMN-16262 (`COMPAT_PIN`
  downgrade), OMN-17190 (this contract).
