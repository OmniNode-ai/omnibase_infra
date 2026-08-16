# C2b — routing uv's git dependency at the local mirror (OMN-16063)

Deployed 2026-08-15T01:37Z on the runner host (`omninode-pc`, 192.168.86.201).
Extends the OMN-14027 C2 git-mirror component. Host-side only: no workflow file
changes, no container recreate, no image rebuild.

## What problem this closes

C2's pre-seed warms the workspace for the job's **own** repository, so it only
accelerates `actions/checkout`. It does nothing for the other full clone every
job performs: every `uv sync --no-cache` re-resolves

    onex-change-control @ git+https://github.com/OmniNode-ai/onex_change_control.git@<sha>

directly from github.com. That is a ~93MB object graph, `ci.yml` invokes
`uv sync` in 9 jobs, and `--no-cache` means uv resolves into a throwaway cache
directory every time — so there is nothing on the runner to reuse and no
uv-level cache that could be pre-seeded. Under wave load, ~40 concurrent jobs
NAT'd through one uplink, this is the dominant remaining source of the
GnuTLS/`early EOF` clone churn that C2 exists to remove.

`omnibase_core` is deliberately **not** rewritten — see "Scope" below.

## Mechanism

A fetch-only `url.<mirror>.insteadOf`, exported through `GITHUB_ENV` as
`GIT_CONFIG_KEY_*` / `GIT_CONFIG_VALUE_*` / `GIT_CONFIG_COUNT` by
`wire_uv_git_mirror_rewrite()` in `runner-job-started.sh`.

Chosen over writing a gitconfig file into the containers because nothing is
written inside the 72 runners: the rewrite exists only in the job's own
environment and evaporates when the job ends. Reverting is restoring one host
file — there is no per-container residue to chase. uv shells out to `git`, and
git honours `GIT_CONFIG_COUNT`/`KEY`/`VALUE` from the environment, so the
rewrite reaches uv's clone.

### Only the `.git`-suffixed URL is rewritten

This is the load-bearing scoping decision.

| consumer | URL form | rewritten? |
|---|---|---|
| uv (from `uv.lock`) | `https://github.com/OmniNode-ai/onex_change_control.git` | yes |
| `actions/checkout` | `https://github.com/OmniNode-ai/onex_change_control` (no suffix) | no |

`actions/checkout` computes `${GITHUB_SERVER_URL}/${owner}/${repo}` with no
`.git` suffix — the same fact C2's pre-seed already depends on. So rewriting
only the suffixed form hits uv and **cannot** touch any checkout, including the
`ci.yml` steps that check out `onex_change_control` as a sibling repo: those
keep their authenticated github.com fetch and their exact-ref semantics.

Verified with `GIT_TRACE=1` in a container off `omninode-runner:latest`:

    no-suffix  -> git-remote-https https://github.com/OmniNode-ai/onex_change_control
    .git       -> From git://172.18.0.1:9418/onex_change_control

There is no blanket github.com rewrite.

### Push is pinned back on github.com by an identity `pushInsteadOf`

The first cut of this component asserted that "no `pushInsteadOf`" meant pushes
were untouched. **That was wrong**, and it was corrected on 2026-08-15.
`url.<mirror>.insteadOf` on its own also rewrites the PUSH url -- documented git
behaviour, not an edge case. Measured with only the insteadOf installed:

    $ git remote -v
    origin  git://172.18.0.1:9418/onex_change_control.git (fetch)
    origin  git://172.18.0.1:9418/onex_change_control.git (push)
    $ git push ...
    fatal: remote error: access denied or repository not exported: /onex_change_control.git

The daemon deliberately serves no receive-pack, so any job that pushed to the
`.git`-suffixed URL would have failed CLOSED -- exactly the class of outcome the
rest of this component is gated to prevent.

git resolves a push URL against `pushInsteadOf` rules FIRST and falls back to
`insteadOf` only when none match. The hook therefore emits a second, IDENTITY
rule per rewritten repo (upstream -> itself), which pins pushes back on
github.com while leaving the fetch redirect intact:

    GIT_CONFIG_KEY_0=url.git://172.18.0.1:9418/onex_change_control.git.insteadOf
    GIT_CONFIG_VALUE_0=https://github.com/OmniNode-ai/onex_change_control.git
    GIT_CONFIG_KEY_1=url.https://github.com/OmniNode-ai/onex_change_control.git.pushInsteadOf
    GIT_CONFIG_VALUE_1=https://github.com/OmniNode-ai/onex_change_control.git
    GIT_CONFIG_COUNT=2

Verified in a container off `omninode-runner:latest`:

    origin  git://172.18.0.1:9418/onex_change_control.git (fetch)
    origin  https://github.com/OmniNode-ai/onex_change_control.git (push)
    by-SHA fetch of the pin -> rc=0 (still served by the mirror)

`gh` API calls and the checkout action's token path remain untouched.

## Fail-open gating (why this cannot fail a job)

`insteadOf` has **no fallback**. Measured 2026-08-14 by pointing the rewrite at
a mirror lacking the pinned commit:

    fatal: remote error: upload-pack: not our ref 2dd26ade...
    -> uv exits 1. It does NOT retry against github.com.

An unconditional rewrite would therefore convert "mirror is stale" into "job
fails", which is strictly worse than the problem being solved. So the hook
installs the rewrite only after proving the mirror can serve the exact commit
this job will ask for:

1. Resolve the pin the job actually uses — `uv.lock` at `GITHUB_SHA`, read via
   a delta fetch against the already-seeded workspace (not the mirror's
   default-branch tip, which a PR may have bumped away from).
2. Probe the mirror with `git fetch --depth=1 --filter=tree:0 <mirror> <pin>` —
   the same by-SHA question uv will ask, but the server sends the commit object
   and nothing else. Measured: present 104ms/132K, wrong repo 46ms, nonexistent
   SHA 52ms, both correctly reported absent.
3. Install the rewrite only on success.

Every other outcome — no `uv.lock`, no pin, pin not servable, mirror
unreachable, `GITHUB_ENV` unwritable, any unexpected error — installs nothing
and leaves the job on today's github.com path.

An `ls-remote` advertised-refs check was tried first and **rejected**: uv pins
are ordinarily mid-history commits, not ref tips, so it reported the real pin
missing and the rewrite never engaged.

## Mirror-side requirements

`git-mirror-refresh.sh` applies these to every mirror on every pass
(`apply_mirror_serving_config`), not just at clone time, so a re-cloned mirror
cannot silently lose them:

- `uploadpack.allowFilter=true` — without it the `--filter=tree:0` probe fails,
  the gate reads "absent", and every job silently forgoes the mirror.
- `uploadpack.allowAnySHA1InWant=true` — uv fetches an exact pinned SHA, usually
  not a ref tip; without this, upload-pack refuses it as "not our ref".

The mirror set (`MIRROR_REPOS`) already contained `onex_change_control` and
`omnibase_core`, refreshed every 2 minutes with `+refs/*:refs/*` (all heads,
tags, and `refs/pull/*`). No new mirror was needed.

## Scope: why `omnibase_core` is not rewritten

- It is **not a git dependency**. `uv.lock` pins `omnibase-core==0.46.8` from
  `pypi.org`, already served by the C1 devpi pull-through cache. uv never
  clones it, so there is no uv clone to redirect.
- Its remaining traffic is `actions/checkout` of a sibling repo (5 sites) and
  one `git clone --depth=1`, all resolving a *branch*, not a pinned SHA. Those
  cannot be gated on an exact commit, and redirecting them would introduce a
  silent-staleness class: the job would get the mirror's tip, up to 2 minutes
  behind github. That is a correctness change for a comparatively cheap shallow
  clone, so it was not made.

To enable it anyway (accepting the above), set on the host:
`OMNI_GIT_MIRROR_REWRITE_REPOS="onex_change_control omnibase_core"`.

## Files changed on the host

    /home/jonah/.omnibase/runners/docker/runners/runner-job-started.sh
      backup: runner-job-started.sh.bak.pre-c2b-20260815T013533Z      (pre-C2b)
      backup: runner-job-started.sh.bak.pre-pushfix-20260815T022031Z  (pre-push-fix)
    /home/jonah/.omnibase/runners/docker/runners/git-mirror-refresh.sh
      backup: git-mirror-refresh.sh.bak.pre-c2b-20260815T013834Z
    /home/jonah/.omnibase/runners/docker/runners/hook-mount-drift-check.sh
      backup: hook-mount-drift-check.sh.bak.pre-deployrunner-20260815T022152Z

## Revert

One line, then propagate to the running containers:

    cat /home/jonah/.omnibase/runners/docker/runners/runner-job-started.sh.bak.pre-c2b-20260815T013533Z \
      > /home/jonah/.omnibase/runners/docker/runners/runner-job-started.sh

To disable without editing anything, set `OMNI_GIT_MIRROR_REWRITE_DISABLE=1`
(the C2 kill switch `OMNI_GIT_MIRROR_DISABLE=1` also disables it).

The mirror-side `git-mirror-refresh.sh` change is additive and safe to leave in
place; it only widens what the daemon is willing to serve.

## MUST READ: bind-mounts are by inode

`docker` binds these scripts by **inode, not path**. Any tool that replaces the
host file by rename — `mv`, `install`, rsync's temp+rename, most editors'
atomic save — leaves every running container bound to the old, now-unlinked
inode. The host file then looks correct to every reviewer while the fleet keeps
executing the previous content.

This happened during this very deploy: `install` was used at 01:35Z and split
the inode for all 72 runners. Always write **in place**:

    cat <new-content> > /home/jonah/.omnibase/runners/docker/runners/runner-job-started.sh

If the inode is already split, repair it through a container's mount — one
write fixes every container that shares the stale inode, and it neither
restarts a container nor disturbs an in-flight job:

    pid=$(docker inspect -f '{{.State.Pid}}' omninode-runner-1)
    sudo nsenter -t "$pid" -m -- mount -o remount,bind,rw /usr/local/bin/runner-job-started.sh
    sudo nsenter -t "$pid" -m -- tee /usr/local/bin/runner-job-started.sh \
      < /home/jonah/.omnibase/runners/docker/runners/runner-job-started.sh >/dev/null
    sudo nsenter -t "$pid" -m -- mount -o remount,bind,ro /usr/local/bin/runner-job-started.sh

Containers recreated at different times may hold *different* stale inodes
(runner-5 did), so always finish with:

    bash /home/jonah/.omnibase/runners/docker/runners/hook-mount-drift-check.sh

and repeat the repair for any container still listed. Post-deploy this reported
`OK 216 mount(s) checked, all match the host copy`.

## Log lines to grep in a job

    [c2-mirror-rewrite] onex_change_control pin <sha12> present on mirror; uv git fetch -> git://172.18.0.1:9418/onex_change_control.git (actions/checkout unaffected).
    [c2-mirror-rewrite] onex_change_control pin <sha12> not served by ...; leaving uv on github.com (fail-open).
    [c2-mirror-rewrite] could not read uv.lock at <sha>; leaving uv on github.com (fail-open).

The first line means the fast path engaged; the others mean the job ran exactly
as it did before this change.


---

# 2026-08-15 follow-up: push scoping + a checker that was lying

Three defects were found while verifying the deploy above. All are fixed and
re-verified; details of the push fix are in "Push is pinned back on github.com"
above.

## 1. The rewrite was not fetch-only (fixed)

See above. One-line revert:

    cat /home/jonah/.omnibase/runners/docker/runners/runner-job-started.sh.bak.pre-pushfix-20260815T022031Z \
      > /home/jonah/.omnibase/runners/docker/runners/runner-job-started.sh

(then propagate to containers -- see "bind-mounts are by inode" above).

## 2. `omninode-deploy-runner` had NEVER received C1, C2 or C2b

It was pinned to the original pre-C2 hook inode
(`runner-job-started.sh.pre-c2.bak`, 2026-08-10) -- no devpi wiring, no mirror
pre-seed, no uv rewrite. Its job logs showed `[runner-job-started] Resetting
workspace` and then nothing, which is what exposed it.

This was invisible because the drift checker only scanned
`^omninode-runner-`, and `omninode-deploy-runner` does not match that prefix.
The checker reported `OK 216 mount(s) checked, all match the host copy` while
one container it never looked at was five days stale. **A checker whose scope
is narrower than the thing it certifies manufactures false confidence.**

Repaired through the container's own mount (no restart, no in-flight job
disturbed). It now runs the same hook as the fleet.

## 3. Drift-check coverage gap (fixed)

`hook-mount-drift-check.sh` now selects on a regex covering both the numbered
fleet and the deploy runner:

    CONTAINER_MATCH="${CONTAINER_MATCH:-^omninode-runner-|^omninode-deploy-runner$}"

Coverage went 216 -> 219 mounts (73 containers x 3 scripts). The three new
mounts immediately surfaced a second real drift: `omninode-deploy-runner`'s
`entrypoint.sh` was an older revision. The difference was **comment-only**
(0 non-comment lines, verified by diff), so converging it changed no behaviour
even at next restart; it was repaired the same way.

Revert:

    cat /home/jonah/.omnibase/runners/docker/runners/hook-mount-drift-check.sh.bak.pre-deployrunner-20260815T022152Z \
      > /home/jonah/.omnibase/runners/docker/runners/hook-mount-drift-check.sh

Note this checker runs from cron every 15 minutes
(`>> /tmp/hook-mount-drift.log`), so reverting it re-blinds that cron to the
deploy runner.

Post-fix state: `OK 219 mount(s) checked, all match the host copy`.

## Verification performed (2026-08-15 ~02:20Z)

- Mirror serves the pin in `omnibase_infra` dev's `uv.lock`
  (`2dd26ade7caaa7131e532473ec9d8a207d0e77ab`): full clone from a sibling
  container on `docker_default` = **13.3s / 42MB**, pinned SHA present.
- Hook run end-to-end in a container against the PRODUCTION mirror:
  rewrite installed, `GIT_CONFIG_COUNT=2`, hook rc=0.
- Hook run end-to-end against a scratch daemon (port 9419) serving a real
  `omnibase_infra` mirror but an EMPTY `onex_change_control`, i.e. the exact
  missing-SHA case: `pin ... not served by ...; leaving uv on github.com
  (fail-open)`, **no** `GIT_CONFIG_*` written, hook rc=0. The job degrades to
  github.com; it does not fail.
- Scoping: `.git` URL -> `From git://172.18.0.1:9418/...`; no-suffix URL ->
  `git-remote-https https://github.com/...`; push -> github.com.
