<!--
SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
SPDX-License-Identifier: MIT
-->

# BuildKit build-cache ceiling (OMN-16367)

`daemon-builder-gc.json` in this directory is the **standing** ceiling on retained
BuildKit build cache for the `.201` lab host. `scripts/disk-gc.sh` is the
**scheduled** enforcement of the same number, run hourly by `onex-disk-gc.timer`.
They are two halves of one policy.

> **This fragment is NOT applied by anything in this repository.** Installing it
> edits `/etc/docker/daemon.json` and requires a dockerd reload, which is a host
> daemon-scope mutation affecting every lane on the box. That is the operator's
> window. Nothing here should be run by an agent on its own initiative.

## Why it exists

On 2026-09-18 `docker buildx du` on `.201` reported **749.7 GB of build cache,
100% reclaimable, across 9,183 records**, while `/data` free space was tracking
toward zero at a gross ~43 GB/h.

`/etc/docker/daemon.json` on that host carries **no `builder` block at all**, so
BuildKit derives its GC defaults from the size of the volume — a 3.6 TB disk —
and lands on a catch-all rule of:

| field | default derived from disk size |
|---|---|
| `reservedSpace` | 341.8 GiB |
| `maxUsedSpace` | 2.668 TiB |
| `minFreeSpace` | 683.6 GiB |

`reservedSpace` is a **floor**: GC will not evict below it. A 341.8 GiB permanent
cache floor is untenable on a disk that also holds ~674 GB of Docker volumes and
the containerd layer store. The fragment replaces those disk-derived numbers with
ones sized to the lab's actual working set.

## The vocabulary trap — read this before editing the fragment

**Do not rewrite this policy in the old `keepStorage` / `--keep-storage` words.**

`keepStorage` is now an alias for `reservedSpace`, which is a **floor** (space
always allowed to be kept), *not* a target to prune down to. The old flag meant
"prune down to N"; the field it now aliases means "never prune below N". With no
maximum and no free-space target set, BuildKit has nothing to prune toward.

Measured live on `.201`, 2026-09-18:

```
$ docker builder prune --force --keep-storage 200GB
Flag --keep-storage has been deprecated, keep-storage flag has been changed to reserved-space
Total:	0B
```

Exit code 0. Nothing reclaimed. **A caller reading only the exit status records a
successful prune that freed nothing** — the worst shape a disk-pressure remedy
can take. `maxUsedSpace` is the operative ceiling.

Evidence, verbatim commands and readbacks: `ROLLING_WORK_LEDGER.md` rows 3921,
3932 and 3933.

## The second trap — `--all` is load-bearing on this host

`.201` runs the containerd snapshotter (`docker info` reports
`Storage Driver: overlayfs` with `driver-type: io.containerd.snapshotter.v1`), so
build-cache records share the content store with image layers, and a **default**
builder prune deliberately excludes every record an existing image references.

On 2026-09-18 a prune with the *correct* cap flag but without `--all` still
reclaimed **0B at exit 0**, twice, while `buildx du` reported 655–678 GB
reclaimable. `scripts/disk-gc.sh` therefore passes `--all`, and
`tests/unit/scripts/test_disk_gc_builder_cache_size_cap.py` pins it so a future
"that looks too aggressive" edit is a red test rather than a silent no-op.

## Keeping the two halves in step

The cap appears in two places and they should not drift:

| surface | field | role |
|---|---|---|
| `deploy/disk-gc/daemon-builder-gc.json` | `builder.gc.policy[].maxUsedSpace` | standing ceiling, enforced continuously by dockerd |
| `deploy/disk-gc/keep-list.yaml` | `builder_cache_max_size` | hourly enforcement by `scripts/disk-gc.sh` |

Change one, change the other in the same PR.

## Applying it (operator window only)

Not automated deliberately. The steps, for the operator:

1. Merge the fragment's `builder` key into `/etc/docker/daemon.json`, preserving
   the existing keys — the host's `data-root` is `/data/docker` and must survive.
2. Validate before reloading: `dockerd --validate --config-file /etc/docker/daemon.json`.
3. Reload. `systemctl reload docker` re-reads the config without stopping
   containers where the daemon supports it; a full restart bounces every lane on
   the host and is the reason this is an operator decision rather than a script.
4. Read back: `docker buildx inspect default` must show a catch-all (`All: true`)
   rule whose reserved space is at or below the configured floor. That readback is
   AC1's falsifier — if it still prints a > 100 GiB reserved space, the policy did
   not take.

The cost of a lower ceiling is a colder next build, never a less correct one. The
deploy agent already models this (`COLD_CACHE_MULTIPLIER` in
`deploy_agent/host_conditions.py`) and widens its deploy ceiling accordingly, so a
cold cache is not misread as a hung build.
