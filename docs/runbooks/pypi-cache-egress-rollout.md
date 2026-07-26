# PyPI pull-through cache (egress) rollout — OMN-14027 C1

**Status:** SHOVEL-READY / execution soak-gated. Do **not** run these steps until
the OMN-14027 execution gate trips (the ticket is DESIGN-ONLY until then; the same
Track-2 discipline as OMN-13954 and the §3.19 mirror go/no-go). This runbook is the
`APPLY.md` companion to the staged cache; keeping it in-repo means the rollout is
reviewable before it runs.

## What this is

64 self-hosted runners NAT through one home uplink. Under concurrent `merge_group`
full-suite load, each runner independently cold-downloads the same PyPI wheels,
saturating egress and tripping `uv sync` download timeouts (the OMN-14017 failure
class). A devpi pull-through cache fetches each wheel from PyPI **once**, then serves
all 64 runners from the LAN — removing the redundant bytes from the shared uplink.
This is the independent-throughput lever; it complements the C3 stampede cap that is
already active fleet-wide (see below).

Sibling of the §3.19 Docker Hub pull-through mirror (images vs wheels; same host,
same non-disruptive rollback discipline). Land together or sequence — not mutually
exclusive.

## Artifacts in this repo (already landed by the design/canary PR)

| Artifact | Role | State |
|---|---|---|
| `docker/pypi-cache/Dockerfile` + `entrypoint.sh` | devpi-server image (build-at-rollout) | inert |
| `docker/docker-compose.pypi-cache.yml` | standalone cache service (NOT in the runner fleet) | inert |
| `config/runner_fleet.yaml` → `pypi_cache:` | endpoint source-of-truth | `active: false` |
| `docker/docker-compose.runners.yml` → commented `UV_DEFAULT_INDEX` block | fleet wiring | inert (commented) |
| `docker/docker-compose.runners.yml` → `UV_CONCURRENT_*` / `UV_HTTP_TIMEOUT` | C3 stampede cap | **active** |

## C3 (already active — not gated)

The fleet-wide uv concurrency cap ships active in the canary PR because it cannot
regress any path: the hardened `setup-python-uv` composite already pins
`UV_CONCURRENT_DOWNLOADS=${...:-1}` = 1, so a fleet default of 1 leaves that path
unchanged while capping the raw-`uv` paths (the OMN-14193 workflows) down from uv's
built-in default. **Do not raise the fleet cap to 2 until C1 below is active and the
canary metrics below are green** — 2 doubles the composite path's download fan under
a still-saturated uplink.

## Rollout (gated — run only after the gate trips)

Run on the runner host (`.201` / `omninode-pc.tail75df5e.ts.net`). This is a NEW
service; it does not mutate any prod daemon or the runner containers.

### 1. Freeze the pins

- Pin the `python:3.12-slim` base image to a `@sha256:` digest in
  `docker/pypi-cache/Dockerfile` (match the runner-image reproducibility discipline).
- Freeze `DEVPI_SERVER_SPEC` / `DEVPI_WEB_SPEC` to exact versions.

### 2. Stand up devpi

```bash
docker compose -f docker/docker-compose.pypi-cache.yml up -d --build
# wait for healthy:
docker inspect --format '{{.State.Health.Status}}' omninode-pypi-cache
# prove the pull-through index answers:
curl -fsS http://localhost:3141/root/pypi/+simple/ >/dev/null && echo "cache index OK"
```

### 3. Wire ONE canary runner (leave the other 63 on direct egress)

Recreate a single runner container with the cache env set on that container only —
do **not** uncomment the fleet-wide block yet.

```bash
UV_DEFAULT_INDEX="http://omninode-pc.tail75df5e.ts.net:3141/root/pypi/+simple/" \
UV_INDEX_STRATEGY="unsafe-best-match" \
PIP_INDEX_URL="http://omninode-pc.tail75df5e.ts.net:3141/root/pypi/+simple/" \
PIP_EXTRA_INDEX_URL="https://pypi.org/simple/" \
  docker compose -f docker/docker-compose.runners.yml up -d --no-deps \
    --force-recreate omninode-runner-1
```

PyPI stays configured as the fallback (`PIP_EXTRA_INDEX_URL` +
`UV_INDEX_STRATEGY=unsafe-best-match`) so a cache miss/outage degrades, not
fails-closed.

### 4. Measure on the canary (short soak)

- Wheel-cache hit rate on the canary → target ≥ ~90% steady-state
  (`config/runner_fleet.yaml` `pypi_cache.target_hit_rate`).
- p95 `uv sync` wall time vs a direct-egress runner under the same load.
- Zero `UV_HTTP_TIMEOUT` / `virtualenv download` failures on `merge_group`
  full-suite jobs under 64/64-busy load.
- No throughput regression vs direct egress.

### 5. Fleet rollout (only if the canary is green)

- Set `pypi_cache.active: true` in `config/runner_fleet.yaml`.
- Uncomment the `UV_DEFAULT_INDEX` / `UV_INDEX_STRATEGY` / `PIP_INDEX_URL` /
  `PIP_EXTRA_INDEX_URL` block in `docker/docker-compose.runners.yml`.
- Optionally raise `UV_CONCURRENT_*` from 1 to 2 now that egress pressure is
  removed (tune from the canary measurement).
- Roll the env to the full fleet via runner-image rebuild + rolling recreate
  (idle-only, same care as fleet-heal). C2 (Actions/uv-binary acceleration) folds
  in with the same rebuild — see the ticket.

## Rollback (non-disruptive)

Fallback is to drop the cache wiring; the cache is additive, so removing it returns
the fleet to direct egress.

1. Re-comment the `UV_DEFAULT_INDEX` block in `docker-compose.runners.yml` (and set
   `pypi_cache.active: false`), rebuild the runner image, rolling-recreate.
2. Tear down the cache service (its own project — not a runtime lane, so the
   no-bare-compose-teardown gate does not apply):

   ```bash
   docker compose -f docker/docker-compose.pypi-cache.yml down
   ```

The named volume `omninode-pypi-cache-data` persists the warmed corpus across a
recreate; add `-v` only to discard the cache entirely.

## References

- OMN-14027 (this work), parent OMN-13932
- OMN-14017 — Layer-A per-workflow `uv` timeout band-aid this makes non-load-bearing
- OMN-14193 — 18 workflows bypassing the hardened composite (the raw-`uv` class C3 caps)
- OMN-14192 — sibling Docker Hub pull-through mirror (§3.19)
- Plan: `omni_home/docs/plans/2026-07-05-runner-fleet-permanent-fix-plan.md`
