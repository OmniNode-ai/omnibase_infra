# Gateway lane deploy (`omninode-gateway` compose project on `.201`)

This runbook documents the sanctioned deploy path for the `.201` operator-edge
gateway forwarder — the standalone process that bridges the cloud MSK bus to
the `.201` local Redpanda listener for cloud→local inference (OMN-12908 hybrid
gateway). It exists because, before OMN-15521, this lane had **no** repo-
resident deploy path at all: it was stood up on 2026-07-29 by hand-copying
`docker/docker-compose.gateway.yml` and `docker/gateway/beta-gateway-canary.yaml`
into a root-owned directory and running compose there directly. That left the
lane invisible to `deploy-runtime.sh` (whose `-p omnibase-infra` scope never
touches it), running an image with an **empty**
`org.opencontainers.image.revision` label, and with no recorded rollback
target.

Ticket: **OMN-15521**.

> Scope: this is the `.201` `omninode-gateway` lane ONLY (container
> `omninode-gateway-forwarder`, systemd unit `onex-gateway-forwarder`). It is
> unrelated to the `omnibase-infra` runtime lane `deploy-runtime.sh` manages —
> no migrations, no `RUNTIME_SERVICES` restart, no broker readiness preflight.
> See `docs/runbooks/cold-lane-full-bringup.md` for that lane instead.

---

## What's already on the host (read this before you "fix" it)

The `.201` box already has a working, restart-safe supervision layer for this
lane that this runbook does **not** replace:

- systemd unit `/etc/systemd/system/onex-gateway-forwarder.service` (repo copy:
  `docker/gateway/onex-gateway-forwarder.service`) — starts the container via
  `docker compose ... up -d --no-build --wait`, and its `ExecStartPre` **hard
  refuses to start** unless `/etc/omninode/gateway/gateway.env`'s
  `GATEWAY_IMAGE=` line is a real `sha256:<64 hex>` digest. It never builds —
  the image must already exist locally.
- `/etc/omninode/gateway/gateway.env` (root-owned, mode `0444`) — the AWS
  Roles Anywhere / TPM / container-UID variables the compose file requires,
  plus the pinned `GATEWAY_IMAGE` digest.
- `/opt/omninode/gateway/` (root-owned, mode `0444`) — the compose file +
  canary config the systemd unit's `WorkingDirectory` points at.

What was missing is everything **upstream** of that: a repeatable way to
build a new image from merged dev, stamp it with real provenance labels,
update the pinned digest, and sync the host files without hand-editing a
root-owned directory. `scripts/deploy-gateway.sh` is that path.

---

## Procedure

### 0. Pre-flight: sync the canonical clone to the merged-dev tip

Run from the canonical `omnibase_infra` clone on `.201` (same convention as
`deploy-runtime.sh` and `refresh_stability_lane.sh` — this script is **not**
run from a worktree):

```bash
cd /data/omninode/omni_home/omnibase_infra   # or wherever the canonical clone lives on this host
git pull --ff-only
```

### 1. Preview (dry-run, the default)

```bash
./scripts/deploy-gateway.sh
```

Dry-run prints the resolved version/git SHA and the exact compose build
command it would run, and lists every file/registry mutation it would
perform, without touching anything.

### 2. Inspect the exact build command

```bash
./scripts/deploy-gateway.sh --print-compose-cmd
```

This is the fix for the ticket's core finding: the printed command carries
`--build-arg VCS_REF=<sha> --build-arg RUNTIME_VERSION=<version> --build-arg
COMPOSE_PROJECT=omninode-gateway --build-arg RUNTIME_SOURCE_HASH=<sha>
--build-arg PROMOTION_CLASS=clean-main --build-arg NON_MAIN_LINEAGE=false` —
the same OCI provenance build-args `deploy-runtime.sh`'s `build_images()`
passes for every `omnibase-infra` runtime container. The compose file's own
declared `build.args` block only ever carried `BUILD_SOURCE`/
`EXPECTED_BUILD_SOURCE` (mirroring `docker-compose.infra.yml`'s split between
compose-declared and CLI-supplied build-args) — that split is why the
hand-built image had `rev=(empty)`.

### 3. Execute the deploy

```bash
./scripts/deploy-gateway.sh --execute
```

In order, this:

1. Resolves `repo_root` / `version` (pyproject.toml) / `git_sha` (HEAD).
2. Reads the **existing** `GATEWAY_IMAGE` digest from
   `/etc/omninode/gateway/gateway.env` and records it as the rollback target.
3. Sources that same env file so the AWS/TPM/UID variables the compose file
   requires resolve for the build.
4. Builds `docker-gateway-forwarder:build` with the provenance build-args from
   step 2 above.
5. Resolves the built image's digest (`docker image inspect --format='{{.Id}}'`).
6. Syncs `docker/docker-compose.gateway.yml` and
   `docker/gateway/beta-gateway-canary.yaml` from **this checkout** into
   `/opt/omninode/gateway/` (`sudo install -m 0444 -o root -g root`, matching
   the existing file posture) — replacing the 2026-07-29 hand-copy as the
   source of truth. Every deploy re-syncs, so the host copy can never drift
   from a merged commit again.
7. Rewrites `/etc/omninode/gateway/gateway.env`'s `GATEWAY_IMAGE=` line to the
   new digest, leaving every other key untouched.
8. Writes `~/.omnibase/gateway/registry.json` recording `active_digest`,
   `previous_digest` (the rollback target), `git_sha`, `deployed_at`, and a
   ready-to-run `rollback_command` — the same convention
   `~/.omnibase/infra/registry.json` uses for the `omnibase-infra` lane.
9. `sudo systemctl reload onex-gateway-forwarder` — the unit's existing
   `ExecReload` force-recreates the container on the new digest.
10. Verifies: labels are non-empty, and the two OMN-12912 files
    (`service_gateway_delivery.py`, `store_sqlite.py`) are present inside the
    running container.

Pass `--skip-reload` to build + sync + pin the new digest without recreating
the running container yet (the old container keeps running on the old
digest until a manual `sudo systemctl reload onex-gateway-forwarder`).

### 4. Verify

```bash
docker inspect omninode-gateway-forwarder \
  --format='rev={{index .Config.Labels "org.opencontainers.image.revision"}} src={{index .Config.Labels "com.omninode.build_source"}}'
# -> rev=<12-char sha> src=release   (never empty)

docker exec omninode-gateway-forwarder \
  ls /app/src/omnibase_infra/nodes/node_bus_forwarder_effect/services/
# -> must include service_gateway_delivery.py

docker exec omninode-gateway-forwarder \
  ls /app/src/omnibase_infra/idempotency/
# -> must include store_sqlite.py

diff /opt/omninode/gateway/docker-compose.gateway.yml docker/docker-compose.gateway.yml
# -> empty

cat ~/.omnibase/gateway/registry.json | jq .
```

---

## Rollback

`~/.omnibase/gateway/registry.json`'s `previous_digest` names the last-known-
good digest. To roll back to it:

```bash
sudo sed -i "s|^GATEWAY_IMAGE=.*|GATEWAY_IMAGE=$(jq -r .previous_digest ~/.omnibase/gateway/registry.json)|" \
  /etc/omninode/gateway/gateway.env
sudo systemctl reload onex-gateway-forwarder
```

(`registry.json`'s own `rollback_command` field carries this exact command
pre-filled with the digest recorded at deploy time.) This mirrors the
`omnibase-infra` lane's own manual rollback-via-`registry.json` pattern —
`deploy-runtime.sh` has no automated `--rollback` flag either; a prior
digest is always restored by hand from the registry.

---

## What OMN-15521 does NOT cover

The OMN-12912 restart/redelivery proof (source-offset-ack / dedupe receipt,
run against the freshly deployed forwarder after a restart) is **out of
scope for this ticket** — it lands on OMN-12912, not here, per that ticket's
own falsifiable acceptance criteria.

## Related runbooks

- `docs/runbooks/cold-lane-full-bringup.md` — the `omnibase-infra` runtime
  lane's own cold bring-up (a different compose project, different scope).
- `docker/gateway/onex-gateway-forwarder.service` — the systemd unit this
  script's `--execute` reloads.
