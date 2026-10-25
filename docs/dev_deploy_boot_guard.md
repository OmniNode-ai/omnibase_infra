# Dev deploy boot guard (OMN-21279)

The host deploy agent supervises `omninode-runtime`, `runtime-effects`, and
`runtime-worker` on its selected **dev** compose project. Disabled services are
excluded using the instance's existing lane configuration. Prod, stability,
customer-plane runners, and their service units do not use this guard.

Before updating the deploy checkout, the agent captures the Docker-healthy
runtime family. Its compose configuration hashes must match the running
containers. It retains every image by image ID and records the commit revision
and resolved compose settings in
`$DEPLOY_AGENT_STATE_DIR/dev-boot-guard/<project>/last-healthy.json`. The snapshot
contains environment values and is owner-readable only. A previous snapshot
survives agent restarts; without a verified healthy history the agent refuses
runtime replacement. Bring a new lane up through its bootstrap procedure before
enabling guarded deploys.

After building, before runtime `compose up`, each selected runtime image runs a
small Python check with its compose environment and mounts, using `run --rm
--no-deps --pull never` and an overridden entrypoint. If Bifrost rendering is
enabled, its base-contract setting must be nonblank and name an existing file.
Every package named by `ONEX_ACTIVE_RUNTIME_PACKAGES` must be discoverable. This
does not install packages or run the kernel. An invalid candidate fails before
the healthy containers are replaced.

After deployment, **every replica of all three services** must report Docker
healthy. The shared deadline is the largest declared container health budget:
`start_period + retries * (interval + timeout)`. An exhausted boot streak, an
exited/dead container, or Docker unhealthy fails immediately. In particular,
`BOOT FAILED 5/5 ... BIFROST_BASE_CONTRACT_PATH is unset or blank` cannot sit in
the entrypoint's exhausted-retries parking loop for the rest of the window.
The ordinary database and HTTP verification checks still run after this gate.

A compose startup error, failed boot, or failed verification marks the job
failed before recovery. While still holding the dev lane lock, the agent
forward-redeploys the retained image IDs using the saved compose settings,
`--no-build --pull never --no-deps --force-recreate`, and checks boot and health
again. Existing volumes and networks remain external; the recovery never
rebuilds images, pulls a newer tag, restarts dependencies, or changes Git history.
A candidate refused while the original family is still healthy needs no recreate.
The candidate job remains failed even after a successful recovery. A failure of
Docker or the retained revision's recovery remains a failed job and an incident
with `recovery_verified=false`; the agent never reports that as a successful
recovery.

Each failure first writes an owner-readable `incident-<job>.json` beside the
snapshot, naming the failed commit, boot error, retained revision, and recovery
outcome. Its STATUS and exception deliveries are independent and retried by the
agent's existing publish-retry worker, including at startup. Reporting never
precedes recovery. The exception uses the canonical queue's
`dev-runtime-boot-failed` class and job ID as its case key. A queued ledger relay
is not treated as an acknowledged STATUS.

The dev-201 launcher derives the ledger path and bus writer command from its
protected registry root. Set DEPLOY_AGENT_REPORT_PLUGIN_DIR to the installed
reporting plugin directory (CLAUDE_PLUGIN_ROOT is also accepted). Other dev instances must declare their
own `ONEX_LEDGER_PATH` and `ONEX_LEDGER_WRITE_VIA=bus` plus
`ONEX_LEDGER_BUS_APPEND_COMMAND` in their environment. The reporter invokes the
registry's dispatch interpreter and canonical ledger/exception scripts, resolving
that registry through `UNIT_DRIFT_REGISTRY_ROOT`, then `OMNI_HOME`. An unavailable
writer leaves the incident pending. It does not gate recovery on a running
runtime bus worker.

Run the recorded-incident and lane-fence replays with:

```sh
uv run --project scripts/deploy-agent --extra dev pytest scripts/deploy-agent/tests/unit/test_dev_boot_guard_omn21279.py -q
```

The tests isolate Docker and the reporting transports. The PR's runtime boot
smoke remains responsible for proving a real image boots. This change does not
repair the image involved in the 2026-10-25 outage; that belongs to lane
`dev-runtime-down-1710`.

The host adapter loads the canonical handler, models, and protocol from this
checkout. The declarative node contract is node_dev_boot_guard_effect.
