# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One warm interpreter for the whole runtime boot preflight (OMN-17372).

``docker/entrypoint-runtime.sh`` used to reach four (up to five) separate cold
Python interpreters before the kernel ever started, each paying a full cold
``import omnibase_infra``:

* ``python -m omnibase_infra.runtime.util_schema_fingerprint`` -- 141.3 s
* ``python -m omnibase_infra.runtime.render_bifrost_delegation_contract`` -- 51.9 s
* ``python -m omnibase_infra.runtime.render_secret_resolver_config`` -- 50.2 s
* ``exec onex-runtime`` (the kernel itself) -- 48.6 s
* total before any wiring work at all: **292.0 s**

Measured live on 2026-09-06 from the onex-dev ``omninode-runtime`` container on
the dev-system cluster, container start 21:34:27Z: 551 s of an 1124 s boot
elapsed before the FIRST subscription was attempted, 292 s of it in those four
interpreter starts. The same phase on the .201 dev lane is
~8x faster on the same code, because the onex-dev container runs at
``requests.cpu 100m`` on a node whose CPU limits are 547 % overcommitted — the
signature of ~0.05 effective cores paying four cold imports.

This module is that boot preflight, in ONE process. Every step below is the
same code the shell used to launch, called as a function, in the order the
shell called it, with the same guards, the same retry policy, the same exit
codes and the same stdout/stderr text. When the preflight is done it hands off
to the CMD: for the packaged kernel entry point (``onex-runtime``) it calls
:func:`omnibase_infra.runtime.kernel.main` **in this same process**, so the
kernel starts with ``omnibase_infra`` already imported; for any other CMD it
``os.execvp``s, preserving the previous ``exec "$@"`` semantics exactly.

What deliberately stays in the shell: the root-only volume-ownership bootstrap
and the ``gosu`` privilege drop (they run before the unprivileged user exists,
and re-exec the script), and the deployment-identity banner (pure ``echo``, no
interpreter).

Bounded restart (OMN-18550): the container's restart policy relaunches this
process after every failing exit, so on its own it loops forever on a boot that
can never succeed. :func:`main` therefore counts consecutive failing exits in the
durable boot-failure ledger, publishes a boot-failed event (cause and count) for
each, and once the declared bound is reached records the exhausted event and
parks the container instead of booting again: the restart policy has nothing left
to relaunch, and the container reports unhealthy with the cause in its log.

Related Tickets:
    - OMN-18550: unbounded, unreported restart loop on a malformed contract.
    - OMN-17372: runtime boot latency — four cold interpreters before the kernel.
    - OMN-13666: required (own DB) vs best-effort (non-owned DB) stamp policy.
    - OMN-15807: the Bifrost renderer always rebuilds from the packaged base.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path

from omnibase_infra.handlers.handler_boot_failure_ledger import (
    HandlerBootFailureLedger,
)
from omnibase_infra.models.health.model_boot_attempt_outcome import (
    ModelBootAttemptOutcome,
)
from omnibase_infra.models.health.model_boot_identity import ModelBootIdentity
from omnibase_infra.models.health.model_runtime_boot_failed_event import (
    ModelRuntimeBootFailedEvent,
)

#: Retry policy for the schema-fingerprint stamp, ported verbatim from the
#: shell loop it replaces (5 attempts, 1 s apart, exit 2 is not retried).
MAX_STAMP_ATTEMPTS: int = 5
STAMP_RETRY_SLEEP_SECONDS: float = 1.0

#: Exit code the stamp returns for "schema mismatch" — never retried.
STAMP_RC_MISMATCH: int = 2

#: CMD values whose handoff is an in-process call rather than an ``execvp``.
#: ``onex-runtime`` is the packaged console script declared in ``pyproject.toml``
#: (``omnibase_infra.runtime.kernel:main``) and the image's ``CMD``.
IN_PROCESS_KERNEL_COMMANDS: frozenset[str] = frozenset({"onex-runtime"})

#: OMN-18550: declared bound on consecutive failing boots before the container
#: stops retrying, and the uptime after which a later failure is a new incident.
#: Both are overridable per service through the environment variables below.
ENV_BOOT_FAILURE_MAX: str = "ONEX_BOOT_FAILURE_MAX_CONSECUTIVE"
ENV_BOOT_FAILURE_HEALTHY_AFTER: str = "ONEX_BOOT_FAILURE_HEALTHY_AFTER_SECONDS"
DEFAULT_MAX_CONSECUTIVE_BOOT_FAILURES: int = 5
DEFAULT_HEALTHY_AFTER_SECONDS: float = 600.0

#: The logger whose ERROR records carry the kernel's own account of why it died.
#: The capture handler hangs on this logger and NOT on the root logger: the
#: kernel's ``logging.basicConfig`` is a no-op once the root has a handler.
_KERNEL_LOGGER_NAME: str = "omnibase_infra.runtime.service_kernel"
_MAX_CAUSE_CHARS: int = 500
_PARK_SLEEP_SECONDS: float = 3600.0
_PUBLISH_TIMEOUT_SECONDS: float = 8.0


def _echo(message: str) -> None:
    """Write an entrypoint line to stdout, unbuffered, like the shell's echo."""
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def _echo_err(message: str) -> None:
    """Write an entrypoint line to stderr, unbuffered, like the shell's echo >&2."""
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def _safe_dsn(db_url: str) -> str:
    """Strip scheme and userinfo from a DSN, leaving ``host:port/db``.

    Mirrors the shell's ``sed 's|^[^/]*//[^@]*@||'`` so the log line is
    byte-identical and no credential is ever printed.
    """
    scheme_split = db_url.split("//", 1)
    if len(scheme_split) == 1:
        return db_url
    remainder = scheme_split[1]
    if "@" not in remainder:
        # sed's `s|^[^/]*//[^@]*@||` does not substitute at all without an `@`,
        # so the DSN is echoed unchanged. Match that exactly.
        return db_url
    return remainder.split("@", 1)[1]


def stamp_fingerprint(
    *,
    manifest_name: str,
    db_url: str,
    required: bool,
    sleep: Callable[[float], None] | None = None,
) -> int:
    """Stamp one database's schema fingerprint, with the shell's retry policy.

    Args:
        manifest_name: ``omnibase_infra`` or ``omniintelligence``.
        db_url: DSN for that database.
        required: ``True`` for the runtime's OWN database, whose stamp drives
            the kernel's startup fingerprint assertion — a failure aborts boot.
            ``False`` for a non-owned database, where the runtime DB user
            legitimately lacks write permission and a failure only warns
            (OMN-13666).
        sleep: Injected sleep so tests drive the retry loop without wall
            clock. Resolved lazily so a patched ``time.sleep`` is honoured.

    Returns:
        ``0`` when boot may proceed, ``1`` when a REQUIRED stamp failed and boot
        must abort.
    """
    from omnibase_infra.runtime.util_schema_fingerprint import (
        stamp_manifest_fingerprint,
    )

    sleeper: Callable[[float], None] = time.sleep if sleep is None else sleep
    required_label = "required" if required else "optional"
    _echo(
        f"[entrypoint] Stamping schema fingerprint for {manifest_name} "
        f"(db: {_safe_dsn(db_url)}, {required_label})..."
    )

    stamp_ok = False
    attempt = 1
    while attempt <= MAX_STAMP_ATTEMPTS:
        rc = stamp_manifest_fingerprint(manifest_name=manifest_name, db_url=db_url)
        if rc == 0:
            stamp_ok = True
            _echo(f"[entrypoint] Schema fingerprint stamped for {manifest_name}.")
            break
        if rc == STAMP_RC_MISMATCH:
            _echo(
                f"[entrypoint] WARNING: {manifest_name} fingerprint mismatch "
                "(exit 2) -- not retrying"
            )
            break
        _echo(
            f"[entrypoint] {manifest_name} stamp attempt "
            f"{attempt}/{MAX_STAMP_ATTEMPTS} failed (exit {rc})"
        )
        attempt += 1
        if attempt <= MAX_STAMP_ATTEMPTS:
            sleeper(STAMP_RETRY_SLEEP_SECONDS)

    if not stamp_ok:
        if required:
            _echo_err(
                f"[entrypoint] ERROR: {manifest_name} (PRIMARY/owned DB) "
                "fingerprint stamp failed -- aborting boot"
            )
            return 1
        _echo(
            f"[entrypoint] WARNING: {manifest_name} (secondary/non-owned DB) "
            "fingerprint stamp did not succeed -- continuing best-effort"
        )
    return 0


def stamp_all_fingerprints(env: Mapping[str, str]) -> int:
    """Stamp both declared databases. Returns a non-zero boot-abort code."""
    infra_db_url = env.get("OMNIBASE_INFRA_DB_URL", "")
    if infra_db_url:
        rc = stamp_fingerprint(
            manifest_name="omnibase_infra", db_url=infra_db_url, required=True
        )
        if rc != 0:
            return rc
    else:
        _echo(
            "[entrypoint] OMNIBASE_INFRA_DB_URL not set -- skipping fingerprint stamp"
        )

    intel_db_url = env.get("OMNIINTELLIGENCE_DB_URL", "")
    if intel_db_url:
        rc = stamp_fingerprint(
            manifest_name="omniintelligence", db_url=intel_db_url, required=False
        )
        if rc != 0:
            return rc
    else:
        _echo(
            "[entrypoint] OMNIINTELLIGENCE_DB_URL not set -- skipping "
            "omniintelligence fingerprint stamp"
        )
    return 0


def render_bifrost_contract(env: Mapping[str, str]) -> int:
    """Render the Bifrost delegation contract when its target path is declared.

    OMN-15807: the renderer always rebuilds from the packaged base contract and
    the mounted typed lane overlay; no endpoint or model environment binding is
    accepted, so a stale volume cannot route.
    """
    if not env.get("BIFROST_CONTRACT_PATH"):
        return 0
    from omnibase_infra.runtime.render_bifrost_delegation_contract import (
        main as render_bifrost_main,
    )

    _echo("[entrypoint] Rendering Bifrost delegation contract from typed overlay...")
    return render_bifrost_main()


def render_resolver_config(env: Mapping[str, str]) -> int:
    """Render the secret-resolver config when its target path is declared."""
    if not env.get("ONEX_SECRET_RESOLVER_CONFIG_PATH"):
        return 0
    from omnibase_infra.runtime.render_secret_resolver_config import (
        main as render_resolver_main,
    )

    _echo("[entrypoint] Rendering secret resolver config...")
    return render_resolver_main()


def run_preflight(env: MutableMapping[str, str]) -> int:
    """Run every preflight step in the shell's original order.

    Returns ``0`` when the kernel may start, or the boot-abort exit code.
    """
    rc = stamp_all_fingerprints(env)
    if rc != 0:
        return rc
    rc = render_bifrost_contract(env)
    if rc != 0:
        return rc
    rc = render_resolver_config(env)
    if rc != 0:
        return rc
    return 0


def handoff(argv: list[str]) -> int:
    """Start the CMD. Never returns for the ``execvp`` path.

    The packaged kernel entry point is started IN THIS PROCESS so it inherits
    the already-warm ``omnibase_infra`` import — the whole point of OMN-17372.
    Any other CMD keeps the previous ``exec "$@"`` semantics.
    """
    if not argv:
        _echo_err("[entrypoint] ERROR: no CMD to start -- aborting boot")
        return 1

    _echo("[entrypoint] Starting runtime kernel...")

    if Path(argv[0]).name in IN_PROCESS_KERNEL_COMMANDS and len(argv) == 1:
        from omnibase_infra.runtime.kernel import main as kernel_main

        kernel_main()
        return 0

    os.execvp(argv[0], argv)  # noqa: S606 — this IS the exec the shell used to do
    raise AssertionError("unreachable: os.execvp does not return")  # pragma: no cover


class KernelErrorCapture(logging.Handler):
    """Remembers the kernel's last ERROR record as a sanitized one-line cause."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.cause: str = ""

    def emit(self, record: logging.LogRecord) -> None:
        from omnibase_infra.utils.util_error_sanitization import (
            sanitize_error_message,
        )

        cause = record.getMessage()
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, Exception):
            cause = f"{cause} | {sanitize_error_message(exc)}"
        self.cause = cause[:_MAX_CAUSE_CHARS]


def _exit_code(exc: SystemExit) -> int:
    """The process exit status a ``SystemExit`` stands for."""
    if exc.code is None:
        return 0
    return exc.code if isinstance(exc.code, int) else 1


def _env_number(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        _echo_err(f"[entrypoint] ignoring non-numeric {name}={raw!r}")
        return default
    return value if value > 0 else default


def _publish_boot_failed(
    event: ModelRuntimeBootFailedEvent, env: Mapping[str, str]
) -> None:
    """Publish the event on the bus. Best effort: the ledger already holds it.

    The bus may be the very thing the boot failed on, so a publish failure is
    reported and swallowed rather than allowed to mask the exit code.
    """
    bootstrap = env.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not bootstrap:
        _echo_err(
            "[entrypoint] boot-failed event not published: no KAFKA_BOOTSTRAP_SERVERS"
        )
        return

    from aiokafka.errors import KafkaError

    from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
        EnumOmnibaseInfraTopic,
    )

    async def _send() -> None:
        from aiokafka import AIOKafkaProducer

        from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
        from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
            ModelKafkaEventBusConfig,
        )
        from omnibase_infra.runtime.gateway_forwarder import ensure_topics_exist

        topic = EnumOmnibaseInfraTopic.EVT_RUNTIME_BOOT_FAILED_V1.value
        bus_config = ModelKafkaEventBusConfig.default()
        # A lane broker with auto-create off has never seen this topic if no
        # healthy boot of this build has provisioned it yet; creation is
        # idempotent and best effort, like the publish.
        await ensure_topics_exist(bus_config, (topic,))
        # The bus is authenticated on the lab lanes: the producer needs the same
        # SASL/TLS settings the kernel's own producers get from the environment.
        producer = AIOKafkaProducer(
            bootstrap_servers=bus_config.bootstrap_servers,
            request_timeout_ms=5000,
            **build_aiokafka_auth_kwargs(bus_config),
        )
        await producer.start()
        try:
            await producer.send_and_wait(
                topic,
                value=event.model_dump_json().encode("utf-8"),
                key=event.service.encode("utf-8"),
            )
        finally:
            await producer.stop()

    try:
        asyncio.run(asyncio.wait_for(_send(), timeout=_PUBLISH_TIMEOUT_SECONDS))
    except (KafkaError, OSError, TimeoutError) as exc:
        _echo_err(
            f"[entrypoint] boot-failed event not published to the bus: {type(exc).__name__}"
        )


def _park(event: ModelRuntimeBootFailedEvent) -> int:
    """Stop retrying: report why, then idle so the restart policy has nothing to relaunch.

    Exiting would hand the container straight back to its restart policy, which
    relaunches on any exit under ``unless-stopped``. Idling leaves no health
    endpoint, so the container reads unhealthy. ``SIGTERM`` ends it as usual; a
    new image (new source hash) or removing the ledger lets the next start boot.
    """
    _echo_err(
        "[entrypoint] RETRIES EXHAUSTED: "
        f"{event.consecutive_failures} consecutive failing boots of "
        f"{event.runtime_source_hash} (declared bound {event.max_consecutive_failures}); "
        f"last exit {event.exit_code}; cause: {event.cause}"
    )
    _echo_err(
        "[entrypoint] not booting again: recreate the container (a redeploy does), "
        "or remove boot_failures/ from the state directory, to retry"
    )
    while True:
        time.sleep(_PARK_SLEEP_SECONDS)


def _record_failure(
    ledger: HandlerBootFailureLedger,
    identity: ModelBootIdentity,
    env: Mapping[str, str],
    *,
    exit_code: int,
    cause: str,
    started: float,
    healthy_after: float,
) -> ModelRuntimeBootFailedEvent:
    """Count one failing boot, say so on stderr, and publish its event."""
    event = ledger.handle(
        ModelBootAttemptOutcome(
            identity=identity,
            exit_code=exit_code,
            cause=cause or f"runtime exited {exit_code} without a logged error",
            uptime_seconds=time.monotonic() - started,
            healthy_after_seconds=healthy_after,
            observed_at=datetime.now(UTC),
        )
    )
    _echo_err(
        f"[entrypoint] BOOT FAILED {event.consecutive_failures}/"
        f"{event.max_consecutive_failures}: exit {event.exit_code}: {event.cause}"
    )
    _publish_boot_failed(event, env)
    return event


def _supervised_start(cmd: list[str], env: MutableMapping[str, str]) -> int:
    """Run preflight and the CMD once; count and report a failing exit (OMN-18550)."""
    state_dir = (
        env.get("ONEX_STATE_DIR", "").strip() or env.get("ONEX_STATE_ROOT", "").strip()
    )
    if not state_dir:
        rc = run_preflight(env)
        return rc if rc != 0 else handoff(cmd)

    ledger = HandlerBootFailureLedger()
    identity = ModelBootIdentity(
        state_dir=state_dir,
        service=env.get("ONEX_GROUP_ID", "").strip() or "onex-runtime",
        container_ref=env.get("HOSTNAME", "").strip() or socket.gethostname(),
        runtime_source_hash=env.get("RUNTIME_SOURCE_HASH", "").strip() or "unknown",
        max_consecutive_failures=int(
            _env_number(
                env, ENV_BOOT_FAILURE_MAX, DEFAULT_MAX_CONSECUTIVE_BOOT_FAILURES
            )
        ),
    )
    healthy_after = _env_number(
        env, ENV_BOOT_FAILURE_HEALTHY_AFTER, DEFAULT_HEALTHY_AFTER_SECONDS
    )

    exhausted = ledger.exhausted_event(identity)
    if exhausted is not None:
        return _park(exhausted)

    capture = KernelErrorCapture()
    kernel_logger = logging.getLogger(_KERNEL_LOGGER_NAME)
    kernel_logger.addHandler(capture)
    started = time.monotonic()
    cause = ""
    try:
        rc = run_preflight(env)
        if rc != 0:
            cause = f"boot preflight aborted with exit {rc}"
        else:
            rc = handoff(cmd)
    except SystemExit as exc:
        rc = _exit_code(exc)
    except Exception as exc:
        # An unhandled crash is a failing boot too: count it, then let it
        # propagate so the traceback and exit status are unchanged. The next
        # start reads the ledger and parks if this was the last allowed boot.
        from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

        _record_failure(
            ledger,
            identity,
            env,
            exit_code=1,
            cause=sanitize_error_message(exc),
            started=started,
            healthy_after=healthy_after,
        )
        raise
    finally:
        kernel_logger.removeHandler(capture)

    if rc == 0:
        ledger.record_clean_exit(identity)
        return 0

    event = _record_failure(
        ledger,
        identity,
        env,
        exit_code=rc,
        cause=capture.cause or cause,
        started=started,
        healthy_after=healthy_after,
    )
    if event.retries_exhausted:
        return _park(event)
    return rc


def main(argv: list[str] | None = None) -> int:
    """Entry point: preflight in one warm interpreter, then start the CMD."""
    cmd = list(sys.argv[1:] if argv is None else argv)
    return _supervised_start(cmd, os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
