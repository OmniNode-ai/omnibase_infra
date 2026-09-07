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

Measured live on 2026-09-06 from the onex-dev ``omninode-runtime`` container
(dev-system cluster ``i-06169517a92b45f86``), container start 21:34:27Z: 551 s
of an 1124 s boot elapsed before the FIRST subscription was attempted, 292 s of
it in those four interpreter starts. The same phase on the .201 dev lane is
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

Related Tickets:
    - OMN-17372: runtime boot latency — four cold interpreters before the kernel.
    - OMN-13666: required (own DB) vs best-effort (non-owned DB) stamp policy.
    - OMN-15807: the Bifrost renderer always rebuilds from the packaged base.
    - OMN-15628: DELEGATION_ROUTING_TIERS_PATH self-heal.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path

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


def self_heal_delegation_tiers_path(env: MutableMapping[str, str]) -> None:
    """Re-derive ``DELEGATION_ROUTING_TIERS_PATH`` when the pinned path is gone.

    OMN-15628: the k8s manifests pin the path as a literal string embedding the
    venv's Python minor version. A base-image Python bump silently invalidates
    it with no signal until the routing reducer fails closed at first use.
    Re-derive from the installed ``omnimarket`` package's OWN location, which
    always matches whatever Python is actually running in this image.

    Best-effort correction, never a silent fallback: if re-derivation does not
    land on a real file, the original (possibly-stale) pinned value is left
    untouched and the routing reducer still fails closed attributably.
    """
    pinned = env.get("DELEGATION_ROUTING_TIERS_PATH", "")
    if not pinned or Path(pinned).is_file():
        return

    _echo(
        f"[entrypoint] WARNING: DELEGATION_ROUTING_TIERS_PATH={pinned} does not "
        "exist -- attempting to re-derive from the installed omnimarket package"
    )
    resolved = ""
    try:
        import omnimarket

        resolved = str(
            Path(omnimarket.__file__).resolve().parent
            / "configs"
            / "routing_tiers.yaml"
        )
    except Exception:  # noqa: BLE001 — boundary: any import failure is "unresolved"
        resolved = ""

    if resolved and Path(resolved).is_file():
        _echo(f"[entrypoint] Re-derived DELEGATION_ROUTING_TIERS_PATH={resolved}")
        env["DELEGATION_ROUTING_TIERS_PATH"] = resolved
        return

    _echo(
        "[entrypoint] WARNING: could not re-derive a valid routing_tiers.yaml "
        "path -- leaving DELEGATION_ROUTING_TIERS_PATH as pinned; the routing "
        "reducer fails closed with an attributable error if it truly does not "
        "exist (OMN-15628)"
    )


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
    self_heal_delegation_tiers_path(env)
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


def main(argv: list[str] | None = None) -> int:
    """Entry point: preflight in one warm interpreter, then start the CMD."""
    cmd = list(sys.argv[1:] if argv is None else argv)
    rc = run_preflight(os.environ)
    if rc != 0:
        return rc
    return handoff(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
