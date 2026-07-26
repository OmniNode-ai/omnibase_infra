# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every contract-declared handler must expose a dispatch entrypoint (OMN-14135).

THE DEFECT CLASS THIS GATE CLOSES
---------------------------------
A handler can be contract-declared, auto-wired, instantiated, boot clean, and GREEN
in CI — and still have NO dispatch entrypoint at all.

``_make_dispatch_callback`` (``omnibase_infra.runtime.auto_wiring.handler_wiring``)
resolves the runtime entrypoint by looking for ``handle_async``, then ``handle``.
Finding neither, it binds ``_missing_handle``, which raises ``ModelOnexError`` on the
FIRST real dispatch. The dispatcher still REGISTERS and the payload still ROUTES to
it — so every upstream signal stays green:

  * contract validation passes (the contract is well-formed);
  * auto-wiring succeeds (registration does not require an entrypoint);
  * the node boots;
  * dispatch-selection oracles show the route as present.

"Registered" and "routable" are NOT "executable". ``HandlerA2ATask``
(node_remote_agent_invoke_effect) shipped this way: contract-declared, wired,
ingress-valid, CI-green, exposing only ``submit()``/``watch()``. The A2A remote-agent
branch passed ingress and then died on first dispatch. A platform census found it was
not alone: 40 of 153 distinct contract-declared handlers had no entrypoint.

WHY EXISTING GATES WERE BLIND
-----------------------------
Contract tests prove INGRESS and never prove DISPATCH. The OMN-14488 route-coverage
gate proves a payload ROUTES to a registered dispatcher — but a dispatcher bound to
``_missing_handle`` registers and routes just fine, and that gate's fake handler HAS a
``handle``, so it cannot see this. This validator tests the DISPATCH BIND itself, using
the exact predicate auto-wiring uses.

RATCHET SEMANTICS (shrink-only; see config/validation/handler_dispatch_entrypoint_baseline.yaml)
-----------------------------------------------------------------------------------------------
  * a handler NOT in the baseline with no entrypoint  -> FAIL (no new instances, ever);
  * a handler IN the baseline that GAINS an entrypoint -> FAIL until it is removed from
    the baseline (a fixed handler still listed is STALE, so the list cannot rot).

The baseline can only shrink. End state is an empty list, at which point
``_missing_handle`` becomes a wiring-time hard failure instead of a per-message one.

Usage (pre-commit / CI):
    uv run python -m omnibase_infra.validators.handler_dispatch_entrypoint src/omnibase_infra
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")
DEFAULT_BASELINE = Path("config/validation/handler_dispatch_entrypoint_baseline.yaml")

# A scan that discovers far fewer handlers than the platform actually has is a broken
# scan, not a clean repo. A gate over an empty/collapsed set is vacuously green, so the
# validator fails closed below this floor rather than reporting success.
MIN_EXPECTED_DECLARED_HANDLERS = 100


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class DeclaredDispatchTarget:
    """A dispatch target (handler class) named in a contract's ``handler_routing.handlers[]``."""

    contract: str
    handler: str
    module: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.contract, self.handler)


def has_dispatch_entrypoint(cls: type) -> bool:
    """The EXACT predicate ``_make_dispatch_callback`` uses to bind an entrypoint.

    Kept deliberately identical to the auto-wiring resolution order (``handle_async``
    then ``handle``). If this drifts from ``handler_wiring``, the gate stops describing
    the runtime and starts describing itself.
    """
    return callable(getattr(cls, "handle_async", None)) or callable(
        getattr(cls, "handle", None)
    )


def declared_handlers(scan_root: Path) -> list[DeclaredDispatchTarget]:
    """Every handler declared in every ``contract.yaml`` under ``scan_root``."""
    found: list[DeclaredDispatchTarget] = []
    for contract_path in sorted(scan_root.rglob("contract.yaml")):
        data = yaml.safe_load(contract_path.read_text())
        if not isinstance(data, dict):
            continue
        contract = str(data.get("name") or contract_path.parent.name)
        routing = data.get("handler_routing") or {}
        for entry in routing.get("handlers") or []:
            handler = (entry or {}).get("handler") or {}
            name, module = handler.get("name"), handler.get("module")
            if name and module:
                found.append(DeclaredDispatchTarget(contract, str(name), str(module)))
    return found


def entrypointless(handlers: Sequence[DeclaredDispatchTarget]) -> set[tuple[str, str]]:
    """Declared handlers that expose NEITHER ``handle`` nor ``handle_async``.

    An unimportable handler is skipped: import health is a separate gate, and failing
    here would misattribute an import error to a missing entrypoint.
    """
    missing: set[tuple[str, str]] = set()
    for declared in handlers:
        try:
            cls = getattr(importlib.import_module(declared.module), declared.handler)
        except Exception:  # noqa: BLE001 — import health is a separate gate
            continue
        if not has_dispatch_entrypoint(cls):
            missing.add(declared.key)
    return missing


def load_baseline(baseline_path: Path) -> set[tuple[str, str]]:
    """Load the frozen shrink-only burn-down baseline."""
    if not baseline_path.is_file():
        return set()
    data = yaml.safe_load(baseline_path.read_text()) or {}
    return {
        (str(row["contract"]), str(row["handler"]))
        for row in (data.get("known_entrypointless") or [])
    }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail any contract-declared handler that auto-wiring would bind to "
            "_missing_handle (no handle/handle_async dispatch entrypoint)."
        )
    )
    parser.add_argument(
        "scan_root",
        nargs="?",
        default=str(DEFAULT_SCAN_ROOT),
        help="Root to scan for contract.yaml files.",
    )
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Frozen shrink-only burn-down baseline.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    scan_root, baseline_path = Path(args.scan_root), Path(args.baseline)

    handlers = declared_handlers(scan_root)
    if len(handlers) < MIN_EXPECTED_DECLARED_HANDLERS:
        sys.stderr.write(
            f"[handler-dispatch-entrypoint] FAIL (vacuity guard): only {len(handlers)} "
            f"contract-declared handlers found under {scan_root} (expected >= "
            f"{MIN_EXPECTED_DECLARED_HANDLERS}). The contract scan is broken; a gate "
            f"over an empty set proves nothing.\n"
        )
        return 1

    baseline = load_baseline(baseline_path)
    live = entrypointless(handlers)

    violations = sorted(live - baseline)
    stale = sorted(baseline - live)
    exit_code = 0

    if violations:
        exit_code = 1
        sys.stderr.write(
            "[handler-dispatch-entrypoint] FAIL: contract-declared handler(s) expose "
            "NEITHER handle() nor handle_async(). Auto-wiring binds these to "
            "_missing_handle, so EVERY dispatch raises ModelOnexError at runtime while "
            "CI stays green:\n"
        )
        for contract, handler in violations:
            sys.stderr.write(f"  - {contract}: {handler}\n")
        sys.stderr.write(
            "\n  Add a def-B `handle(request) -> response` entrypoint. Do NOT add the "
            f"handler to {baseline_path} — that baseline is frozen and shrink-only.\n"
        )

    if stale:
        exit_code = 1
        sys.stderr.write(
            f"[handler-dispatch-entrypoint] FAIL: handler(s) now HAVE a dispatch "
            f"entrypoint but are still listed in {baseline_path}. Remove them; the "
            f"baseline is shrink-only and must never go stale:\n"
        )
        for contract, handler in stale:
            sys.stderr.write(f"  - {contract}: {handler}\n")

    if exit_code == 0:
        distinct = len({declared.key for declared in handlers})
        sys.stderr.write(
            f"[handler-dispatch-entrypoint] OK: {distinct} distinct contract-declared "
            f"handlers ({len(handlers)} declaration rows), {len(live)} entrypointless "
            f"(all in the frozen baseline), 0 new violations.\n"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
