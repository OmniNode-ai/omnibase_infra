# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the contract-declared OTEL exporter endpoint for the ONEX kernel.

The kernel's ``descriptor.otel_exporter_otlp_endpoint`` is the single source of
truth for the OTLP HTTP endpoint ``configure_tracing`` exports spans to
(OMN-13558 Wave-1 endpoint→overlay migration). It is declared with the
``${env.VAR}`` overlay convention so an operator overlay / the per-lane
``tracing`` bundle env supplies the real endpoint per lane (e.g.
``http://phoenix:6006``) — never read directly from ``os.environ`` in
``runtime/tracing.py``.

Resolution goes through ``expand_contract_env_refs`` — the one sanctioned
env-reading boundary in the overlay package — so ``configure_tracing`` never
reads ``os.environ`` for the endpoint directly.

Unlike the fail-closed service endpoints (QDRANT_URL / GRAPH_BOLT_URI), OTEL
tracing is **opt-in** by design (OMN-3811): an unset endpoint is a valid
"tracing disabled" state, not a misconfigured deploy. The descriptor therefore
returns the resolved value (empty string when unset, via the contract's empty
inline default) and leaves the enable/disable decision to the caller — it does
NOT raise on absence.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.runtime.overlay.contract_env_ref import expand_contract_env_refs

_CONTRACT = Path(__file__).resolve().parent / "tracing_contract.yaml"


def _load_contract(contract_path: Path) -> dict[str, object]:
    # ONEX_EXCLUDE: io_audit - Module-level contract load keeps tracing policy contract-owned
    with contract_path.open(encoding="utf-8") as contract_file:
        raw = yaml.safe_load(contract_file)
    if not isinstance(raw, dict):
        raise ValueError(f"contract {contract_path} must contain a mapping")
    return raw


def contract_otel_exporter_endpoint(contract_path: Path = _CONTRACT) -> str:
    """Return the resolved ``descriptor.otel_exporter_otlp_endpoint`` (may be empty).

    The value is contract-declared (overridable by an operator overlay contract)
    via the ``${env.OTEL_EXPORTER_OTLP_ENDPOINT:}`` convention — never read from
    ``os.environ`` directly in ``runtime/tracing.py``. Returns the resolved,
    stripped endpoint, or the empty string when the var is unset (preserving the
    opt-in "tracing disabled" semantics of OMN-3811). The caller treats empty as
    "tracing disabled".

    Raises ``ValueError`` only on a structurally malformed contract (missing
    descriptor mapping or non-string field) — a deploy-time wiring bug, distinct
    from the runtime opt-out of an unset endpoint.
    """
    raw = _load_contract(contract_path)
    descriptor = raw.get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError(
            f"contract {contract_path} must declare a descriptor mapping with "
            "otel_exporter_otlp_endpoint"
        )
    declared = descriptor.get("otel_exporter_otlp_endpoint")
    if not isinstance(declared, str):
        raise ValueError(
            f"contract {contract_path} must declare a string "
            "descriptor.otel_exporter_otlp_endpoint (the "
            "${env.OTEL_EXPORTER_OTLP_ENDPOINT:} overlay value the kernel uses "
            "as the OTLP exporter endpoint)"
        )
    return expand_contract_env_refs(declared).strip()


__all__: list[str] = ["contract_otel_exporter_endpoint"]
