# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""OMN-10858 -- end-to-end cloud workflow proof harness (skeleton).

Chain proven: **login -> tenant -> submit -> terminal readback -> cross-tenant
denial**, against the managed-staging one-gateway / one-synthetic-tenant surface
frozen by OMN-15123.

Status of this file
-------------------
This is a **skeleton**, and it says so out loud rather than pretending otherwise.
Every stage function names *exactly* which endpoint or topic it asserts against
(see ``STAGE_SURFACES``, matched field-by-field against
``docs/runbooks/managed-staging-proof-kit/fields.yaml`` by
``tests/ci/test_managed_staging_proof_kit_seam.py``), but the live bodies raise
:class:`StageNotImplementedError`. They are filled in when the canary lane is up; the
skeleton exists so the *shape* of the proof is reviewable and seam-checked now,
not invented under time pressure on the day of the run.

Safety posture
--------------
* ``--live`` **defaults OFF**. With no flags this program performs **no network,
  no broker, and no database I/O**: it prints the plan -- each stage and the
  surface it would assert against -- and exits 0.
* ``--live`` requires a fully resolved :class:`HarnessConfig`; anything missing
  raises :class:`HarnessConfigError`. It **fails closed**, never degrades to a
  partial run, and there is no default/fallback endpoint anywhere in this file.
* Nothing here targets prod. Gateway, topic prefix, brokers, and DSN all come
  from explicit flags or ``ONEX_E2E_*`` environment variables.

Usage
-----
::

    # dry run -- safe anywhere, and the default
    uv run python scripts/proof/e2e_cloud_workflow_harness.py

    # live run -- HELD FOR OPERATOR; every value must be supplied
    uv run python scripts/proof/e2e_cloud_workflow_harness.py --live \\
        --gateway-base-url https://<gateway> \\
        --tenant-id <uuid> --other-tenant-id <uuid> \\
        --topic-prefix onex.mstg1. \\
        --bootstrap-servers <msk-bootstrap> \\
        --projection-dsn "$CANARY_DSN"

A completed live run emits the per-stage evidence that OMN-10858's
``workflow_receipt.json`` is assembled from; the receipt's verifier must not be
the runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- errors -----------------------------------------------------------------


class HarnessConfigError(RuntimeError):
    """A live run was requested without a fully resolved configuration.

    Raised *before* any I/O. This is the fail-closed boundary: the harness never
    substitutes a default endpoint, a plaintext broker, or a relaxed sslmode.
    """


class StageNotImplementedError(NotImplementedError):
    """A live stage body is not implemented yet in this skeleton."""


# --- config -----------------------------------------------------------------


@dataclass(frozen=True)
class HarnessConfig:
    """Fully resolved live-run configuration. Every field is required."""

    gateway_base_url: str
    tenant_id: str
    other_tenant_id: str
    topic_prefix: str
    bootstrap_servers: str
    projection_dsn: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> HarnessConfig:
        """Resolve config from flags, falling back to ``ONEX_E2E_*`` env vars.

        Missing or blank values raise :class:`HarnessConfigError` naming every
        missing key at once -- never one-at-a-time discovery mid-run.
        """
        env_keys = {
            "gateway_base_url": "ONEX_E2E_GATEWAY_BASE_URL",
            "tenant_id": "ONEX_E2E_TENANT_ID",
            "other_tenant_id": "ONEX_E2E_OTHER_TENANT_ID",
            "topic_prefix": "ONEX_E2E_TOPIC_PREFIX",
            "bootstrap_servers": "ONEX_E2E_BOOTSTRAP_SERVERS",
            "projection_dsn": "ONEX_E2E_PROJECTION_DSN",
        }
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for name, env_key in env_keys.items():
            value = (getattr(args, name, None) or os.environ.get(env_key) or "").strip()
            if not value:
                missing.append(f"--{name.replace('_', '-')} (or ${env_key})")
            resolved[name] = value
        if missing:
            raise HarnessConfigError(
                "live run refused -- unresolved configuration: " + ", ".join(missing)
            )
        if resolved["tenant_id"] == resolved["other_tenant_id"]:
            raise HarnessConfigError(
                "live run refused -- --other-tenant-id must differ from --tenant-id, "
                "or the cross-tenant denial stage is vacuous"
            )
        return cls(**resolved)


# --- results ----------------------------------------------------------------


@dataclass
class StageResult:
    """One stage outcome. ``PLANNED`` means "not executed", never "passed"."""

    stage_id: str
    status: str  # PLANNED | PASS | FAIL | ERROR
    surface: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessPlan:
    live: bool
    results: list[StageResult]

    @property
    def ok(self) -> bool:
        return all(r.status in {"PLANNED", "PASS"} for r in self.results)


# --- stage surfaces (mirrored in the proof-kit manifest) --------------------

STAGE_SURFACES: dict[str, str] = {
    "login": (
        "gateway: POST {gateway_base_url}/auth/token (Keycloak-backed); "
        "tenant-scoped claim required"
    ),
    "tenant": "gateway: GET {gateway_base_url}/v1/tenants/{tenant_id}",
    "submit": "topic: {topic_prefix}onex.cmd.omnimarket.alpha-text-analysis-requested.v1",
    "terminal_readback": (
        "topic: {topic_prefix}onex.evt.omnimarket.alpha-text-analysis-completed.v1 "
        "(or ...-failed.v1); table: alpha_workflow_results"
    ),
    "cross_tenant_denial": (
        "gateway: GET {gateway_base_url}/v1/workflows/{workflow_id} with tenant B's token"
    ),
}


def _surface(stage_id: str, config: HarnessConfig | None) -> str:
    """Render a stage surface, with placeholders left intact on a dry run."""
    template = STAGE_SURFACES[stage_id]
    if config is None:
        return template
    return template.format(
        gateway_base_url=config.gateway_base_url,
        tenant_id=config.tenant_id,
        topic_prefix=config.topic_prefix,
        workflow_id="<workflow_id>",
    )


# --- stages -----------------------------------------------------------------
#
# Each stage takes the resolved config plus the mutable run context (tokens,
# correlation_id, workflow_id produced by earlier stages) and returns a
# StageResult. The assertions are written out in each docstring so the skeleton
# is reviewable against OMN-10858's DoD before the bodies exist.


def stage_login(config: HarnessConfig, ctx: dict[str, Any]) -> StageResult:
    """Assert: the gateway issues a token whose claims are scoped to ``tenant_id``.

    FAIL if: no token, or a token whose tenant claim is absent, wildcard, or
    mismatched -- a wildcard-tenant token would make ``cross_tenant_denial``
    vacuous.
    Evidence: token claim set (never the raw token), gateway auth context.
    """
    raise StageNotImplementedError(
        "stage_login: live body pending canary lane bring-up"
    )


def stage_tenant(config: HarnessConfig, ctx: dict[str, Any]) -> StageResult:
    """Assert: the synthetic tenant resolves and equals the frozen tuple's tenant.

    FAIL if: 404, or the returned tenant id differs from OMN-15123's
    ``synthetic_tenant_id`` -- proving against a different tenant proves nothing
    about the canary.
    Evidence: tenant id, tenant status, gateway response hash.
    """
    raise StageNotImplementedError(
        "stage_tenant: live body pending canary lane bring-up"
    )


def stage_submit(config: HarnessConfig, ctx: dict[str, Any]) -> StageResult:
    """Assert: one reference command is published and accepted.

    Payload ``{prompt, max_tokens, tenant_id, correlation_id}`` onto
    ``{topic_prefix}onex.cmd.omnimarket.alpha-text-analysis-requested.v1``.
    FAIL if: an invalid schema is rejected *after* publication rather than before
    it (OMN-10858 negative case 2 requires pre-publication rejection), or the
    broker auto-created the topic (auto-create must be OFF -- see OMN-15124).
    Evidence: the full submitted envelope, correlation_id, publish offset.
    """
    raise StageNotImplementedError(
        "stage_submit: live body pending canary lane bring-up"
    )


def stage_terminal_readback(config: HarnessConfig, ctx: dict[str, Any]) -> StageResult:
    """Assert: a terminal event AND a projection row, correlation-linked.

    Reads ``...-completed.v1`` / ``...-failed.v1`` and the
    ``alpha_workflow_results`` row carrying the same correlation/causation chain.
    FAIL if: no terminal event inside the deadline; a terminal event with no
    projection row (or the reverse); a correlation/causation break; or total
    latency over OMN-10858's 30s budget for the reference workflow.
    Evidence: every emitted event with correlation + causation links, projection
    row before and after, SHA-256 of the result payload, latency_ms.
    """
    raise StageNotImplementedError(
        "stage_terminal_readback: live body pending canary lane bring-up"
    )


def stage_cross_tenant_denial(
    config: HarnessConfig, ctx: dict[str, Any]
) -> StageResult:
    """NEGATIVE CONTROL. Assert: tenant B cannot read tenant A's workflow.

    Re-requests the same ``workflow_id`` with a token scoped to
    ``other_tenant_id``.
    PASS only on 403/404. **200 is a FAIL, and a transport error is an ERROR, not
    a pass** -- a connection refusal proves nothing about authorization.
    Evidence: status code, response body hash, both tenant ids.
    """
    raise StageNotImplementedError(
        "stage_cross_tenant_denial: live body pending canary lane bring-up"
    )


StageFn = Callable[[HarnessConfig, dict[str, Any]], StageResult]

STAGES: dict[str, StageFn] = {
    "login": stage_login,
    "tenant": stage_tenant,
    "submit": stage_submit,
    "terminal_readback": stage_terminal_readback,
    "cross_tenant_denial": stage_cross_tenant_denial,
}


# --- driver -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e2e_cloud_workflow_harness",
        description=(
            "OMN-10858 end-to-end cloud workflow proof harness. "
            "Dry run by default; --live is required to touch anything."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="execute the assertions against the real surfaces (default: OFF, plan only)",
    )
    parser.add_argument("--gateway-base-url", dest="gateway_base_url", default=None)
    parser.add_argument("--tenant-id", dest="tenant_id", default=None)
    parser.add_argument(
        "--other-tenant-id",
        dest="other_tenant_id",
        default=None,
        help="tenant B, used only by the cross-tenant denial negative control",
    )
    parser.add_argument("--topic-prefix", dest="topic_prefix", default=None)
    parser.add_argument("--bootstrap-servers", dest="bootstrap_servers", default=None)
    parser.add_argument("--projection-dsn", dest="projection_dsn", default=None)
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="directory to write per-stage evidence into (live runs only)",
    )
    return parser


def run(args: argparse.Namespace) -> HarnessPlan:
    """Plan (default) or execute (``--live``) the proof chain.

    A dry run returns one ``PLANNED`` result per stage and performs no I/O.
    """
    if not args.live:
        return HarnessPlan(
            live=False,
            results=[
                StageResult(stage_id=sid, status="PLANNED", surface=_surface(sid, None))
                for sid in STAGES
            ],
        )

    config = HarnessConfig.from_args(args)
    ctx: dict[str, Any] = {}
    results: list[StageResult] = []
    for stage_id, fn in STAGES.items():
        try:
            results.append(fn(config, ctx))
        except StageNotImplementedError as exc:
            results.append(
                StageResult(
                    stage_id=stage_id,
                    status="ERROR",
                    surface=_surface(stage_id, config),
                    detail=str(exc),
                )
            )
            break
    return HarnessPlan(live=True, results=results)


def _render(plan: HarnessPlan) -> str:
    mode = "LIVE" if plan.live else "DRY RUN (no I/O performed; pass --live to execute)"
    lines = [f"OMN-10858 e2e cloud workflow proof harness -- {mode}", ""]
    for result in plan.results:
        lines.append(f"  [{result.status:<7}] {result.stage_id}")
        lines.append(f"            surface: {result.surface}")
        if result.detail:
            lines.append(f"            detail:  {result.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = run(args)
    except HarnessConfigError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(_render(plan))
    if plan.live and args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "harness_stage_results.json").write_text(
            json.dumps(
                [
                    {
                        "stage_id": r.stage_id,
                        "status": r.status,
                        "surface": r.surface,
                        "detail": r.detail,
                        "evidence": r.evidence,
                    }
                    for r in plan.results
                ],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 0 if plan.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
