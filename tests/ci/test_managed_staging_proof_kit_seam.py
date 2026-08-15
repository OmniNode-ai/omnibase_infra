# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Seam test for the managed-staging proof kit.

Tickets: OMN-15123 (frozen one-tenant contract), OMN-15124 (candidate-in-isolation
compatibility proof), OMN-15125 (Aug-5 readiness/rollback packet), OMN-10858
(end-to-end cloud workflow proof harness).

The seam has three sides and this test matches them field-by-field:

1. **Ticket acceptance criteria** -- pinned here as ``REQUIRED_FIELDS`` /
   ``REQUIRED_STAGES``, transcribed from the ticket bodies. This side is the
   contract; it does not read the manifest, so a field silently dropped from the
   manifest fails here rather than passing vacuously.
2. **The manifest** -- ``docs/runbooks/managed-staging-proof-kit/fields.yaml``:
   every required field present, every field carrying a real evidence source
   (a placeholder such as ``TBD`` is a failure, because "the evidence source is
   named" is exactly what the tickets ask for).
3. **The rendered surfaces** -- each markdown packet template must carry one
   table row per manifest field id with a non-placeholder evidence cell, and the
   harness module must expose one ``stage_<id>`` callable per manifest stage,
   with ``--live`` defaulting OFF.

Failure mode this guards: a packet that *looks* complete but leaves an evidence
source unnamed, so the go/no-go decision is made on prose instead of a readback.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT / "docs" / "runbooks" / "managed-staging-proof-kit" / "fields.yaml"
)

# Placeholder tokens that mean "the evidence source is not actually named".
PLACEHOLDER_RE = re.compile(r"\b(TBD|TODO|FIXME|N/?A|XXX|\?{3,})\b", re.IGNORECASE)

# --- side 1: ticket acceptance criteria, transcribed ------------------------

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    # OMN-15123: "account/region/namespace, one gateway, one synthetic tenant,
    # source/image/config digests, approved onex.mstg1. catalog, unique MSK
    # epoch, signed start/reset policy, rollback authority, zero-prod-diff
    # assertion, omnidash exclusion" + digest readback + plan-row binding.
    "one_tenant_contract_freeze": (
        "aws_account",
        "aws_region",
        "k8s_namespace",
        "gateway_endpoint",
        "synthetic_tenant_id",
        "source_digest",
        "image_digest",
        "config_digest",
        "topic_catalog",
        "zero_collision_readback",
        "msk_epoch",
        "group_start_reset_policy",
        "rollback_authority",
        "zero_prod_diff",
        "omnidash_exclusion",
        "plan_row_binding",
    ),
    # OMN-15124: IAM/TLS signer + token refresh, explicit bootstrap with
    # auto-create off, broker/group perms, RDS verify-full, typed config
    # authority, no raw endpoint fallback, dashboard zero authority, negative
    # control, isolation lane, plan-row binding.
    "candidate_isolation_compatibility": (
        "isolation_lane",
        "msk_iam_signer",
        "token_refresh_cycle",
        "auto_create_off",
        "explicit_topic_bootstrap",
        "negative_control_out_of_catalog",
        "broker_group_perms",
        "rds_verify_full",
        "typed_config_authority",
        "no_raw_endpoint_fallback",
        "dashboard_zero_authority",
        "plan_row_binding",
    ),
    # OMN-15125: source + previous digests, linux/amd64 manifest, config/policy
    # hashes, vulnerability result, A6 thresholds with live samples, staffed
    # monitoring owner/actions, B12 psql readback, OMN-14772 teardown readback,
    # executable rollback + the three items that convert Aug 5 from target to
    # forecast (blocker graph, dated chain with slack, T20 handoff).
    "aug5_readiness_rollback": (
        "source_digest",
        "previous_digest",
        "amd64_manifest",
        "config_hash",
        "policy_hash",
        "vulnerability_result",
        "a6_thresholds_with_live_samples",
        "monitoring_owner_actions",
        "b12_psql_readback",
        "teardown_readback",
        "executable_rollback",
        "reconciled_blocker_graph",
        "dated_chain_with_slack",
        "t20_handoff",
        "plan_row_binding",
    ),
}

# OMN-10858 reference chain: login -> tenant -> submit -> terminal readback ->
# cross-tenant denial.
REQUIRED_STAGES: tuple[str, ...] = (
    "login",
    "tenant",
    "submit",
    "terminal_readback",
    "cross_tenant_denial",
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.is_file(), f"proof-kit manifest missing: {MANIFEST_PATH}"
    loaded = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), "manifest must be a mapping"
    return loaded


def _packet(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    packets = manifest.get("packets") or {}
    assert key in packets, f"manifest is missing packet '{key}'"
    packet = packets[key]
    assert isinstance(packet, dict), f"packet '{key}' must be a mapping"
    return packet


def _fields(packet: dict[str, Any]) -> list[dict[str, Any]]:
    fields = packet.get("fields") or []
    assert isinstance(fields, list), "packet.fields must be a list"
    return [f for f in fields if isinstance(f, dict)]


@pytest.mark.parametrize("packet_key", sorted(REQUIRED_FIELDS))
def test_manifest_covers_ticket_required_fields(
    manifest: dict[str, Any], packet_key: str
) -> None:
    """Side 1 -> side 2: every field the ticket names exists in the manifest."""
    packet = _packet(manifest, packet_key)
    present = {f.get("id") for f in _fields(packet)}
    missing = [fid for fid in REQUIRED_FIELDS[packet_key] if fid not in present]
    assert not missing, (
        f"packet '{packet_key}' ({packet.get('ticket')}) is missing required "
        f"fields: {missing}"
    )


@pytest.mark.parametrize("packet_key", sorted(REQUIRED_FIELDS))
def test_every_field_names_a_real_evidence_source(
    manifest: dict[str, Any], packet_key: str
) -> None:
    """No field may ship with a placeholder evidence source."""
    packet = _packet(manifest, packet_key)
    bad: list[str] = []
    for field in _fields(packet):
        fid = str(field.get("id"))
        source = str(field.get("evidence_source") or "").strip()
        if not source or PLACEHOLDER_RE.search(source):
            bad.append(f"{fid}={source!r}")
        if not str(field.get("label") or "").strip():
            bad.append(f"{fid}=<missing label>")
    assert not bad, f"packet '{packet_key}' has unnamed evidence sources: {bad}"


@pytest.mark.parametrize("packet_key", sorted(REQUIRED_FIELDS))
def test_template_has_a_row_per_manifest_field(
    manifest: dict[str, Any], packet_key: str
) -> None:
    """Side 2 -> side 3: the markdown template renders every manifest field."""
    packet = _packet(manifest, packet_key)
    template_rel = str(packet.get("template") or "")
    assert template_rel, f"packet '{packet_key}' declares no template"
    template_path = REPO_ROOT / template_rel
    assert template_path.is_file(), f"template missing: {template_path}"
    text = template_path.read_text(encoding="utf-8")

    rows: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        rows[cells[0].strip("`")] = cells[-1]

    missing = [f["id"] for f in _fields(packet) if str(f.get("id")) not in rows]
    assert not missing, (
        f"template {template_rel} is missing a table row for manifest fields: {missing}"
    )

    placeholder = [
        fid
        for fid in (str(f.get("id")) for f in _fields(packet))
        if not rows[fid].strip("` ") or PLACEHOLDER_RE.search(rows[fid])
    ]
    assert not placeholder, (
        f"template {template_rel} leaves the evidence-source cell unnamed for: {placeholder}"
    )


def _load_harness(manifest: dict[str, Any]) -> ModuleType:
    script_rel = str((manifest.get("harness") or {}).get("script") or "")
    assert script_rel, "manifest declares no harness script"
    script_path = REPO_ROOT / script_rel
    assert script_path.is_file(), f"harness script missing: {script_path}"
    spec = importlib.util.spec_from_file_location(
        "managed_staging_e2e_harness_under_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves ``sys.modules[cls.__module__]`` while building a
    # frozen dataclass, so the module must be registered before exec.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_harness_declares_the_required_stages(manifest: dict[str, Any]) -> None:
    stages = [
        str(s.get("id"))
        for s in ((manifest.get("harness") or {}).get("stages") or [])
        if isinstance(s, dict)
    ]
    assert tuple(stages) == REQUIRED_STAGES, (
        "harness stage order must be the OMN-10858 chain "
        f"{REQUIRED_STAGES}, got {tuple(stages)}"
    )
    for stage in (manifest.get("harness") or {}).get("stages") or []:
        surface = str(stage.get("surface") or "").strip()
        assert surface and not PLACEHOLDER_RE.search(surface), (
            f"stage {stage.get('id')!r} does not name a real endpoint/topic surface: {surface!r}"
        )


def test_harness_implements_every_manifest_stage(manifest: dict[str, Any]) -> None:
    module = _load_harness(manifest)
    registry = getattr(module, "STAGES", None)
    assert isinstance(registry, dict), "harness must expose a STAGES registry mapping"
    declared = [
        str(s.get("id"))
        for s in ((manifest.get("harness") or {}).get("stages") or [])
        if isinstance(s, dict)
    ]
    assert list(registry) == declared, (
        f"harness STAGES {list(registry)} does not match manifest stages {declared}"
    )
    for stage_id, fn in registry.items():
        assert callable(fn), f"stage {stage_id} is not callable"


def test_live_flag_defaults_off_and_dry_run_touches_nothing(
    manifest: dict[str, Any],
) -> None:
    """The harness must be safe to run with no flags: plan only, no I/O."""
    module = _load_harness(manifest)
    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.live is False, "--live must default to OFF"

    plan = module.run(args)
    assert plan.live is False
    assert [r.stage_id for r in plan.results] == list(module.STAGES)
    assert all(r.status == "PLANNED" for r in plan.results), (
        "a dry run must not execute any assertion: "
        f"{[(r.stage_id, r.status) for r in plan.results]}"
    )
    assert all(r.surface for r in plan.results), (
        "every planned stage must print its surface"
    )


def test_live_mode_fails_closed_without_config(manifest: dict[str, Any]) -> None:
    """--live with no resolved config must refuse, not silently no-op."""
    module = _load_harness(manifest)
    args = module.build_parser().parse_args(["--live"])
    with pytest.raises(module.HarnessConfigError):
        module.run(args)
