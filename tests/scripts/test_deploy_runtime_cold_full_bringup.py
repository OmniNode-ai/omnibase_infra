# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Coverage for the OMN-13414 cold-lane FULL bring-up path in deploy-runtime.sh.

The dev lane is ephemeral (GC/idle-reclaimed). Bringing a fully COLD lane back
up is materially harder than the warm "recreate the runtime services" restart,
and two undocumented gotchas cost real time during the 2026-06-21 runtime-e2e
run (evidence: .onex_state/runtime-e2e-2026-06-21/02-dev-deploy/):

1. Runtime services are gated behind the compose ``runtime`` profile. A bare
   ``docker compose up -d`` matches NO profiled service and starts NOTHING — the
   ``--profile runtime`` selector is mandatory.
2. deploy-runtime.sh defaults ``BUILD_SOURCE=release``; a cold/GC-reclaimed lane
   must be rebuilt from the merged-dev workspace siblings, which needs
   ``BUILD_SOURCE=workspace`` + ``OMNI_HOME`` + the sibling REF build-args and
   ``scripts/runtime_build/stage_workspace.sh``.

These tests lock the ``--cold`` flag that wires both gotchas into one documented
path: deps + migration one-shots + a FULL ``--profile runtime`` up, built in
workspace mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"


def _deploy_script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _deploy_script_noncomment() -> str:
    """deploy-runtime.sh with comment-only lines stripped.

    Assertions about *active* behavior must not be satisfied by a comment that
    merely mentions the token, so the active-code checks run against this view.
    """
    lines = [
        line
        for line in _deploy_script_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    return "\n".join(lines)


def _function_body(name: str) -> str:
    """Return the source of a top-level shell function ``name() { ... }``.

    Every top-level function in deploy-runtime.sh closes with a ``}`` at column
    0; nested braces are indented. We slice from the ``name() {`` line to the
    next bare ``}`` so an assertion about one function cannot be satisfied by
    code that lives in another.
    """
    text = _deploy_script_text()
    # Anchor at line start so e.g. main() is not matched inside
    # read_repo_ref_or_main().
    anchored = text.find(f"\n{name}() {{")
    start = 0 if text.startswith(f"{name}() {{") else anchored + 1
    assert anchored != -1 or text.startswith(f"{name}() {{"), (
        f"function {name}() not found in deploy-runtime.sh"
    )
    rest = text[start:]
    end_rel = rest.find("\n}\n")
    assert end_rel != -1, f"could not find end of function {name}()"
    return rest[: end_rel + 3]


# ---------------------------------------------------------------------------
# Flag parsing + defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cold_full_bringup_defaults_false() -> None:
    """The cold flag must default OFF so existing invocations are unchanged."""
    assert "COLD_FULL_BRINGUP=false" in _deploy_script_noncomment()


@pytest.mark.unit
def test_cold_flag_is_parsed() -> None:
    """parse_args must accept --cold and set COLD_FULL_BRINGUP=true."""
    body = _function_body("parse_args")
    assert "--cold)" in body, "--cold case missing from parse_args"
    assert "COLD_FULL_BRINGUP=true" in body


# ---------------------------------------------------------------------------
# Gotcha 2: --cold forces workspace build source (never release)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cold_forces_workspace_build_source() -> None:
    """resolve_build_source must return workspace whenever --cold is set.

    A cold/GC-reclaimed lane cannot be rebuilt from the PyPI release packages —
    they cannot carry un-released merged-dev code. --cold therefore overrides the
    default ``release`` source with ``workspace`` (OMN-13414).
    """
    body = _function_body("resolve_build_source")
    assert "COLD_FULL_BRINGUP" in body, (
        "resolve_build_source must honor COLD_FULL_BRINGUP and force workspace."
    )
    assert 'echo "workspace"' in body


@pytest.mark.unit
def test_cold_rejects_explicit_release_build_source() -> None:
    """An operator setting BUILD_SOURCE=release WITH --cold is a contradiction."""
    text = _deploy_script_noncomment()
    # Fail-fast rather than silently overriding the operator's BUILD_SOURCE.
    assert "BUILD_SOURCE=release was set" in _deploy_script_text()
    assert "exit 64" in text


@pytest.mark.unit
def test_cold_incompatible_with_prod_lane() -> None:
    """--cold is a workspace (non-main-lineage) build; it cannot target prod."""
    text = _deploy_script_text()
    assert "--cold is a workspace-mode" in text, (
        "Missing the --cold/--prod mutual-exclusion guard. Workspace images are "
        "stability-candidate / non-main-lineage and the prod-promotion gate "
        "refuses them (OMN-13669)."
    )


# ---------------------------------------------------------------------------
# Gotcha 1: full bring-up uses the mandatory profile and is NOT --no-deps
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bringup_full_stack_defined() -> None:
    """A dedicated full bring-up function must exist."""
    assert "bringup_full_stack()" in _deploy_script_text()


@pytest.mark.unit
def test_bringup_full_stack_uses_profile_and_full_fanout() -> None:
    """The full bring-up must pass --profile and must NOT pass --no-deps.

    Gotcha 1: a bare ``docker compose up -d`` starts nothing because runtime
    services are gated behind the ``runtime`` profile. Unlike restart_services
    (which recreates only the RUNTIME_SERVICES subset with ``--no-deps``), the
    cold full bring-up must fan out across the WHOLE profile and honor the
    compose depends_on chain, so it must NOT pass --no-deps.
    """
    body = _function_body("bringup_full_stack")
    assert '--profile "${COMPOSE_PROFILE}"' in body, (
        "bringup_full_stack must pass the mandatory --profile selector."
    )
    assert "up -d" in body
    # Strip comment lines: the docstring-style comment legitimately *contrasts*
    # with --no-deps, but the active compose command must not pass it.
    active = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--no-deps" not in active, (
        "Cold full bring-up must NOT pass --no-deps — the whole point is to fan "
        "out over the full profile and honor depends_on (OMN-13414)."
    )


# ---------------------------------------------------------------------------
# main() wiring: deps + migration one-shots + full up, under --cold
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_main_cold_path_runs_deps_migration_then_full_up() -> None:
    """Under --cold, main() must run deps + broker + migration one-shots, then full up."""
    body = _function_body("main")
    assert "COLD_FULL_BRINGUP" in body, "main() does not branch on COLD_FULL_BRINGUP"

    core = body.find('ensure_core_infra_ready "${deploy_target}"')
    warm = body.find('warm_broker_topic_provisioning "${deploy_target}"')
    mig = body.find('run_runtime_migration_preflight "${deploy_target}"')
    full = body.find('bringup_full_stack "${deploy_target}"')

    assert core != -1, "ensure_core_infra_ready not called in main()"
    assert warm != -1, "warm_broker_topic_provisioning not called in main()"
    assert mig != -1, "run_runtime_migration_preflight not called in main()"
    assert full != -1, "bringup_full_stack not called in main()"
    assert core < warm < mig < full, (
        "Cold bring-up order must be core-infra -> broker warmup -> migration "
        "one-shots -> FULL up (OMN-13414 / OMN-13594 / OMN-13220)."
    )


@pytest.mark.unit
def test_main_cold_exports_kafka_cold_start_budget() -> None:
    """The cold path must also raise KAFKA_TIMEOUT_SECONDS for the cold boot."""
    body = _function_body("main")
    # The cold-start consumer-join budget guard (OMN-13220) must cover the cold
    # full bring-up, not just the warm --restart path.
    assert "COLD_FULL_BRINGUP" in body
    assert 'export KAFKA_TIMEOUT_SECONDS="${cold_start_timeout}"' in body


@pytest.mark.unit
def test_main_cold_runs_verify() -> None:
    """Verification must run after a cold bring-up, not only after --restart."""
    body = _function_body("main")
    verify = body.find('verify_deployment "${git_sha}"')
    assert verify != -1, "verify_deployment not called in main()"
    # The verify guard must reference the cold flag (before the verify call) so a
    # cold-only invocation is still verified, not only a warm --restart.
    assert "COLD_FULL_BRINGUP" in body[:verify], (
        "verify_deployment must run under --cold as well as --restart."
    )


@pytest.mark.unit
def test_summary_treats_cold_path_as_containers_running() -> None:
    """After --cold, summary must not tell the operator to start containers."""
    body = _function_body("show_summary")
    assert '"${COLD_FULL_BRINGUP}" == false' in body, (
        "show_summary must treat --cold like --restart because both paths already "
        "bring containers up before the summary is printed."
    )


# ---------------------------------------------------------------------------
# Operator-facing documentation in usage()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_usage_documents_cold_flag() -> None:
    """usage() must document --cold and both gotchas it solves."""
    body = _function_body("usage")
    assert "--cold" in body, "usage() does not document --cold"
    assert "workspace" in body, "usage() must mention workspace-mode build for --cold"
    assert "profile" in body, "usage() must mention the mandatory runtime profile"
