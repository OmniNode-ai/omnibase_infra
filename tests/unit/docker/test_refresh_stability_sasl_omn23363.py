# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The governed stability refresh re-asserts the SASL flip every run (OMN-23363).

``scripts/runtime_build/refresh_stability_lane.sh`` composes
``REFRESH_BUILD_SERVICES`` from ``CORE_SERVICES`` + the two SASL one-shots + the
projection writers, hands that one array to onex-runtime-deploy as the sole
``RUNTIME_BUILD_SERVICES_OVERRIDE``, and re-raises a nonzero deploy exit so a
failed deploy fails the refresh instead of reading as a green one. These
tests pin that wiring from the script text alone:

* both SASL one-shots ride the build scope, principal before flip;
* the rollback set is the build scope minus exactly the two image-only
  one-shots (they run a pinned upstream image, so no
  ``<project>-<service>:latest`` anchor exists for them);
* the deploy exit path is captured, guarded and re-raised;
* the scope reaches the deploy through exactly one override line;
* the deploy step itself is RUN, with a stub deploy, so a nonzero deploy exit fails the
  refresh, ``LANE_BROKER_AUTH_ENV_FILE`` reaches the deploy's environment and a ref
  with no SASL flip is deployed without the flip one-shots (OMN-24908).
"""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

from tests.unit.docker.test_stability_sasl_omn23363 import (
    _merged_services,
    _readers,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "runtime_build" / "refresh_stability_lane.sh"
SCRAM_USER = "lane-redpanda-scram-user"
SASL_ENABLE = "lane-redpanda-sasl-enable"
ONE_SHOTS = (SCRAM_USER, SASL_ENABLE)
OVERRIDE_EXPORT_PREFIX = "export RUNTIME_BUILD_SERVICES_OVERRIDE="
OVERRIDE_EXPORT = f'{OVERRIDE_EXPORT_PREFIX}"${{REFRESH_BUILD_SERVICES[*]}}"'
CAPTURE_NEEDLE = "|| DEPLOY_EXIT=$?"
DEPLOY_GUARD = 'if [[ "${DEPLOY_EXIT}" -ne 0 ]]; then'
EXIT_LINE = 'exit "${DEPLOY_EXIT}"'


def _script_lines() -> list[str]:
    return SCRIPT_PATH.read_text(encoding="utf-8").splitlines()


def _find_prefix(lines: list[str], prefix: str) -> int:
    matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
    if len(matches) != 1:
        pytest.fail(f"expected exactly one line starting {prefix!r}, got {matches}")
    return matches[0]


def _scope_block(lines: list[str]) -> str:
    """From the ``readonly CORE_SERVICES=`` line through the ROLLBACK_SERVICES line."""
    start = _find_prefix(lines, "readonly CORE_SERVICES=")
    rollback = _find_prefix(lines, "readonly ROLLBACK_SERVICES=(")
    return "\n".join(lines[start : rollback + 1])


def _print_array(block: str, array: str) -> list[str]:
    """Evaluate the scope block under bash and print one array, one per line."""
    command = f'{block}; printf "%s\\n" "${{{array}[@]}}"'
    proc = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"bash failed printing {array}: {proc.stderr.strip()}")
    return (proc.stdout or "").splitlines()


def _core_services(block: str) -> list[str]:
    match = re.search(r"readonly CORE_SERVICES=\(([^)]*)\)", block)
    if match is None:
        pytest.fail("no readonly CORE_SERVICES=(...) in the extracted scope block")
    return match.group(1).split()


def _fail_unless_member(array: str, service: str, members: list[str]) -> None:
    if service not in members:
        pytest.fail(f"{service} missing from {array}: {members}")


def _fail_if_out_of_order(
    array: str,
    first: str,
    second: str,
    members: list[str],
) -> None:
    if members.index(first) >= members.index(second):
        pytest.fail(
            f"{first} (index {members.index(first)}) must precede "
            f"{second} (index {members.index(second)}) in {array}: {members}"
        )


def _line_index(lines: list[str], needle: str, start: int = 0) -> int:
    for idx in range(start, len(lines)):
        if needle in lines[idx]:
            return idx
    return -1


def _stripped_line_index(lines: list[str], exact: str, start: int) -> int:
    for idx in range(start, len(lines)):
        if lines[idx].strip() == exact:
            return idx
    return -1


def _fail_unless_anchor_chain(anchors: dict[str, int]) -> None:
    """Every anchor must exist and appear in strictly increasing line order."""
    missing = [name for name, idx in anchors.items() if idx < 0]
    indexes = list(anchors.values())
    if missing:
        pytest.fail(f"deploy-exit anchors missing from {SCRIPT_PATH.name}: {missing}")
    if any(indexes[i] >= indexes[i + 1] for i in range(len(indexes) - 1)):
        pytest.fail(
            f"deploy-exit anchors not strictly increasing in {SCRIPT_PATH.name}: "
            f"{anchors}"
        )


@pytest.mark.unit
def test_refresh_stability_scope_runs_both_sasl_one_shots_in_order() -> None:
    block = _scope_block(_script_lines())
    build = _print_array(block, "REFRESH_BUILD_SERVICES")
    rollback = _print_array(block, "ROLLBACK_SERVICES")
    for one_shot in ONE_SHOTS:
        _fail_unless_member("REFRESH_BUILD_SERVICES", one_shot, build)
    _fail_if_out_of_order("REFRESH_BUILD_SERVICES", SCRAM_USER, SASL_ENABLE, build)
    for service in _core_services(block):
        _fail_unless_member("REFRESH_BUILD_SERVICES", service, build)
        _fail_if_out_of_order("REFRESH_BUILD_SERVICES", service, SCRAM_USER, build)
    if not rollback:
        pytest.fail("the extracted scope block left ROLLBACK_SERVICES empty")


@pytest.mark.unit
def test_refresh_stability_rollback_set_excludes_the_image_only_one_shots() -> None:
    block = _scope_block(_script_lines())
    build = _print_array(block, "REFRESH_BUILD_SERVICES")
    rollback = _print_array(block, "ROLLBACK_SERVICES")
    for one_shot in ONE_SHOTS:
        if one_shot in rollback:
            pytest.fail(f"{one_shot} must not be in ROLLBACK_SERVICES: {rollback}")
    missing = sorted(set(build) - set(rollback) - set(ONE_SHOTS))
    if missing:
        pytest.fail(f"build-scope services missing from ROLLBACK_SERVICES: {missing}")
    extras = sorted(set(rollback) - set(build))
    if extras:
        pytest.fail(f"ROLLBACK_SERVICES members outside the build scope: {extras}")


@pytest.mark.unit
def test_refresh_stability_deploy_exit_fails_the_refresh() -> None:
    lines = _script_lines()
    override_idx = _line_index(lines, OVERRIDE_EXPORT)
    capture_idx = _line_index(lines, CAPTURE_NEEDLE, override_idx + 1)
    guard_idx = _line_index(lines, DEPLOY_GUARD, capture_idx + 1)
    exit_idx = _stripped_line_index(lines, EXIT_LINE, guard_idx + 1)
    _fail_unless_anchor_chain(
        {
            "override export": override_idx,
            "exit capture": capture_idx,
            "nonzero guard": guard_idx,
            "re-raise exit": exit_idx,
        }
    )


OVERRIDE_ASSIGNMENT = re.compile(
    r"^\s*(?:export\s+|\[\[[^\]]*\]\]\s*\|\|\s*)?RUNTIME_BUILD_SERVICES_OVERRIDE="
)


@pytest.mark.unit
def test_refresh_stability_scope_is_handed_to_the_deploy_as_one_override() -> None:
    """The full scope is exported once; the one other assignment narrows it to the
    rollback set for a ref with no SASL flip. Every assignment is counted, not only
    the ``export`` one, so a third cannot slip in beside them."""
    lines = _script_lines()
    matches = [i for i, line in enumerate(lines) if OVERRIDE_ASSIGNMENT.match(line)]
    if len(matches) != 2:
        pytest.fail(
            f"expected exactly two assignments to RUNTIME_BUILD_SERVICES_OVERRIDE in "
            f"{SCRIPT_PATH.name}, found {len(matches)} "
            f"(1-based lines: {[i + 1 for i in matches]})"
        )
    first, second = (lines[i].strip() for i in matches)
    if first != OVERRIDE_EXPORT:
        pytest.fail(f"unexpected override line: {first!r}")
    if (
        second
        != '[[ "${PRE_SASL_REF}" != true ]] || RUNTIME_BUILD_SERVICES_OVERRIDE="${ROLLBACK_SERVICES[*]}"'
    ):
        pytest.fail(f"unexpected second override assignment: {second!r}")


DEPLOY_BLOCK_START = 'log "=== Build + restart (scoped to:'
AUTH_FILE = "/lane/broker-auth.env"


def _deploy_block(lines: list[str]) -> str:
    """From the build-and-restart log line through the guard's closing ``fi``."""
    start = _find_prefix(lines, DEPLOY_BLOCK_START)
    guard = _line_index(lines, DEPLOY_GUARD, start)
    exit_line = _stripped_line_index(lines, EXIT_LINE, guard + 1)
    assert min(guard, exit_line) > 0, (guard, exit_line)
    assert lines[exit_line + 1].strip() == "fi", lines[exit_line + 1]
    return "\n".join(lines[start : exit_line + 2])


def _run_deploy_block(
    tmp_path: Path,
    *,
    deploy_exit: int,
    pre_sasl_ref: bool = False,
    auth_file: str | None = AUTH_FILE,
) -> subprocess.CompletedProcess[str]:
    """Run the script's own deploy step against a stub ``onex-runtime-deploy``.

    The scope arrays are the script's real declarations. ``LANE_BROKER_AUTH_ENV_FILE``
    is set as a plain shell variable (not exported), the way a sourced operator env
    leaves it, so only the script's own pass-through can hand it to the deploy.
    """
    lines = _script_lines()
    stub = tmp_path / "deploy-stub"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/bin/sh
            {{
              echo "auth=${{LANE_BROKER_AUTH_ENV_FILE-unset}}"
              echo "scope=${{RUNTIME_BUILD_SERVICES_OVERRIDE-unset}}"
            }} > "{tmp_path}/deploy.env"
            exit {deploy_exit}
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(0o755)
    auth = f'LANE_BROKER_AUTH_ENV_FILE="{auth_file}"' if auth_file is not None else ":"
    program = "\n".join(
        [
            "set -euo pipefail",
            'log() { echo "$*"; }',
            'err() { echo "$*" >&2; }',
            # the broker's own state handling is run in tests/scripts/test_stability_lane_rollback_omn18061.py
            'broker_sasl() { [[ "$1" != get ]] || echo true; }',
            "restore_broker_sasl() { :; }",
            _scope_block(lines),
            f'DEPLOY_RUNTIME="{stub}"',
            "OMNI_HOME=/omni REPO_ROOT=/repo COMPOSE_PROJECT=omnibase-infra-lane",
            "REF=ref DEPLOY_SIBLING_REFS= RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS=5",
            "NEW_INFRA_SHA_SHORT=abc",
            f"PRE_SASL_REF={'true' if pre_sasl_ref else 'false'}",
            auth,
            _deploy_block(lines),
            "echo REACHED_AFTER_DEPLOY",
        ]
    )
    env = {k: v for k, v in os.environ.items() if k != "LANE_BROKER_AUTH_ENV_FILE"}
    return subprocess.run(
        ["bash", "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )


def _deploy_seen(tmp_path: Path) -> dict[str, str]:
    seen = (tmp_path / "deploy.env").read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in seen)


@pytest.mark.unit
def test_refresh_stability_a_nonzero_deploy_fails_the_refresh_when_run(
    tmp_path: Path,
) -> None:
    (tmp_path / "ok").mkdir()
    ok = _run_deploy_block(tmp_path / "ok", deploy_exit=0)
    assert ok.returncode == 0, ok.stderr
    assert "REACHED_AFTER_DEPLOY" in ok.stdout
    (tmp_path / "bad").mkdir()
    bad = _run_deploy_block(tmp_path / "bad", deploy_exit=3)
    assert bad.returncode == 3, (bad.stdout, bad.stderr)
    assert "failing the refresh" in bad.stderr
    assert "REACHED_AFTER_DEPLOY" not in bad.stdout


@pytest.mark.unit
def test_refresh_stability_lane_broker_auth_env_file_reaches_the_deploy(
    tmp_path: Path,
) -> None:
    (tmp_path / "set").mkdir()
    run = _run_deploy_block(tmp_path / "set", deploy_exit=0)
    assert run.returncode == 0, run.stderr
    assert _deploy_seen(tmp_path / "set")["auth"] == AUTH_FILE


@pytest.mark.unit
@pytest.mark.parametrize("auth_file", [None, ""], ids=["unset", "empty"])
def test_refresh_stability_lane_broker_auth_env_file_is_not_exported_when_empty(
    tmp_path: Path, auth_file: str | None
) -> None:
    """A host with no auth file hands the deploy NO variable at all.

    An empty export is not inert: it beats a ``LANE_BROKER_AUTH_ENV_FILE`` set in a
    compose ``.env`` (shell environment outranks ``.env``), and every client would
    then start on PLAINTEXT with both one-shots exiting 0. Red at 2604e98f6da8, where
    the script exported ``""``.
    """
    run = _run_deploy_block(tmp_path, deploy_exit=0, auth_file=auth_file)
    assert run.returncode == 0, run.stderr
    assert _deploy_seen(tmp_path)["auth"] == "unset"


@pytest.mark.unit
def test_refresh_stability_a_ref_without_the_flip_is_deployed_without_the_one_shots(
    tmp_path: Path,
) -> None:
    for name in ("sasl", "pre"):
        (tmp_path / name).mkdir()
    assert _run_deploy_block(tmp_path / "sasl", deploy_exit=0).returncode == 0
    scope = _deploy_seen(tmp_path / "sasl")["scope"].split()
    assert set(ONE_SHOTS) <= set(scope)
    assert (
        _run_deploy_block(tmp_path / "pre", deploy_exit=0, pre_sasl_ref=True).returncode
        == 0
    )
    pre = _deploy_seen(tmp_path / "pre")["scope"].split()
    assert not set(ONE_SHOTS) & set(pre), pre
    assert set(pre) == set(scope) - set(ONE_SHOTS)


#: Derived broker clients the refresh need not recreate, each with the reason.
READER_EXEMPTIONS = {
    # OMN-21288: effects run inside main; the deploy retires this service and the
    # refresh must never rebuild it.
    "runtime-effects": "folded into omninode-runtime",
    # A one-shot with restart "no": it has exited before the refresh ends and no
    # long-running client of it exists to lock out.
    "redpanda-partition-cap": "one-shot that has already exited",
}


@pytest.mark.unit
def test_refresh_stability_readers_are_inside_the_build_scope() -> None:
    """Every SASL client the overlay derives is recreated by the refresh.

    ``up --no-deps`` recreates only the services it names, so a client outside
    REFRESH_BUILD_SERVICES keeps the settings it started with while the flip changes
    what the broker requires: a client left on PLAINTEXT is locked out, and nothing
    fails until its next restart.
    """
    scope = set(_print_array(_scope_block(_script_lines()), "REFRESH_BUILD_SERVICES"))
    readers = _readers(_merged_services())
    assert "tenant-projection-writer" in readers, sorted(readers)
    assert set(READER_EXEMPTIONS) <= readers, sorted(set(READER_EXEMPTIONS) - readers)
    outside = sorted(readers - scope - set(READER_EXEMPTIONS))
    assert outside == [], f"derived SASL clients the refresh never recreates: {outside}"
    exempt_in_scope = sorted(set(READER_EXEMPTIONS) & scope)
    assert exempt_in_scope == [], (
        f"an exemption that is in scope needs no exemption: {exempt_in_scope}"
    )


@pytest.mark.unit
def test_refresh_stability_readers_scope_check_fails_when_a_client_is_dropped() -> None:
    """Positive control: the same check over a scope missing one client reports it."""
    scope = set(_print_array(_scope_block(_script_lines()), "REFRESH_BUILD_SERVICES"))
    scope.discard("tenant-projection-writer")
    readers = _readers(_merged_services())
    assert sorted(readers - scope - set(READER_EXEMPTIONS)) == [
        "tenant-projection-writer"
    ]
