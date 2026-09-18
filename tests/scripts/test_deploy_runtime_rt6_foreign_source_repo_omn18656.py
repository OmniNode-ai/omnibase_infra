# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RT-6 must compare each service to the ref of the repo that BUILT it [OMN-18656].

Defect, measured on the .201 compose dev lane 2026-09-18T02:48Z (lane
`dev-lane-warm-redeploy-0230`):

``readback_deployed_ref()`` asserted every in-scope service's
``org.opencontainers.image.revision`` equalled THIS repo's git sha. ``onex-api``
is tag-referenced from an **omninode_infra** image -- ``docker/onex-api`` is a
self-contained ``python:3.12-slim`` app with no omnibase wheel, built by the
lab-overlay applier and pinned through ``ONEX_API_IMAGE`` -- so its revision is
an omninode_infra commit and can never equal an omnibase_infra one. The readback
exited non-zero on a lane where every other service had read back correctly and
every health probe was green::

    FAIL revision 99fdbd375f3b6c17d564b588f505754160b1d1f2
         != intended 25e15ca25b71

and the ensuing rollback walked ``omnibase-infra-omninode-runtime:latest``
BACKWARDS to a pre-build image id on a lane whose containers were all running
the new build.

The fix reads the repo that built the image off the CONTAINER's own labels --
``ai.omninode.image.source-repo`` (stamped by ``deploy_agent/lab_overlay.py``,
already read by ``scripts/ci/check_lane_onex_api_revision.py``) then the OCI
``org.opencontainers.image.source`` URL (stamped by ``docker/Dockerfile.runtime``
on everything this repo builds) -- never a hardcoded list of service names,
which is what goes stale the next time a lane gains a foreign-sourced service.

The second half is the rollback: a ``:latest`` tag rollback is a remedy for a
lane that came up BROKEN. On a lane whose health probes all answered green it is
pure damage, and ``docker tag`` cannot recall a container already recreated on
the new image.

Same seam-level harness as ``test_deploy_runtime_rt6_one_shot_omn16729.py``: the
real functions are extracted from ``scripts/deploy-runtime.sh`` and executed
under bash with only ``docker`` stubbed.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
COMPOSE_FILES_SH = REPO_ROOT / "scripts" / "runtime_build" / "compose_files.sh"

#: The omnibase_infra ref this deploy intends -- the `intended` half of the
#: 2026-09-18 failure line, truncated exactly as the log recorded it.
GIT_SHA = "25e15ca25b71"
#: The omninode_infra ref the lane's onex-api image was actually built from,
#: read live off the container on 2026-09-17 (OMN-18572 fixture).
FOREIGN_SHA = "99fdbd375f3b6c17d564b588f505754160b1d1f2"
VERSION = "9.9.9"
DEPLOY_STARTED_AT = "2026-09-18T02:33:05Z"

FOREIGN = "onex-api"
CORE = "runtime-effects"

OWN_SOURCE_URL = "https://github.com/OmniNode-ai/omnibase_infra"
FOREIGN_SOURCE_REPO = "omninode_infra"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        _script_text(),
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from deploy-runtime.sh"
    )
    return match.group(0)


def _extract_array(name: str) -> str:
    match = re.search(
        rf"^readonly {re.escape(name)}=\(.*?\n\)",
        _script_text(),
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract array {name}=() from deploy-runtime.sh"
    )
    return match.group(0)


def _extract_scalar(name: str) -> str:
    """Bind the harness to the script's OWN value, so renaming this repo in the
    script without renaming it here is a red test rather than a silent pass."""
    match = re.search(
        rf'^readonly {re.escape(name)}="[^"]*"$', _script_text(), re.MULTILINE
    )
    assert match is not None, (
        f"could not extract readonly {name}= from deploy-runtime.sh"
    )
    return match.group(0)


def _write_docker_stub(bin_dir: Path) -> None:
    """A `docker` that answers `compose ps [-q|-aq] <svc>` from disk and
    `inspect -f <go-template>` per template, including the two source-repo
    label spellings the fix reads."""
    stub = bin_dir / "docker"
    stub.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${DOCKER_STUB_DIR}/calls.log"

if [[ "$1" == "compose" ]]; then
    shift
    args=("$@")
    n=${#args[@]}
    all=false
    for ((i = 0; i < n; i++)); do
        case "${args[$i]}" in
            -aq|-a) all=true ;;
        esac
    done
    for ((i = 0; i < n; i++)); do
        if [[ "${args[$i]}" == "ps" ]]; then
            service="${args[$((n - 1))]}"
            if [[ "${all}" == true ]]; then
                map_file="${DOCKER_STUB_DIR}/ps_all/${service}"
            else
                map_file="${DOCKER_STUB_DIR}/ps/${service}"
            fi
            if [[ -f "${map_file}" ]]; then
                cat "${map_file}"
                exit 0
            fi
            exit 0
        fi
    done
    exit 1
fi

if [[ "$1" == "inspect" ]]; then
    shift
    fmt=""
    container=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--format) fmt="$2"; shift 2 ;;
            --format=*) fmt="${1#--format=}"; shift ;;
            *) container="$1"; shift ;;
        esac
    done
    case "${fmt}" in
        *ai.omninode.image.source-repo*)     key="source_repo" ;;
        *org.opencontainers.image.source*)   key="source_url" ;;
        *RestartPolicy*)                     key="restart" ;;
        *State.Status*)                      key="state" ;;
        *ExitCode*)                          key="exit_code" ;;
        *FinishedAt*)                        key="finished_at" ;;
        *)                                   key="revision" ;;
    esac
    f="${DOCKER_STUB_DIR}/inspect/${container}.${key}"
    if [[ -f "${f}" ]]; then
        cat "${f}"
    fi
    exit 0
fi

if [[ "$1" == "tag" ]]; then
    shift
    printf '%s -> %s\n' "$1" "$2" >> "${DOCKER_STUB_DIR}/retags.log"
    exit 0
fi

if [[ "$1" == "rmi" ]]; then
    shift
    printf 'rmi %s\n' "$1" >> "${DOCKER_STUB_DIR}/retags.log"
    exit 0
fi

exit 1
""",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _stub_dir(tmp_path: Path) -> Path:
    stub_dir = tmp_path / "stubs"
    for sub in ("ps", "ps_all", "inspect", "version"):
        (stub_dir / sub).mkdir(parents=True, exist_ok=True)
    (stub_dir / "calls.log").write_text("", encoding="utf-8")
    (stub_dir / "retags.log").write_text("", encoding="utf-8")
    _write_docker_stub(stub_dir)
    return stub_dir


def _run_readback(
    tmp_path: Path,
    *,
    services: list[str],
    running: dict[str, str],
    all_containers: dict[str, str],
    inspect: dict[str, dict[str, str]],
    expected_revisions: str | None = None,
) -> subprocess.CompletedProcess[str]:
    stub_dir = _stub_dir(tmp_path)

    for service, container in running.items():
        (stub_dir / "ps" / service).write_text(container + "\n", encoding="utf-8")
    for service, container in all_containers.items():
        (stub_dir / "ps_all" / service).write_text(container + "\n", encoding="utf-8")
    for container, fields in inspect.items():
        for key, value in fields.items():
            (stub_dir / "inspect" / f"{container}.{key}").write_text(
                value, encoding="utf-8"
            )

    services_literal = " ".join(f'"{s}"' for s in services)
    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_step() { printf 'STEP: %s\\n' \"$*\" >&2; }",
            "log_info() { printf 'INFO: %s\\n' \"$*\" >&2; }",
            "log_warn() { printf 'WARN: %s\\n' \"$*\" >&2; }",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            "log_cmd() { printf 'CMD: %s\\n' \"$*\" >&2; }",
            f'DEPLOY_STARTED_AT="{DEPLOY_STARTED_AT}"',
            f'source "{COMPOSE_FILES_SH}"',
            _extract_function("resolve_lane_runtime_container_name"),
            _extract_array("DEV_LANE_ONLY_RUNTIME_SERVICES"),
            _extract_array("STABILITY_TEST_LANE_ONLY_RUNTIME_SERVICES"),
            _extract_scalar("OWN_SOURCE_REPO"),
            _extract_function("resolve_lane_runtime_services"),
            _extract_function("service_is_one_shot"),
            _extract_function("readback_one_shot_service"),
            _extract_function("resolve_service_source_repo"),
            _extract_function("resolve_expected_foreign_revision"),
            _extract_function("readback_foreign_sourced_service"),
            _extract_function("readback_deployed_ref"),
            f'RUNTIME_BUILD_SERVICES_OVERRIDE="{" ".join(services)}"',
            f"RUNTIME_BUILD_SERVICES=({services_literal})",
            (
                f'readback_deployed_ref "{GIT_SHA}" "{VERSION}" '
                f'"omnibase-infra" "{REPO_ROOT}" "/tmp/fake-deploy-target"'
            ),
        ]
    )

    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    env["DOCKER_STUB_DIR"] = str(stub_dir)
    if expected_revisions is not None:
        env["READBACK_EXPECTED_REVISIONS"] = expected_revisions
    else:
        env.pop("READBACK_EXPECTED_REVISIONS", None)
    return subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=False, env=env
    )


def _foreign_container(revision: str = FOREIGN_SHA) -> dict[str, str]:
    """The onex-api container's real label shape, as read live on 2026-09-17:
    an explicit `ai.omninode.image.source-repo`, a revision from THAT repo, and
    no OCI source URL at all."""
    return {
        "restart": "unless-stopped",
        "state": "running",
        "source_repo": FOREIGN_SOURCE_REPO,
        "revision": revision,
    }


def _own_container(revision: str = GIT_SHA) -> dict[str, str]:
    """A lane-built container: no explicit source-repo label, an OCI source URL
    naming this repo, and a revision stamped from this build's VCS_REF."""
    return {
        "restart": "unless-stopped",
        "state": "running",
        "source_url": OWN_SOURCE_URL,
        "revision": revision,
    }


# ─── AC1: a foreign-sourced service at its OWN repo's ref passes ──────────────


@pytest.mark.unit
def test_foreign_sourced_service_at_its_own_repo_ref_passes(tmp_path: Path) -> None:
    """AC1/AC3, the live 2026-09-18T02:48Z case and the RED-first reproduction.

    `onex-api` carries an omninode_infra revision that is not, and can never be,
    the omnibase_infra ref this deploy built. Against the pre-fix script this
    exits 1 with `is NOT the intended ref 25e15ca25b71`; it must pass.
    """
    result = _run_readback(
        tmp_path,
        services=[FOREIGN],
        running={FOREIGN: "onexapi1"},
        all_containers={FOREIGN: "onexapi1"},
        inspect={"onexapi1": _foreign_container()},
    )
    assert result.returncode == 0, (
        "a service whose image another repo builds must not be compared to this "
        f"repo's ref. stderr={result.stderr!r}"
    )
    assert f"sourced from {FOREIGN_SOURCE_REPO}" in result.stderr
    assert FOREIGN_SHA in result.stderr
    assert "is NOT the intended ref" not in result.stderr


@pytest.mark.unit
def test_lane_built_service_is_still_compared_to_this_repos_ref(
    tmp_path: Path,
) -> None:
    """AC1 bound: the relaxation reaches foreign images only. A lane-built
    service at a STALE ref must still fail -- that is RT-6's whole job."""
    result = _run_readback(
        tmp_path,
        services=[CORE],
        running={CORE: "core1"},
        all_containers={CORE: "core1"},
        inspect={"core1": _own_container(revision="deadbeef1234")},
    )
    assert result.returncode != 0
    assert "is NOT the intended ref" in result.stderr


@pytest.mark.unit
def test_lane_built_service_at_the_intended_ref_passes(tmp_path: Path) -> None:
    """AC1 control: the unchanged happy path for an image this repo builds."""
    result = _run_readback(
        tmp_path,
        services=[CORE],
        running={CORE: "core1"},
        all_containers={CORE: "core1"},
        inspect={"core1": _own_container()},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert f"{CORE} (core1) revision == {GIT_SHA}" in result.stderr


@pytest.mark.unit
def test_mixed_scope_reproduces_the_incident_scope(tmp_path: Path) -> None:
    """AC1: the 2026-09-18 scope in miniature -- lane-built services at the
    intended ref alongside one foreign-sourced service. The whole run passes."""
    result = _run_readback(
        tmp_path,
        services=[CORE, FOREIGN],
        running={CORE: "core1", FOREIGN: "onexapi1"},
        all_containers={CORE: "core1", FOREIGN: "onexapi1"},
        inspect={"core1": _own_container(), "onexapi1": _foreign_container()},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert f"{CORE} (core1) revision == {GIT_SHA}" in result.stderr
    assert f"sourced from {FOREIGN_SOURCE_REPO}" in result.stderr


# ─── AC1 bound: a foreign service is not exempt, only differently asserted ────


@pytest.mark.unit
def test_foreign_sourced_service_with_no_revision_label_is_rejected(
    tmp_path: Path,
) -> None:
    """An absent label is UNKNOWN, never unchanged. Images built before
    OMN-18113 carry no OCI labels at all; reading that as "nothing to compare,
    carry on" is the false-green shape OMN-18200 removed."""
    container = _foreign_container()
    container.pop("revision")
    result = _run_readback(
        tmp_path,
        services=[FOREIGN],
        running={FOREIGN: "onexapi1"},
        all_containers={FOREIGN: "onexapi1"},
        inspect={"onexapi1": container},
    )
    assert result.returncode != 0
    assert "NO org.opencontainers.image.revision label" in result.stderr
    assert "UNKNOWN, never unchanged" in result.stderr


@pytest.mark.unit
def test_declared_foreign_ref_is_asserted_when_it_matches(tmp_path: Path) -> None:
    """A caller that KNOWS the owning repo's ref can declare it, and the
    foreign service is then held to it through the same comparator."""
    result = _run_readback(
        tmp_path,
        services=[FOREIGN],
        running={FOREIGN: "onexapi1"},
        all_containers={FOREIGN: "onexapi1"},
        inspect={"onexapi1": _foreign_container()},
        expected_revisions=f"{FOREIGN}={FOREIGN_SHA}",
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert f"revision == {FOREIGN_SHA} ({FOREIGN_SOURCE_REPO}" in result.stderr


@pytest.mark.unit
def test_declared_foreign_ref_fails_a_stale_foreign_image(tmp_path: Path) -> None:
    """The declared-ref arm is a real assertion, not a formality: a foreign
    service pinned to a stale image still fails the deploy."""
    result = _run_readback(
        tmp_path,
        services=[FOREIGN],
        running={FOREIGN: "onexapi1"},
        all_containers={FOREIGN: "onexapi1"},
        inspect={"onexapi1": _foreign_container(revision="0000000000000000")},
        expected_revisions=f"{FOREIGN}={FOREIGN_SHA}",
    )
    assert result.returncode != 0
    assert "is not at the declared" in result.stderr


@pytest.mark.unit
def test_oci_source_url_naming_this_repo_is_treated_as_lane_built(
    tmp_path: Path,
) -> None:
    """The OCI arm projects a repo URL onto a repo name, so an image stamped
    only by Dockerfile.runtime lands in the same vocabulary as the explicit
    label -- and is therefore held to this repo's ref."""
    container = _own_container(revision="deadbeef1234")
    result = _run_readback(
        tmp_path,
        services=[CORE],
        running={CORE: "core1"},
        all_containers={CORE: "core1"},
        inspect={"core1": container},
    )
    assert result.returncode != 0
    assert "is NOT the intended ref" in result.stderr


@pytest.mark.unit
def test_source_repo_is_never_derived_from_a_service_name_list() -> None:
    """The derivation must stay a label read. A hardcoded service-name list here
    is what goes stale the next time a lane gains a foreign-sourced service --
    the OMN-13826 / OMN-16729 lesson, twice recorded in this file already."""
    fn = _extract_function("resolve_service_source_repo")
    assert "ai.omninode.image.source-repo" in fn
    assert "org.opencontainers.image.source" in fn

    # Comments may name the service that motivated the fix -- the CODE may not.
    code = "\n".join(
        line for line in fn.splitlines() if not line.lstrip().startswith("#")
    )
    for service in (FOREIGN, CORE, "omninode-runtime", "projection-api"):
        assert service not in code, (
            f"resolve_service_source_repo() names {service!r}: the source repo "
            "is read off the container's labels, never matched against a list "
            "of service names, which is what goes stale."
        )


# ─── AC2: the rollback must not retag on a readback-only failure ─────────────


def _run_restore(
    tmp_path: Path, *, health_passed: bool
) -> subprocess.CompletedProcess[str]:
    stub_dir = _stub_dir(tmp_path)
    snapshot = tmp_path / "latest-tags.tsv"
    snapshot.write_text(
        "omninode-runtime\tsha256:94f6885b1b1e\nruntime-effects\tsha256:896c48c03760\n",
        encoding="utf-8",
    )

    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_info() { printf 'INFO: %s\\n' \"$*\" >&2; }",
            "log_warn() { printf 'WARN: %s\\n' \"$*\" >&2; }",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            f'LATEST_TAG_SNAPSHOT_FILE="{snapshot}"',
            'DEPLOY_COMPOSE_PROJECT="omnibase-infra"',
            f"HEALTH_PROBES_PASSED={'true' if health_passed else 'false'}",
            _extract_function("restore_latest_image_tags"),
            "restore_latest_image_tags",
        ]
    )
    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    env["DOCKER_STUB_DIR"] = str(stub_dir)
    result = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=False, env=env
    )
    result.retags = (stub_dir / "retags.log").read_text(encoding="utf-8")  # type: ignore[attr-defined]
    return result


@pytest.mark.unit
def test_no_retag_when_every_health_probe_passed(tmp_path: Path) -> None:
    """AC2, and the falsifier named on the ticket: a run whose RT-6 readback
    failed while every health probe stayed green must retag NOTHING.

    On 2026-09-18T02:48Z this path walked
    `omnibase-infra-omninode-runtime:latest` backwards to a pre-build id on a
    lane whose containers were all running the new build.
    """
    result = _run_restore(tmp_path, health_passed=True)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert result.retags == "", (  # type: ignore[attr-defined]
        "a healthy lane must not have its :latest tags rolled backwards. "
        f"retags={result.retags!r}"  # type: ignore[attr-defined]
    )
    assert "NOT restoring :latest image tags" in result.stderr
    assert "cannot recall them" in result.stderr


@pytest.mark.unit
def test_retag_still_happens_when_health_failed(tmp_path: Path) -> None:
    """AC2 bound: the rollback is REMOVED from the readback-only case, not from
    the case it was built for. A lane that never came up healthy still has its
    pre-build tags restored (OMN-15352 F3)."""
    result = _run_restore(tmp_path, health_passed=False)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "omnibase-infra-omninode-runtime:latest" in result.retags  # type: ignore[attr-defined]
    assert "omnibase-infra-runtime-effects:latest" in result.retags  # type: ignore[attr-defined]
    assert "Restored" in result.stderr


@pytest.mark.unit
def test_health_verdict_is_recorded_by_the_health_probe_itself() -> None:
    """The two halves must not drift: the flag the rollback reads has to be set
    by the branch that proved the lane healthy, not by an unrelated caller."""
    fn = _extract_function("verify_deployment")
    assert "HEALTH_PROBES_PASSED=true" in fn, (
        "verify_deployment() must record the health verdict the rollback guard "
        "reads; otherwise a readback-only failure retags again."
    )
    body = _script_text()
    assert body.count("HEALTH_PROBES_PASSED=true") == 1, (
        "exactly one place may declare the lane healthy"
    )


# ─── The non-mutating readback mode ──────────────────────────────────────────


@pytest.mark.unit
def test_readback_only_mode_returns_before_any_mutating_phase() -> None:
    """The lab proof for this change is a readback run against the live dev
    lane, so the mode has to be non-mutating BY CONSTRUCTION -- it must return
    above the lane lock, the sync, the build and the recreate, not merely
    promise not to call them."""
    body = _script_text()
    assert "--readback-only" in body

    guard = body.index('if [[ "${READBACK_ONLY}" == true ]]; then')
    for mutator in (
        "lane_lock_acquire",
        "sync_files ",
        "build_images ",
        "restart_services ",
        "write_registry ",
        "snapshot_latest_image_tags",
    ):
        idx = body.index(mutator, guard)
        assert idx > guard, f"{mutator} must be called below the readback-only return"
