# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RT-6 deploy readback must verify exactly the rebuilt service scope [OMN-15348].

Defect: `readback_deployed_ref()` in `scripts/deploy-runtime.sh` was hardcoded to
probe only the `omninode-runtime` container's image-revision label, regardless of
`RUNTIME_BUILD_SERVICES_OVERRIDE` (the OMN-14873 scoped-rebuild override). Observed
live 2026-07-28T23:08-23:17Z on the .201 dev lane: a correctly scoped rebuild
(`RUNTIME_BUILD_SERVICES_OVERRIDE=runtime-effects`) rebuilt + recreated only
`omninode-runtime-effects`. RT-6 then compared the UNTOUCHED `omninode-runtime`
container's stale label against the new build's `VCS_REF`, false-FAILed, and
auto-triggered `restore-previous-deployment` -- reverting `docker-compose.infra.yml`
and `registry.json` on disk while the freshly-recreated `runtime-effects` container
stayed live.

The fix loops the readback over `RUNTIME_BUILD_SERVICES` (the array
`deploy-runtime.sh` already resolves from `RUNTIME_BUILD_SERVICES_OVERRIDE`, or the
full `RUNTIME_SERVICES` set when unset) instead of a single hardcoded container, so
an out-of-scope container's stale label is never probed -- it can neither fail the
deploy nor trigger restore.

These tests drive the ACTUAL script seam: `readback_deployed_ref()` is extracted
(with its real dependencies `resolve_lane_runtime_container_name`,
`resolve_lane_overlay_filename`, `resolve_compose_file_args`) and executed under
bash against the REAL `scripts/verify_deployed_versions.py`, with only `docker`
stubbed on PATH (no daemon required). This is not a surrogate: the exact bash
control flow that decided which container(s) to probe, and the real Python readback
script's pass/fail logic, both run for real.
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

GIT_SHA = "abc123def456"
STALE_SHA = "111111111111"
VERSION = "9.9.9"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    """Return the source text of a single top-level bash function ``name()``."""
    text = _script_text()
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, (
        f"could not extract function {name}() from deploy-runtime.sh"
    )
    return match.group(0)


def _write_docker_stub(bin_dir: Path) -> Path:
    """Write a fake `docker` on PATH that answers `compose ps -q`, `inspect`,
    and `exec ... uv pip show` from files under $DOCKER_STUB_DIR, and appends
    every invocation to $DOCKER_STUB_DIR/calls.log for call-scope assertions.
    """
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf "%s\\n" "$*" >> "${DOCKER_STUB_DIR}/calls.log"\n'
        "\n"
        'if [[ "$1" == "compose" ]]; then\n'
        "    shift\n"
        '    args=("$@")\n'
        "    n=${#args[@]}\n"
        "    for ((i = 0; i < n; i++)); do\n"
        '        if [[ "${args[$i]}" == "ps" ]]; then\n'
        '            service="${args[$((n - 1))]}"\n'
        '            map_file="${DOCKER_STUB_DIR}/ps/${service}"\n'
        '            if [[ -f "${map_file}" ]]; then\n'
        '                cat "${map_file}"\n'
        "                exit 0\n"
        "            fi\n"
        "            exit 1\n"
        "        fi\n"
        "    done\n"
        "    exit 1\n"
        "fi\n"
        "\n"
        'if [[ "$1" == "inspect" ]]; then\n'
        '    container="$2"\n'
        '    rev_file="${DOCKER_STUB_DIR}/revision/${container}"\n'
        '    if [[ -f "${rev_file}" ]]; then\n'
        '        cat "${rev_file}"\n'
        "    fi\n"
        "    exit 0\n"
        "fi\n"
        "\n"
        'if [[ "$1" == "exec" ]]; then\n'
        '    container="$2"\n'
        '    package="${*: -1}"\n'
        '    ver_file="${DOCKER_STUB_DIR}/version/${container}"\n'
        '    if [[ -f "${ver_file}" ]]; then\n'
        '        printf "Name: %s\\nVersion: %s\\n" "${package}" "$(cat "${ver_file}")"\n'
        "        exit 0\n"
        "    fi\n"
        "    exit 1\n"
        "fi\n"
        "\n"
        "exit 1\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _run_readback(
    tmp_path: Path,
    *,
    runtime_build_services: list[str],
    ps_map: dict[str, str],
    revision_map: dict[str, str],
    version_map: dict[str, str] | None = None,
    scoped_override: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Extract + execute readback_deployed_ref() with real dependencies, stubbed docker."""
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "ps").mkdir()
    (stub_dir / "revision").mkdir()
    (stub_dir / "version").mkdir()
    (stub_dir / "calls.log").write_text("", encoding="utf-8")
    _write_docker_stub(stub_dir)

    for service, container in ps_map.items():
        (stub_dir / "ps" / service).write_text(container + "\n", encoding="utf-8")
    for container, revision in revision_map.items():
        (stub_dir / "revision" / container).write_text(revision, encoding="utf-8")
    for container, version in (version_map or {}).items():
        (stub_dir / "version" / container).write_text(version, encoding="utf-8")

    services_literal = " ".join(f'"{s}"' for s in runtime_build_services)
    override_line = (
        f'RUNTIME_BUILD_SERVICES_OVERRIDE="{services_literal}"'
        if scoped_override
        else "unset RUNTIME_BUILD_SERVICES_OVERRIDE"
    )

    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_step() { printf 'STEP: %s\\n' \"$*\" >&2; }",
            "log_info() { printf 'INFO: %s\\n' \"$*\" >&2; }",
            "log_warn() { printf 'WARN: %s\\n' \"$*\" >&2; }",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            "log_cmd() { printf 'CMD: %s\\n' \"$*\" >&2; }",
            _extract_function("resolve_lane_overlay_filename"),
            _extract_function("resolve_compose_file_args"),
            _extract_function("resolve_lane_runtime_container_name"),
            (
                "DEV_LANE_ONLY_RUNTIME_SERVICES=("
                "projection-tenant-registry-writer projection-delegation-writer"
                ")"
            ),
            _extract_function("resolve_lane_runtime_services"),
            _extract_function("readback_deployed_ref"),
            override_line,
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

    return subprocess.run(
        ["bash", "-c", harness],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.unit
def test_scoped_run_ignores_out_of_scope_stale_container(tmp_path: Path) -> None:
    """(i) Scoped run: out-of-scope omninode-runtime has a stale label -> RT-6
    passes and never even probes the out-of-scope container (no restore)."""
    result = _run_readback(
        tmp_path,
        runtime_build_services=["runtime-effects"],
        ps_map={"runtime-effects": "omninode-runtime-effects"},
        revision_map={
            "omninode-runtime-effects": GIT_SHA,
            # Deliberately stale/never-referenced: omninode-runtime is NOT in
            # RUNTIME_BUILD_SERVICES for this scoped run.
            "omninode-runtime": STALE_SHA,
        },
        scoped_override=True,
    )
    assert result.returncode == 0, result.stderr
    calls_log = (tmp_path / "stubs" / "calls.log").read_text(encoding="utf-8")
    # Word-boundary match: "omninode-runtime" as its own token must not appear
    # (it would only appear as a prefix of "omninode-runtime-effects" today).
    assert re.search(r"(?<![\w-])omninode-runtime(?![\w-])", calls_log) is None, (
        "the out-of-scope omninode-runtime container must never be probed by a "
        f"scoped run; docker calls were:\n{calls_log}"
    )


@pytest.mark.unit
def test_scoped_run_fails_and_restores_on_in_scope_mismatch(tmp_path: Path) -> None:
    """(ii) Scoped run: the in-scope container genuinely fails readback -> RT-6
    fails (exit 1), which is the same exit code that fires deploy-runtime.sh's
    auto-restore trap."""
    result = _run_readback(
        tmp_path,
        runtime_build_services=["runtime-effects"],
        ps_map={"runtime-effects": "omninode-runtime-effects"},
        revision_map={"omninode-runtime-effects": STALE_SHA},
        scoped_override=True,
    )
    assert result.returncode == 1, (
        f"expected RT-6 to fail-closed on a genuine in-scope mismatch; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "runtime-effects" in result.stderr


@pytest.mark.unit
def test_unscoped_run_verifies_full_default_service_set(tmp_path: Path) -> None:
    """(iii) Unscoped run (no override): every RUNTIME_SERVICES member is
    verified, unchanged from the pre-fix single-container behavior for
    omninode-runtime, plus every sibling service now also covered."""
    full_services = [
        "omninode-runtime",
        "runtime-effects",
        "runtime-worker",
        "projection-api",
        "agent-actions-consumer",
        "skill-lifecycle-consumer",
        "intelligence-api",
        "omninode-contract-resolver",
        "projection-tenant-registry-writer",
        "projection-delegation-writer",
    ]
    ps_map = {
        "runtime-effects": "omninode-runtime-effects",
        "runtime-worker": "omninode-runtime-worker",
        "projection-api": "omnimarket-projection-api",
        "agent-actions-consumer": "omninode-agent-actions-consumer",
        "skill-lifecycle-consumer": "omninode-skill-lifecycle-consumer",
        "intelligence-api": "omnibase-intelligence-api",
        "omninode-contract-resolver": "omninode-contract-resolver",
        "projection-tenant-registry-writer": "projection-tenant-registry-writer",
        "projection-delegation-writer": "projection-delegation-writer",
    }
    revision_map = dict.fromkeys(ps_map.values(), GIT_SHA)
    # omninode-runtime is resolved via resolve_lane_runtime_container_name, not
    # the ps map, for the dev project used here ("omnibase-infra").
    revision_map["omninode-runtime"] = GIT_SHA
    version_map = {"omninode-runtime": VERSION}

    result = _run_readback(
        tmp_path,
        runtime_build_services=full_services,
        ps_map=ps_map,
        revision_map=revision_map,
        version_map=version_map,
    )
    assert result.returncode == 0, result.stderr
    calls_log = (tmp_path / "stubs" / "calls.log").read_text(encoding="utf-8")
    for service in ps_map:
        assert service in calls_log, (
            f"unscoped run must probe every default RUNTIME_SERVICES member, "
            f"missing '{service}' in docker calls:\n{calls_log}"
        )
    # The package-version check is still asserted for the primary container.
    assert "uv pip show omnibase-infra" in calls_log
