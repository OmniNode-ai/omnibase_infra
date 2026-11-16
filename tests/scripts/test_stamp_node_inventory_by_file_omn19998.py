# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The node inventory must reach the build by FILE, not by argv [OMN-19998].

Failure, measured 2026-11-14: the pass-2 inventory stamp handed the image's
node inventory to the build as ``--build-arg NODE_INVENTORY=<value>``. The
inventory had grown to ~222 KB, and Linux caps a single argument (or
environment value) at MAX_ARG_STRLEN -- 128 KiB -- so the exec of that build
died with "Argument list too long" before docker ever ran. The stamping pass,
and therefore every governed deploy, failed on size alone.

The fix routes the inventory through a generated compose override instead: a
named build context ``node_inventory`` (a DIRECTORY holding
node_inventory.json) plus the ``com.omninode.node_inventory`` build label.
The value is read from the file by jq (--rawfile); it never appears in any
argv element. The Dockerfile's in-image guard reads the same file from the
bind mount of that context.

Same seam-level harness as ``test_deploy_runtime_rt6_one_shot_omn16729.py``:
the real ``stamp_node_inventory()`` is extracted from
``src/omnibase_infra/handlers/handler_runtime_deploy.sh`` and executed under
bash with only ``docker`` stubbed. The stub is a real executable, so it is the
kernel itself that enforces the argument limit on every call it receives.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = (
    REPO_ROOT / "src" / "omnibase_infra" / "handlers" / "handler_runtime_deploy.sh"
)

MAX_ARG_STRLEN = 131072  # Linux caps ONE argument or env value at 32 pages.
SERVICE = "runtime-effects"


def _script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}\s*\(\)\s*\{{.*?\n\}}",
        _script_text(),
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"could not extract {name}() from the deploy script"
    return match.group(0)


def _make_inventory() -> str:
    """~3800 entries x ~59 bytes ~= 222 KB: the size measured on 2026-11-14,
    well past MAX_ARG_STRLEN."""
    inventory = json.dumps(
        {
            "nodes": [
                {"name": f"node-{i:04d}", "effect": "omnibase.infra.runtime"}
                for i in range(3800)
            ]
        }
    )
    assert len(inventory) > MAX_ARG_STRLEN, len(inventory)
    return inventory


def _write_docker_stub(bin_dir: Path) -> Path:
    """A real `docker` executable. `run ... emit` prints the inventory file;
    `compose ... build` copies the override.json named by -f (and the
    node_inventory.json its additional_contexts names) out for the test to
    inspect, and records where the override lived. Each argv element's LENGTH
    is logged, so the test can prove the inventory never rode the argv."""
    stub = bin_dir / "docker"
    stub.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
for arg in "$@"; do
    printf 'len %s\n' "${#arg}" >> "${DOCKER_STUB_DIR}/calls.log"
done
printf 'call %s\n' "${1:-}" >> "${DOCKER_STUB_DIR}/calls.log"

if [[ "${1:-}" == "run" ]]; then
    cat "${DOCKER_STUB_DIR}/inventory.txt"
    exit 0
fi

if [[ "${1:-}" == "compose" ]]; then
    override=""
    args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do
        if [[ "${args[$i]}" == "-f" \
            && "${args[$((i + 1))]:-}" == *override.json ]]; then
            override="${args[$((i + 1))]}"
        fi
    done
    if [[ -z "${override}" ]]; then
        exit 1
    fi
    printf '%s\n' "${override}" > "${DOCKER_STUB_DIR}/override_path.txt"
    cp "${override}" "${DOCKER_STUB_DIR}/last_override.json"
    context_dir="$(jq -r \
        '.services | to_entries[0].value.build.additional_contexts.node_inventory' \
        "${DOCKER_STUB_DIR}/last_override.json")"
    cp "${context_dir}/node_inventory.json" \
        "${DOCKER_STUB_DIR}/last_node_inventory.json"
    exit "${COMPOSE_BUILD_EXIT_CODE:-0}"
fi

exit 1
""",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _run_stamp(
    tmp_path: Path,
    *,
    inventory: str,
    services: list[str],
    compose_exit_code: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "calls.log").write_text("", encoding="utf-8")
    (stub_dir / "inventory.txt").write_text(inventory, encoding="utf-8")
    _write_docker_stub(stub_dir)

    scope_literal = " ".join(f'"{s}"' for s in services)
    harness = "\n".join(
        [
            "set -euo pipefail",
            "log_step() { printf 'STEP: %s\\n' \"$*\" >&2; }",
            "log_info() { printf 'INFO: %s\\n' \"$*\" >&2; }",
            "log_error() { printf 'ERR: %s\\n' \"$*\" >&2; }",
            "COMPOSE_PROFILE=runtime",
            # Resolving the image ref is a different seam; here it is a given.
            "resolve_built_runtime_image() { printf 'img:test\\n'; }",
            _extract_function("stamp_node_inventory"),
            "cmd=(docker compose -p omnibase-infra -f compose.yaml build --quiet)",
            f"build_scope=({scope_literal})",
            'compose_args=("-f" "compose.yaml")',
            "rc=0",
            "stamp_node_inventory cmd build_scope compose_args "
            '"omnibase-infra" "60" || rc=$?',
            "printf 'RC=%s\\n' \"${rc}\"",
        ]
    )

    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    env["DOCKER_STUB_DIR"] = str(stub_dir)
    env["COMPOSE_BUILD_EXIT_CODE"] = str(compose_exit_code)
    result = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=False, env=env
    )
    return result, stub_dir


@pytest.mark.unit
def test_inventory_over_128_kib_is_stamped_by_file(tmp_path: Path) -> None:
    """The live 2026-11-14 case: a ~222 KB inventory reaches the build intact
    -- by file, label and context -- with no argv element anywhere near the
    kernel's per-argument cap."""
    inventory = _make_inventory()
    result, stub_dir = _run_stamp(tmp_path, inventory=inventory, services=[SERVICE])
    assert "RC=0" in result.stdout, f"stderr={result.stderr!r}"
    override = json.loads((stub_dir / "last_override.json").read_text(encoding="utf-8"))
    stamped = override["services"][SERVICE]["build"]
    assert stamped["labels"]["com.omninode.node_inventory"] == inventory
    assert stamped["additional_contexts"]["node_inventory"]
    stamped_file = (stub_dir / "last_node_inventory.json").read_text(encoding="utf-8")
    assert stamped_file == inventory + "\n"

    calls = (stub_dir / "calls.log").read_text(encoding="utf-8").splitlines()
    lengths = [int(line.split()[1]) for line in calls if line.startswith("len ")]
    assert lengths
    assert max(lengths) <= MAX_ARG_STRLEN, "the inventory rode an argv element"


@pytest.mark.unit
def test_the_old_argument_route_fails_at_the_kernel(tmp_path: Path) -> None:
    """The failing control: the same inventory as one --build-arg value cannot
    even exec. The kernel refuses it with E2BIG before docker runs, which is
    exactly the 2026-11-14 failure the by-file route retires."""
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    stub = _write_docker_stub(stub_dir)
    inventory = _make_inventory()
    argv = [str(stub), "compose", "build", "--build-arg", "NODE_INVENTORY=" + inventory]
    with pytest.raises(OSError) as excinfo:
        subprocess.run(argv, check=False)
    assert excinfo.value.errno == errno.E2BIG


@pytest.mark.unit
def test_a_failed_build_fails_the_stamp(tmp_path: Path) -> None:
    """FAIL-CLOSED: a build that exits non-zero fails the stamp, and the
    scratch directory holding the inventory is cleaned up."""
    result, stub_dir = _run_stamp(
        tmp_path,
        inventory=_make_inventory(),
        services=[SERVICE],
        compose_exit_code=3,
    )
    assert "RC=1" in result.stdout, f"stderr={result.stderr!r}"
    assert "stamping pass failed (exit 3;" in result.stderr
    override_txt = (stub_dir / "override_path.txt").read_text(encoding="utf-8").strip()
    override_path = Path(override_txt)
    assert override_path.name == "override.json"
    assert not override_path.exists(), "the scratch dir holding it must be gone"


@pytest.mark.unit
def test_an_empty_inventory_is_refused(tmp_path: Path) -> None:
    """An empty inventory would stamp a label asserting the image ships no
    nodes; it is refused before any build is attempted."""
    result, stub_dir = _run_stamp(tmp_path, inventory="", services=[SERVICE])
    assert "RC=1" in result.stdout, f"stderr={result.stderr!r}"
    assert "EMPTY node inventory" in result.stderr
    calls = (stub_dir / "calls.log").read_text(encoding="utf-8")
    assert "call run" in calls
    assert "call compose" not in calls


@pytest.mark.unit
def test_the_override_names_every_service_in_scope(tmp_path: Path) -> None:
    """The override is a compose file in its own right: every in-scope service
    appears under services, so each build gets the stamped context."""
    scope = ["runtime-effects", "runtime-registry"]
    inventory = json.dumps({"nodes": [{"name": "node-0"}]})
    result, stub_dir = _run_stamp(tmp_path, inventory=inventory, services=scope)
    assert "RC=0" in result.stdout, f"stderr={result.stderr!r}"
    override = json.loads((stub_dir / "last_override.json").read_text(encoding="utf-8"))
    assert set(override["services"]) == set(scope)


@pytest.mark.unit
def test_the_dockerfile_guard_reads_the_inventory_from_the_mounted_file(
    tmp_path: Path,
) -> None:
    """The in-image guard, run as the Dockerfile runs it: the file's content is compared with this
    environment's own discovery pass in process, so no argument carries it. A matching file passes;
    one missing a node is refused naming it."""
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.runtime").read_text(
        encoding="utf-8"
    )
    match = re.search(r"python -c '(import sys;[^']*)'", dockerfile)
    assert match is not None, "the Dockerfile guard must verify from the mounted file"
    emitted = subprocess.run(
        [sys.executable, "-m", "omnibase_infra.runtime.node_inventory", "emit"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    stamped = tmp_path / "node_inventory.json"
    stamped.write_text(emitted + "\n", encoding="utf-8")
    passed = subprocess.run(
        [sys.executable, "-c", match.group(1), str(stamped)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert passed.returncode == 0, passed.stderr
    document = json.loads(emitted)
    key = next(k for k, v in document.items() if isinstance(v, list) and v)
    dropped = document[key].pop()
    stamped.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    refused = subprocess.run(
        [sys.executable, "-c", match.group(1), str(stamped)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 1
    assert dropped["name"] in refused.stderr
