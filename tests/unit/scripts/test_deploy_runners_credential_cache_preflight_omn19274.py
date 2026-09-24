# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Execution-level positive/negative control for the OMN-19274 credential-cache
pre-flight in ``--rolling`` mode.

OMN-18415's test file pins the rest of ``--rolling`` with source-level string
assertions (deliberately -- see its own docstring), because exercising the
real script means real ssh/docker against the live fleet. That approach is
structurally blind to the exact bug class this ticket fixes: rolling
OMN-19206's runner image onto .201 force-recreated ``omninode-runner-3`` and
``omninode-runner-4`` straight into "No credentials found and RUNNER_TOKEN is
not set" even though both runners' per-runner named volumes held intact,
correctly-owned credential files -- the volume held a directory keyed to the
OLD ``RUNNER_LABELS`` value, but the freshly rendered compose config (which a
force-recreate always picks up) computed a NEW key, and no directory existed
under it. A string test asserting "the halt message exists" cannot catch a
live hash mismatch; only running the real key computation against a real
volume can.

This test runs ``_runner_cache_key_for_service`` and ``_runner_cache_ready``
-- the two new helpers ``roll_one_runner`` calls before ever force-recreating
a runner -- for real: real ``sha256sum``, a real local docker named volume,
real directory-existence checks. A fake ``ssh`` on ``PATH`` runs the "remote"
command locally (both helpers shell out via ``ssh "${RUNNER_HOST}" "..."``),
and a ``docker`` shim intercepts only ``compose ... config <service>`` so the
test does not depend on the live fleet's compose files or its 792-line
service list, passing every other docker invocation through to the real
local docker daemon.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runners.sh"

FAKE_LABELS = "fake-label-a,fake-label-b"
FAKE_ORG = "https://github.com/FakeOrg"
EXPECTED_KEY = hashlib.sha256(f"{FAKE_LABELS}:{FAKE_ORG}".encode()).hexdigest()

REAL_DOCKER = shutil.which("docker")
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(REAL_DOCKER is None, reason="docker is not on PATH"),
]


def _extract_function(script_text: str, name: str) -> str:
    start_marker = f"{name}() {{"
    start = script_text.index(start_marker)
    depth = 0
    i = start + len(start_marker) - 1  # at the opening '{'
    end = None
    for i in range(start + len(start_marker) - 1, len(script_text)):
        ch = script_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end is not None, f"unbalanced braces extracting {name}()"
    return script_text[start:end]


@pytest.fixture(scope="module")
def helper_functions() -> str:
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    key_fn = _extract_function(text, "_runner_cache_key_for_service")
    ready_fn = _extract_function(text, "_runner_cache_ready")
    assert "sha256sum" in key_fn
    assert "--entrypoint test" in ready_fn
    return key_fn + "\n" + ready_fn


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    """A PATH-prepended dir with a fake ``ssh`` that runs its remote command
    locally via ``bash -c``, and a ``docker`` shim that fakes only
    ``compose ... config <service>``, delegating everything else (image
    inspection, ``docker run``, volume ops) to the real local docker.
    """
    bindir = tmp_path / "fakebin"
    bindir.mkdir()

    ssh_stub = bindir / "ssh"
    ssh_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "shift  # drop the host argument -- run the remote command locally\n"
        'exec bash -c "$1"\n'
    )
    ssh_stub.chmod(ssh_stub.stat().st_mode | stat.S_IEXEC)

    docker_stub = bindir / "docker"
    docker_stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == "compose" && " $* " == *" config "* ]]; then\n'
        f'  echo "    RUNNER_LABELS: {FAKE_LABELS}"\n'
        f'  echo "    GITHUB_ORG_URL: {FAKE_ORG}"\n'
        "  exit 0\n"
        "fi\n"
        f'exec "{REAL_DOCKER}" "$@"\n'
    )
    docker_stub.chmod(docker_stub.stat().st_mode | stat.S_IEXEC)

    return bindir


@pytest.fixture
def runner_volume() -> Iterator[str]:
    """A real local docker named volume, torn down after the test regardless
    of outcome.
    """
    name = f"omn19274-test-{uuid.uuid4().hex[:12]}-creds"
    subprocess.run(
        ["docker", "volume", "create", name], check=True, capture_output=True
    )
    yield name
    subprocess.run(
        ["docker", "volume", "rm", "-f", name], capture_output=True, check=False
    )


def _run_helpers(
    helper_functions: str, fake_bin: Path, call: str
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["RUNNER_HOST"] = "fake-host.invalid"
    env["RUNNER_HOST_DIR"] = "/home/fake/.omnibase/runners"
    script = f"set -euo pipefail\n{helper_functions}\n{call}\n"
    return subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_key_computed_from_rendered_labels_matches_sha256_of_labels_colon_org(
    helper_functions: str, fake_bin: Path
) -> None:
    """Pins the exact formula entrypoint.sh's own ``_cache_key()`` uses
    (``sha256(RUNNER_LABELS:GITHUB_ORG_URL)``) -- a drift here would silently
    make every pre-flight check compare against the wrong key.
    """
    result = _run_helpers(
        helper_functions,
        fake_bin,
        'name="omninode-runner-99"\n_runner_cache_key_for_service "$name"',
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == EXPECTED_KEY


def test_positive_control_ready_when_the_current_key_directory_exists(
    helper_functions: str, fake_bin: Path, runner_volume: str
) -> None:
    """A runner whose volume DOES hold a directory for the current label
    set's key must read ready (this is the runner-2 case: recreate is safe).
    """
    name = runner_volume.removesuffix("-creds")
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{runner_volume}:/data",
            "alpine:3",
            "mkdir",
            "-p",
            f"/data/{EXPECTED_KEY}",
        ],
        check=True,
        capture_output=True,
    )
    # _runner_cache_ready hardcodes the image tag omninode-runner:latest;
    # point it at a real, tiny local image for the duration of this check.
    subprocess.run(["docker", "tag", "alpine:3", "omninode-runner:latest"], check=True)
    try:
        result = _run_helpers(
            helper_functions,
            fake_bin,
            f'_runner_cache_ready "{name}" "{EXPECTED_KEY}"',
        )
    finally:
        subprocess.run(
            ["docker", "rmi", "omninode-runner:latest"],
            capture_output=True,
            check=False,
        )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_negative_control_not_ready_when_only_the_old_key_directory_exists(
    helper_functions: str, fake_bin: Path, runner_volume: str
) -> None:
    """The exact live failure this ticket fixes: the volume holds credentials,
    just under the OLD label set's key, not the one the current compose
    config would render. Must read NOT ready -- this is what makes
    ``roll_one_runner`` skip instead of force-recreating into an outage.
    """
    old_key = hashlib.sha256(b"old-label-set:https://github.com/FakeOrg").hexdigest()
    assert old_key != EXPECTED_KEY
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{runner_volume}:/data",
            "alpine:3",
            "sh",
            "-c",
            f"mkdir -p /data/{old_key} && touch /data/{old_key}/.credentials /data/{old_key}/.runner",
        ],
        check=True,
        capture_output=True,
    )
    name = runner_volume.removesuffix("-creds")
    subprocess.run(["docker", "tag", "alpine:3", "omninode-runner:latest"], check=True)
    try:
        result = _run_helpers(
            helper_functions,
            fake_bin,
            f'_runner_cache_ready "{name}" "{EXPECTED_KEY}"',
        )
    finally:
        subprocess.run(
            ["docker", "rmi", "omninode-runner:latest"],
            capture_output=True,
            check=False,
        )
    assert result.returncode != 0, (
        "a volume holding only the OLD key's credentials must not read ready "
        "for the NEW key -- this exact gap stranded omninode-runner-3/4 "
        "offline on 2026-09-23"
    )


# --- control-flow wiring (source-level, same method as OMN-18415's suite) ---


@pytest.fixture(scope="module")
def script_text() -> str:
    return DEPLOY_SCRIPT.read_text(encoding="utf-8")


def test_roll_one_runner_skips_rather_than_recreates_when_cache_is_not_ready(
    script_text: str,
) -> None:
    """The fix's core behaviour change: a runner with no matching cache entry
    and no migration token must be skipped (rc 3, never force-recreated),
    not left to force-recreate into the OMN-19206 outage.
    """
    start = script_text.index("roll_one_runner() {")
    end = script_text.index("rolling_deploy() {")
    body = script_text[start:end]
    assert "_runner_cache_ready" in body
    assert "return 3" in body
    skip_branch = body[body.index('if [[ -z "${TOKEN_FILE}" ]]') :][:400]
    assert "SKIPPING" in skip_branch
    assert "return 3" in skip_branch


def test_the_migration_path_reads_the_token_from_a_file_not_argv_or_env(
    script_text: str,
) -> None:
    assert "--token-file=*) TOKEN_FILE=" in script_text
    start = script_text.index('mig_token=$(<"${TOKEN_FILE}")')
    # The token must never be interpolated bare into a logged/echoed line.
    surrounding = script_text[start - 200 : start + 400]
    assert "base64" in surrounding, (
        "the migration token must be base64-encoded before crossing the ssh "
        "boundary, matching the existing default-path convention"
    )


def test_rolling_deploy_reports_skipped_runners_separately_from_a_halt(
    script_text: str,
) -> None:
    start = script_text.index("rolling_deploy() {")
    body = script_text[start:]
    assert "skip_migration" in body
    assert "3) skip_migration+=" in body
    assert "Skipped (no credential-cache entry" in body


def test_build_runner_image_preserves_the_previous_image_before_overwriting_it() -> (
    None
):
    build_script = (REPO_ROOT / "scripts" / "ci" / "build_runner_image.sh").read_text(
        encoding="utf-8"
    )
    assert 'docker tag "${IMAGE_TAG}" "${ROLLBACK_TAG}"' in build_script
    # The keep-list already protects this exact tag -- OMN-19274 wires the
    # build to actually use it, rather than adding a new keep-list entry.
    keep_list = (REPO_ROOT / "deploy" / "disk-gc" / "keep-list.yaml").read_text(
        encoding="utf-8"
    )
    assert "- rollback" in keep_list
