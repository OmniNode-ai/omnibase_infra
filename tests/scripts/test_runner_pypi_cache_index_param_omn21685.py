# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The runner PyPI cache index is a parameter, never a host in this repository (OMN-21685).

The devpi cache moved between lab hosts. The public-repo hygiene gate refuses a
lab host in a shipped file, so the index URL reaches a runner only as the
``OMNI_PYPI_CACHE_INDEX`` container variable, passed through from the runner
host's ``docker/.env`` (set from the private lab overlay). These tests hold the
three halves of that contract:

  1. the compose file passes the variable through with an empty default;
  2. the job-started hook wires nothing, and says so, when the variable is
     unset, and wires the declared index when it is set and answers;
  3. neither the hook nor the canary override names a cache host.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-job-started.sh"
RUNNERS_COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
CANARY_COMPOSE = REPO_ROOT / "docker" / "docker-compose.pypi-canary.yml"
_MAIN_BODY_MARKER = "\nwire_pypi_cache || true\n"

pytestmark = [pytest.mark.unit]


class _HeadOk(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        return


@pytest.fixture
def cache_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HeadOk)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/root/pypi/+simple/"
    finally:
        server.shutdown()
        server.server_close()


def _run_wire_pypi_cache(
    tmp_path: Path, env_overrides: dict[str, str]
) -> tuple[str, str]:
    """Runs wire_pypi_cache from the real hook and returns (GITHUB_ENV text, audit text)."""
    text = HOOK_SCRIPT.read_text()
    idx = text.index(_MAIN_BODY_MARKER)
    driver = tmp_path / "driver.sh"
    driver.write_text(text[:idx] + "\nwire_pypi_cache\n")
    github_env = tmp_path / "github_env"
    github_env.write_text("")
    audit = tmp_path / "audit.log"
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "RUNNER_NAME": "omninode-runner-test",
        "GITHUB_ENV": str(github_env),
        "OMNI_PYPI_CACHE_AUDIT": str(audit),
        **env_overrides,
    }
    result = subprocess.run(
        ["bash", str(driver)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return github_env.read_text(), audit.read_text()


def test_compose_passes_the_index_through_with_an_empty_default() -> None:
    lines = [
        line.strip()
        for line in RUNNERS_COMPOSE.read_text().splitlines()
        if line.strip().startswith("OMNI_PYPI_CACHE_INDEX:")
    ]
    assert lines == ["OMNI_PYPI_CACHE_INDEX: ${OMNI_PYPI_CACHE_INDEX:-}"]


def test_hook_with_no_declared_index_wires_nothing_and_says_so(
    tmp_path: Path,
) -> None:
    github_env, audit = _run_wire_pypi_cache(tmp_path, {})
    assert github_env == ""
    assert "DEGRADE cache-probe-failed index=<OMNI_PYPI_CACHE_INDEX unset>" in audit


def test_hook_with_an_empty_index_wires_nothing(tmp_path: Path) -> None:
    github_env, audit = _run_wire_pypi_cache(tmp_path, {"OMNI_PYPI_CACHE_INDEX": ""})
    assert github_env == ""
    assert "DEGRADE cache-probe-failed index=<OMNI_PYPI_CACHE_INDEX unset>" in audit


def test_hook_wires_the_declared_index_when_it_answers(
    tmp_path: Path, cache_server: str
) -> None:
    github_env, audit = _run_wire_pypi_cache(
        tmp_path, {"OMNI_PYPI_CACHE_INDEX": cache_server}
    )
    assert f"UV_INDEX={cache_server}\n" in github_env
    assert f"PIP_INDEX_URL={cache_server}\n" in github_env
    assert "UV_DEFAULT_INDEX" not in github_env
    assert f"WIRED index={cache_server}" in audit


def test_hook_degrades_open_when_the_declared_index_does_not_answer(
    tmp_path: Path,
) -> None:
    dead = "http://127.0.0.1:9/root/pypi/+simple/"
    github_env, audit = _run_wire_pypi_cache(tmp_path, {"OMNI_PYPI_CACHE_INDEX": dead})
    assert github_env == ""
    assert "DEGRADE cache-probe-failed" in audit


@pytest.mark.parametrize("path", [HOOK_SCRIPT, CANARY_COMPOSE])
def test_no_cache_host_is_named_in_the_hook_or_the_canary_override(
    path: Path,
) -> None:
    text = path.read_text()
    assert ":3141" not in text, f"{path.name} names a cache endpoint"
    assert "tail75df5e" not in text, f"{path.name} names a tailnet host"
