# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime entrypoint must launch exactly ONE Python interpreter (OMN-17372).

RED on the parent commit: ``docker/entrypoint-runtime.sh`` launched FOUR separate
cold interpreters before the kernel ever started — the schema-fingerprint stamp,
the Bifrost render, the secret-resolver render and the
``DELEGATION_ROUTING_TIERS_PATH`` re-derivation probe — and then ``exec``'d the
kernel as a fifth. Each paid a full cold ``import omnibase_infra``. Measured on
the onex-dev runtime container (2026-09-06, container start 21:34:27Z):
141.3 s + 51.9 s + 50.2 s + 48.6 s = **292.0 s** of the 551 s that elapsed before
the first subscription was even attempted.

The assertion here is a **count**, not a duration: a duration would flake on a
loaded host, while the interpreter-launch count is deterministic and is the
thing that actually scales with how cold the CPU allocation is.

Related Tickets:
    - OMN-17372: runtime boot latency.
    - OMN-13666: required vs best-effort stamp policy (behaviour preserved).
    - OMN-15628: DELEGATION_ROUTING_TIERS_PATH self-heal (behaviour preserved).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]

ENTRYPOINT = DOCKER_DIR / "entrypoint-runtime.sh"

#: A shell word that starts a Python interpreter. Matches ``python``,
#: ``python3``, ``python3.12`` and an absolute path to one, at a command
#: position: the start of a line (indentation allowed -- the parent's launches
#: sat inside a shell function, so anchoring at column 0 would have under-counted
#: them to zero), after ``exec``, after ``;``, after a pipe, or inside ``$(``.
_PYTHON_LAUNCH_RE = re.compile(
    r"(?:^[ \t]*|\|\s*|\$\(\s*|;\s*|\bexec\s+)(?:[\w./-]*/)?python[\d.]*\s",
    re.MULTILINE,
)


def _strip_comments(source: str) -> str:
    """Drop full-line and trailing ``#`` comments so prose is not counted.

    Deliberately naive about ``#`` inside quotes: the entrypoint has none, and a
    counter that over-counts would fail this test loudly rather than silently
    pass it.
    """
    lines: list[str] = []
    for raw in source.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("#"):
            continue
        lines.append(raw.split(" #", 1)[0])
    return "\n".join(lines)


def count_python_invocations(source: str) -> int:
    """Count Python interpreter launches in a shell script's executable lines."""
    return len(_PYTHON_LAUNCH_RE.findall(_strip_comments(source)))


# ---------------------------------------------------------------------------
# Positive control — the counter actually counts
# ---------------------------------------------------------------------------


def test_control_counter_finds_the_parent_shape() -> None:
    """POSITIVE CONTROL: the parent commit's four-interpreter shape counts as 4.

    Without this, a counter that silently matched nothing would report ``1``…
    by reporting ``0``, and a regex typo would read as a passing test below.
    """
    parent_shape = """#!/bin/sh
# a comment mentioning python -m something that must not be counted
stamp_fingerprint() {
  while [ "${ATTEMPT}" -le "${MAX_ATTEMPTS}" ]; do
    python -m omnibase_infra.runtime.util_schema_fingerprint \\
      --manifest "${MANIFEST_NAME}" --db-url "${DB_URL}" stamp || RC=$?
  done
}
python -m omnibase_infra.runtime.render_bifrost_delegation_contract
  python -m omnibase_infra.runtime.render_secret_resolver_config
RESOLVED=$(python -c "import omnimarket; print(1)")
echo "this line mentions python -m inline and must not count"
exec "$@"
"""
    assert count_python_invocations(parent_shape) == 4


def test_control_counter_ignores_prose() -> None:
    """POSITIVE CONTROL: comments naming the old commands are not launches."""
    prose_only = """#!/bin/sh
# python -m omnibase_infra.runtime.util_schema_fingerprint
# python -m omnibase_infra.runtime.render_secret_resolver_config
echo "hello"
"""
    assert count_python_invocations(prose_only) == 0


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


def test_entrypoint_launches_exactly_one_python_interpreter() -> None:
    """RED on parent at 4; GREEN at 1."""
    count = count_python_invocations(ENTRYPOINT.read_text())
    assert count == 1, (
        f"docker/entrypoint-runtime.sh launches {count} Python interpreters; "
        "every cold start pays a full `import omnibase_infra` before the kernel "
        "runs (OMN-17372)."
    )


def test_the_single_launch_is_the_boot_preflight_and_it_execs() -> None:
    """That one interpreter is the preflight, and it replaces the shell."""
    source = ENTRYPOINT.read_text()
    assert (
        'exec python -m omnibase_infra.runtime.entrypoint_preflight "$@"' in source
    ), "the preflight must be exec'd so the Python process stays tini's direct child"


def test_no_module_is_launched_out_of_process_any_more() -> None:
    """The four old out-of-process launches are gone from the executable lines."""
    executable = _strip_comments(ENTRYPOINT.read_text())
    for module in (
        "omnibase_infra.runtime.util_schema_fingerprint",
        "omnibase_infra.runtime.render_bifrost_delegation_contract",
        "omnibase_infra.runtime.render_secret_resolver_config",
    ):
        assert module not in executable, f"{module} is still launched out of process"
    assert "python -c" not in executable


# ---------------------------------------------------------------------------
# Nothing the shell still owns was lost
# ---------------------------------------------------------------------------


def test_root_volume_bootstrap_and_privilege_drop_stay_in_the_shell() -> None:
    """The gosu drop runs before the unprivileged user exists — it stays shell."""
    source = ENTRYPOINT.read_text()
    assert 'if [ "$(id -u)" -eq 0 ]; then' in source
    assert "install -d -o omniinfra -g omniinfra" in source
    assert 'exec gosu omniinfra "$0" "$@"' in source


def test_deployment_identity_banner_stays_in_the_shell() -> None:
    """Pure echo — printing it needs no interpreter and must stay first."""
    source = ENTRYPOINT.read_text()
    assert 'echo "=== OmniNode Runtime ==="' in source
    assert 'echo "RUNTIME_SOURCE_HASH=${RUNTIME_SOURCE_HASH:-unknown}"' in source
    banner_pos = source.index('echo "=== OmniNode Runtime ==="')
    preflight_pos = source.index("entrypoint_preflight")
    assert banner_pos < preflight_pos


def test_shellcheck_clean_if_available() -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck not installed")
    result = subprocess.run(
        [shellcheck, str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dockerfile_still_routes_cmd_through_the_entrypoint() -> None:
    """The preflight only helps if it is still what tini spawns."""
    dockerfile: Path = DOCKER_DIR / "Dockerfile.runtime"
    source = dockerfile.read_text()
    assert 'ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint-runtime.sh"]' in source
    assert 'CMD ["onex-runtime"]' in source


__all__: list[str] = []
