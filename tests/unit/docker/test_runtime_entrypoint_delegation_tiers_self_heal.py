# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-15628 remediation: entrypoint self-heal for a stale DELEGATION_ROUTING_TIERS_PATH.

The k8s manifests (omninode_infra: deployment-omninode-runtime{,-effects,-worker}.yaml)
pin ``DELEGATION_ROUTING_TIERS_PATH`` as a literal string embedding the venv's Python
minor version, e.g.::

    /app/.venv/lib/python3.12/site-packages/omnimarket/configs/routing_tiers.yaml

A base-image Python version bump silently invalidates that literal path with no signal
until the routing reducer fails closed at first use. ``docker/entrypoint-runtime.sh``
now self-heals: when the pinned path does not exist on disk, it re-derives the path from
the installed ``omnimarket`` package's OWN location (which always matches whatever
Python actually ships in the image) and exports the corrected value before exec'ing the
kernel. If re-derivation also fails, the original (possibly-stale) value is left
untouched so the routing reducer still fails closed attributably (CLAUDE.md rule 8) --
this is a best-effort correction, never a silent fallback that manufactures a config the
reducer would otherwise refuse to load.

The behavioral tests execute the real ``docker/entrypoint-runtime.sh`` with a stubbed
``python`` on PATH (no Docker, Postgres, or privilege drop involved), mirroring the
harness in ``test_runtime_entrypoint_stamp_tolerance.py``.
"""

from __future__ import annotations

import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]

ENTRYPOINT = DOCKER_DIR / "entrypoint-runtime.sh"

# Stub "python" that:
#   * exits 0 for any `--manifest ... stamp` invocation (schema-fingerprint stamp,
#     already covered by test_runtime_entrypoint_stamp_tolerance.py -- not under test
#     here, so it always succeeds so boot reaches the delegation-tiers block).
#   * for a `-c <code>` invocation (the re-derivation probe), prints
#     $STUB_RESOLVED_TIERS_PATH and exits $STUB_C_RC (default 0) -- emulates a
#     successful `import omnimarket` re-derivation when STUB_RESOLVED_TIERS_PATH is a
#     real file, or a failed one when it is unset/empty.
#   * exits 0 for anything else (render modules -- unexercised here since
#     BIFROST_CONTRACT_PATH / ONEX_SECRET_RESOLVER_CONFIG_PATH stay unset).
_PYTHON_STUB = """#!/bin/sh
case "$1" in
  --manifest)
    exit 0
    ;;
  -c)
    if [ "${STUB_C_RC:-0}" -ne 0 ]; then
      exit "${STUB_C_RC}"
    fi
    printf '%s\\n' "${STUB_RESOLVED_TIERS_PATH:-}"
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
"""


def _run_entrypoint(
    tmp_path: Path,
    *,
    delegation_tiers_path: str | None,
    resolved_tiers_path: str | None = None,
    c_rc: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Run the real entrypoint with a stubbed python and a controlled self-heal probe."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "python"
    stub.write_text(_PYTHON_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "OMNIBASE_INFRA_DB_URL": "postgresql://u:p@db:5432/omnibase_infra",
        "STUB_C_RC": str(c_rc),
    }
    if delegation_tiers_path is not None:
        env["DELEGATION_ROUTING_TIERS_PATH"] = delegation_tiers_path
    if resolved_tiers_path is not None:
        env["STUB_RESOLVED_TIERS_PATH"] = resolved_tiers_path

    return subprocess.run(
        ["sh", str(ENTRYPOINT), "true"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_missing_pinned_path_is_re_derived_and_exported() -> None:
    """Stale pin (file absent) + a real re-derived path -> self-heal, exports the
    corrected value, and boot proceeds.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        stale_pin = td_path / "does-not-exist" / "routing_tiers.yaml"
        real_resolved = td_path / "resolved" / "routing_tiers.yaml"
        real_resolved.parent.mkdir(parents=True)
        real_resolved.write_text("tiers: []\n")

        result = _run_entrypoint(
            td_path,
            delegation_tiers_path=str(stale_pin),
            resolved_tiers_path=str(real_resolved),
        )

    assert result.returncode == 0, result.stderr
    assert (
        f"WARNING: DELEGATION_ROUTING_TIERS_PATH={stale_pin} does not exist"
        in result.stdout
    )
    assert f"Re-derived DELEGATION_ROUTING_TIERS_PATH={real_resolved}" in result.stdout
    assert "Starting runtime kernel..." in result.stdout


def test_missing_pinned_path_and_failed_rederivation_leaves_pin_and_boots() -> None:
    """Stale pin + re-derivation ALSO fails (import error) -> the entrypoint does
    NOT crash and does NOT fabricate a path; it warns and boots, leaving the
    (still-stale) pin in place so the routing reducer fails closed attributably at
    first use -- never a silent config fallback.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        stale_pin = td_path / "does-not-exist" / "routing_tiers.yaml"

        result = _run_entrypoint(
            td_path,
            delegation_tiers_path=str(stale_pin),
            resolved_tiers_path=None,
            c_rc=1,
        )

    assert result.returncode == 0, result.stderr
    assert (
        f"WARNING: DELEGATION_ROUTING_TIERS_PATH={stale_pin} does not exist"
        in result.stdout
    )
    assert (
        "WARNING: could not re-derive a valid routing_tiers.yaml path" in result.stdout
    )
    # Never a fabricated success message when re-derivation genuinely failed.
    assert "Re-derived DELEGATION_ROUTING_TIERS_PATH=" not in result.stdout
    assert "Starting runtime kernel..." in result.stdout


def test_valid_pinned_path_is_left_untouched() -> None:
    """The pinned path exists on disk -> the self-heal block is a no-op (no warning,
    no re-derivation attempt).
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        valid_pin = td_path / "routing_tiers.yaml"
        valid_pin.write_text("tiers: []\n")

        result = _run_entrypoint(td_path, delegation_tiers_path=str(valid_pin))

    assert result.returncode == 0, result.stderr
    assert "does not exist -- attempting to re-derive" not in result.stdout
    assert "Starting runtime kernel..." in result.stdout


def test_unset_delegation_tiers_path_is_a_noop() -> None:
    """DELEGATION_ROUTING_TIERS_PATH unset entirely (e.g. projection-api, which
    deliberately has no delegation surface) -> the self-heal block never fires.
    """
    with tempfile.TemporaryDirectory() as td:
        result = _run_entrypoint(Path(td), delegation_tiers_path=None)

    assert result.returncode == 0, result.stderr
    assert "DELEGATION_ROUTING_TIERS_PATH" not in result.stdout
    assert "Starting runtime kernel..." in result.stdout


def test_self_heal_block_is_source_ordered_after_secret_resolver_render() -> None:
    """Static guard: the self-heal block must run after the existing render blocks
    (BIFROST_CONTRACT_PATH / ONEX_SECRET_RESOLVER_CONFIG_PATH) and before the kernel
    exec, matching the append-only placement of this remediation.
    """
    source = ENTRYPOINT.read_text()
    secret_render_pos = source.index("render_secret_resolver_config")
    self_heal_pos = source.index("does not exist -- attempting to re-derive")
    exec_pos = source.index('exec "$@"')

    assert secret_render_pos < self_heal_pos < exec_pos
