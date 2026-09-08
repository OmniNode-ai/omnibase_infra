# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-15628 remediation: boot self-heal for a stale DELEGATION_ROUTING_TIERS_PATH.

The k8s manifests (omninode_infra: deployment-omninode-runtime{,-effects,-worker}.yaml)
pin ``DELEGATION_ROUTING_TIERS_PATH`` as a literal string embedding the venv's Python
minor version, e.g.::

    /app/.venv/lib/python3.12/site-packages/omnimarket/configs/routing_tiers.yaml

A base-image Python version bump silently invalidates that literal path with no signal
until the routing reducer fails closed at first use. The boot preflight self-heals: when
the pinned path does not exist on disk, it re-derives the path from the installed
``omnimarket`` package's OWN location (which always matches whatever Python actually
ships in the image) and exports the corrected value before the kernel starts. If
re-derivation also fails, the original (possibly-stale) value is left untouched so the
routing reducer still fails closed attributably (CLAUDE.md rule 8) -- this is a
best-effort correction, never a silent fallback that manufactures a config the reducer
would otherwise refuse to load.

OMN-17372 moved this block out of ``docker/entrypoint-runtime.sh`` (where it cost a
FIFTH cold ``python -c`` interpreter start) into
``omnibase_infra.runtime.entrypoint_preflight``, in the same process as every other
boot step. The behaviour, the operator-visible text and the never-fabricate rule are
unchanged; these tests drive the function directly instead of a stubbed shell.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from omnibase_infra.runtime import entrypoint_preflight

pytestmark = [pytest.mark.unit]


def _install_fake_omnimarket(
    monkeypatch: pytest.MonkeyPatch, package_dir: Path
) -> None:
    """Make ``import omnimarket`` resolve to *package_dir*."""
    module = types.ModuleType("omnimarket")
    module.__file__ = str(package_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "omnimarket", module)


def _break_omnimarket_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import omnimarket`` raise, emulating a package that is not installed."""
    monkeypatch.setitem(sys.modules, "omnimarket", None)


def test_missing_pinned_path_is_re_derived_and_exported(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Stale pin + a real re-derived path -> self-heal exports the corrected value."""
    stale_pin = tmp_path / "does-not-exist" / "routing_tiers.yaml"
    package_dir = tmp_path / "site-packages" / "omnimarket"
    (package_dir / "configs").mkdir(parents=True)
    resolved = package_dir / "configs" / "routing_tiers.yaml"
    resolved.write_text("tiers: []\n")
    _install_fake_omnimarket(monkeypatch, package_dir)

    env = {"DELEGATION_ROUTING_TIERS_PATH": str(stale_pin)}
    entrypoint_preflight.self_heal_delegation_tiers_path(env)

    out = capsys.readouterr().out
    assert f"WARNING: DELEGATION_ROUTING_TIERS_PATH={stale_pin} does not exist" in out
    assert f"Re-derived DELEGATION_ROUTING_TIERS_PATH={resolved}" in out
    assert env["DELEGATION_ROUTING_TIERS_PATH"] == str(resolved)


def test_missing_pinned_path_and_failed_rederivation_leaves_pin(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Re-derivation ALSO fails -> warn, never fabricate, leave the stale pin.

    The routing reducer then fails closed with an attributable error at first
    use, which is the intended outcome -- not a silent config fallback.
    """
    stale_pin = tmp_path / "does-not-exist" / "routing_tiers.yaml"
    _break_omnimarket_import(monkeypatch)

    env = {"DELEGATION_ROUTING_TIERS_PATH": str(stale_pin)}
    entrypoint_preflight.self_heal_delegation_tiers_path(env)

    out = capsys.readouterr().out
    assert f"WARNING: DELEGATION_ROUTING_TIERS_PATH={stale_pin} does not exist" in out
    assert "WARNING: could not re-derive a valid routing_tiers.yaml path" in out
    assert "Re-derived DELEGATION_ROUTING_TIERS_PATH=" not in out
    assert env["DELEGATION_ROUTING_TIERS_PATH"] == str(stale_pin)


def test_rederived_path_that_is_not_a_real_file_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """omnimarket imports, but ships no routing_tiers.yaml -> still no fabrication."""
    stale_pin = tmp_path / "does-not-exist" / "routing_tiers.yaml"
    package_dir = tmp_path / "site-packages" / "omnimarket"
    package_dir.mkdir(parents=True)
    _install_fake_omnimarket(monkeypatch, package_dir)

    env = {"DELEGATION_ROUTING_TIERS_PATH": str(stale_pin)}
    entrypoint_preflight.self_heal_delegation_tiers_path(env)

    out = capsys.readouterr().out
    assert "WARNING: could not re-derive a valid routing_tiers.yaml path" in out
    assert env["DELEGATION_ROUTING_TIERS_PATH"] == str(stale_pin)


def test_valid_pinned_path_is_left_untouched(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The pinned path exists -> the self-heal block is a no-op."""
    valid_pin = tmp_path / "routing_tiers.yaml"
    valid_pin.write_text("tiers: []\n")

    env = {"DELEGATION_ROUTING_TIERS_PATH": str(valid_pin)}
    entrypoint_preflight.self_heal_delegation_tiers_path(env)

    out = capsys.readouterr().out
    assert "does not exist -- attempting to re-derive" not in out
    assert env["DELEGATION_ROUTING_TIERS_PATH"] == str(valid_pin)


def test_unset_delegation_tiers_path_is_a_noop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unset entirely (e.g. projection-api) -> the block never fires."""
    env: dict[str, str] = {}
    entrypoint_preflight.self_heal_delegation_tiers_path(env)

    assert capsys.readouterr().out == ""
    assert env == {}


def test_self_heal_runs_after_both_renders_and_before_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order guard: stamp -> bifrost -> resolver -> self-heal, then the CMD.

    This is the append-only placement the shell had, and the renders must run
    before the self-heal because the routing tiers are read after boot config
    is materialised.
    """
    order: list[str] = []

    monkeypatch.setattr(
        entrypoint_preflight,
        "stamp_all_fingerprints",
        lambda _env: (order.append("stamp"), 0)[1],
    )
    monkeypatch.setattr(
        entrypoint_preflight,
        "render_bifrost_contract",
        lambda _env: (order.append("bifrost"), 0)[1],
    )
    monkeypatch.setattr(
        entrypoint_preflight,
        "render_resolver_config",
        lambda _env: (order.append("resolver"), 0)[1],
    )
    monkeypatch.setattr(
        entrypoint_preflight,
        "self_heal_delegation_tiers_path",
        lambda _env: order.append("self_heal"),
    )

    rc = entrypoint_preflight.run_preflight({})

    assert rc == 0
    assert order == ["stamp", "bifrost", "resolver", "self_heal"]


__all__: list[str] = []
