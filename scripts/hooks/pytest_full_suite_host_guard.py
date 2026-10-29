#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard unnarrowed local pytest runs using committed deployment host tables.

CI and narrowed runs stay exempt. Missing private configuration designates no
host and requires configuration or a scoped grant. Present untrusted tables
refuse with their specific reason before a grant can be consumed.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

from scripts.hooks.prepush_override_grant import (
    GrantError,
    consume,
    env_rejection_message,
    has_active_override,
    inherited_override_env_vars,
    inherited_placement_map_env_vars,
    placement_map_rejection_message,
    resolve_head_sha,
    resolve_repo_root,
)

HOST_TABLE_REL = "scripts/hooks/prepush_hosts.tsv"
# OMN-20939: the deployment's own rows live in a private table found through the
# workspace-config resolver, never in this repository.
PRIVATE_TABLE_REL = (
    "config/onex/overlays/prepush_hosts/prepush_hosts.omnibase_infra.tsv"
)
_GIT_LOCATION_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")

# Legacy env override -> host-table row label. An override REPLACES the row it
# names; it never ADDS a hostname to the designated set. That distinction is
# load-bearing: under a table listing several hosts, an override that merely
# appended a name could no longer DE-designate the local machine, silently
# inverting the OMN-15059 guard that
# `test_guard_refuses_full_suite_escalation_on_non_200_host` proves by forcing
# a nonsense hostname.
_LEGACY_OVERRIDE_BY_LABEL = {
    "h200": "PREPUSH_200_HOSTNAME",
    "h201c": "PREPUSH_201_GATE_RUNNER_HOSTNAME",
}


def _override_var(label: str) -> str:
    """Per-row override env var name, matching prepush_dispatch.sh."""
    sanitized = "".join(c if c.isalnum() else "_" for c in label.upper())
    return f"PREPUSH_HOST_OVERRIDE_{sanitized}"


class HostTableError(ValueError):
    """A present host table cannot be trusted."""


def _workspace_config_root(env: dict[str, str] | os._Environ[str]) -> Path | None:
    """Pre-push permits explicit config or the registry sibling only."""
    if configured := env.get("ONEX_WORKSPACE_CONFIG_ROOT", "").strip():
        return Path(configured).resolve()
    if registry := env.get("OMNI_HOME", "").strip():
        from omnibase_infra.runtime.service_kernel import workspace_runtime_config_root

        return workspace_runtime_config_root(
            Path(registry), environment=env, allow_developer_config=False
        )
    return None


def _table_text(root: Path, relative: str, *, private: bool) -> str | None:
    """Read a table, requiring HEAD parity when its root is a git checkout."""
    path = root / relative
    kind = "private host table" if private else "host table"
    git_env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_ENV}
    try:
        if private and not path.exists() and not path.is_symlink():
            return None
        working = path.read_text(encoding="utf-8")
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=git_env,
        )
        if private and (
            top.returncode or Path(top.stdout.strip()).resolve() != root.resolve()
        ):
            return working
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=git_env,
        )
        if committed.returncode:
            raise HostTableError(f"{kind} has no committed copy at HEAD ({path})")
        if working != committed.stdout:
            raise HostTableError(
                f"{kind} differs between the working tree and HEAD ({path})"
            )
        return committed.stdout
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise HostTableError(f"{kind} unreadable ({path}): {exc}") from exc


def _private_table_text(
    env: dict[str, str] | os._Environ[str],
    relative: str = PRIVATE_TABLE_REL,
) -> str | None:
    """None means absent; present untrusted data raises HostTableError."""
    root = _workspace_config_root(env)
    return _table_text(root, relative, private=True) if root and relative else None


def designated_hostnames(env: dict[str, str] | None = None) -> tuple[str, ...]:
    """Committed identities; untrusted tables are refusals, never empty defaults."""
    active_env = os.environ if env is None else env
    table = _table_text(resolve_repo_root(), HOST_TABLE_REL, private=False) or ""
    table += "\n" + (_private_table_text(active_env) or "")
    names = []
    for line in table.splitlines():
        fields = line.split("#", 1)[0].split("\t")
        if len(fields) < 12 or fields[11] != "authorizing":
            continue
        label, hostname = fields[0], fields[2]
        legacy = _LEGACY_OVERRIDE_BY_LABEL.get(label, "")
        names.append(
            active_env.get(_override_var(label)) or active_env.get(legacy) or hostname
        )
    return tuple(names)


def is_ci_environment(env: dict[str, str] | None = None) -> bool:
    """True when running under a CI runner -- this guard never gates CI."""
    active_env: dict[str, str] | os._Environ[str] = os.environ if env is None else env
    return bool(active_env.get("CI") or active_env.get("GITHUB_ACTIONS"))


def resolve_local_hostname() -> str:
    """Short hostname of the current machine, or "" if undetermined."""
    try:
        return socket.gethostname().split(".", 1)[0]
    except OSError:
        return ""


def is_full_suite_target(
    *,
    args: list[str],
    testpaths: list[str],
    keyword: str,
    markexpr: str,
    full_suite_target: str,
) -> bool:
    """Whether this invocation is an unnarrowed collection of the full suite."""
    if keyword or markexpr:
        return False
    targets = list(args) if args else list(testpaths)
    if not targets:
        return False
    normalized_full = full_suite_target.rstrip("/") + "/"
    for raw in targets:
        normalized = str(raw).rstrip("/") + "/"
        if normalized_full.startswith(normalized):
            return True
    return False


def full_suite_host_violation_message(
    *,
    host: str,
    target_hostname: str,
    additional_target_hostnames: tuple[str, ...] = (),
    override_authorized: bool,
) -> str | None:
    """Return a refusal message, or None if this run may proceed."""
    if not host:
        return None
    allowed_hostnames = tuple(
        name for name in (target_hostname, *additional_target_hostnames) if name
    )
    if any(host.lower() == allowed.lower() for allowed in allowed_hostnames):
        return None
    if override_authorized:
        return None
    if not allowed_hostnames:
        return (
            f"direct full-suite pytest invocation refused on host '{host}': no gate "
            "host is configured. The shipped table "
            f"({HOST_TABLE_REL}) carries only a disabled localhost row; "
            "the deployment's hosts come from its own table at "
            f"${{ONEX_WORKSPACE_CONFIG_ROOT}}/{PRIVATE_TABLE_REL} "
            "(fifteen columns, documented in the header of the shipped table). "
            "Create that file and set ONEX_WORKSPACE_CONFIG_ROOT to its root, run "
            "from a configured gate host, OR mint a single-use override grant to "
            "run the full suite on this host anyway (visible, receipted, "
            "degraded-evidence -- do not use as a routine bypass): "
            "`uv run python scripts/hooks/prepush_override_grant.py mint "
            "--reason '<why>'`."
        )
    return (
        f"direct full-suite pytest invocation refused on host '{host}', not a "
        "designated gate host carrying mode=authorizing in the host tables "
        f"({', '.join(repr(allowed) for allowed in allowed_hostnames)}). "
        "This closes OMN-15977 "
        "Hole 1: agent-launched direct `pytest tests/` runs bypass the git-push "
        "guard (scripts/hooks/prepush_smart_tests.sh) entirely, so the designated-"
        "host check was never consulted. Run from a designated gate host instead "
        f"(see {HOST_TABLE_REL}, whose header documents how to add one), OR mint "
        "a single-use "
        "override grant to run the full suite on this host anyway (visible, "
        "receipted, degraded-evidence -- do not use as a routine bypass): "
        "`uv run python scripts/hooks/prepush_override_grant.py mint "
        "--reason '<why>'`."
    )


def override_authorization(*, context: str) -> bool:
    """Whether a scoped override authorizes THIS full-suite run."""
    try:
        repo_root = resolve_repo_root()
        head_sha = resolve_head_sha(repo_root)
    except Exception:  # noqa: BLE001 -- see fail-closed note below
        # Blanket by design, both here and below. This runs inside
        # pytest_configure on EVERY invocation: an unexpected exception must
        # become "not authorized", never a traceback that breaks collection for
        # everyone. Narrowing the clause would trade a fail-closed refusal for a
        # crash in the one path that must not crash.
        return False
    try:
        if has_active_override(repo_root=repo_root, head_sha=head_sha):
            return True
        return consume(repo_root=repo_root, head_sha=head_sha, context=context).accepted
    except Exception:  # noqa: BLE001 -- fail closed, never crash collection
        return False


def enforce(config: object, full_suite_target: str) -> None:
    """pytest_configure entry point -- call from conftest.py."""
    if is_ci_environment():
        return
    option = config.option  # type: ignore[attr-defined]
    if not is_full_suite_target(
        args=list(config.args),  # type: ignore[attr-defined]
        testpaths=list(config.getini("testpaths") or []),  # type: ignore[attr-defined]
        keyword=option.keyword or "",
        markexpr=option.markexpr or "",
        full_suite_target=full_suite_target,
    ):
        return

    import pytest

    # OMN-16480: a leaked PREPUSH_ALLOW_* variable is a REFUSAL, not a bypass.
    # Checked after the narrowing test (a targeted run is never gated) but
    # before the host check, so the leak is reported on the host where it
    # actually causes damage -- and so the OMN-16425 recursion terminates here
    # instead of spawning another full suite.
    leaked = inherited_override_env_vars(os.environ)
    if leaked:
        pytest.exit(env_rejection_message(leaked), returncode=1)

    # OMN-17441: the placement-map class, refused at the same point and for the
    # same reason -- one entry point honoring an inheritable knob is enough to
    # keep the class alive (OMN-15977 Hole 1 existed precisely because only the
    # push path was covered).
    #
    # This check is NOT vacuous despite running inside pytest. It fires at
    # `pytest_configure`, BEFORE collection, and PYTEST_CURRENT_TEST is set per
    # TEST -- so a direct `uv run pytest tests/unit/` launched from a shell
    # carrying a leaked map is refused here exactly as a push is, while a hook
    # subprocess spawned from inside a running test keeps the harness carve-out
    # that tests/ci/_prepush_lab_isolation.py depends on.
    leaked_maps = inherited_placement_map_env_vars(os.environ)
    if leaked_maps:
        pytest.exit(placement_map_rejection_message(leaked_maps), returncode=1)

    host = resolve_local_hostname()
    # OMN-16991: identity comes from the committed host table, not from
    # PREPUSH_* env vars the bash hook's scrub deliberately strips before it
    # spawns this pytest. See designated_hostnames().
    try:
        all_designated = designated_hostnames()
    except (HostTableError, GrantError) as exc:
        pytest.exit(f"direct full-suite pytest invocation refused: {exc}", returncode=1)
    target_hostname = all_designated[0] if all_designated else ""
    additional_hostnames = tuple(all_designated[1:])
    if (
        full_suite_host_violation_message(
            host=host,
            target_hostname=target_hostname,
            additional_target_hostnames=additional_hostnames,
            override_authorized=False,
        )
        is None
    ):
        return
    # Host mismatch. Only NOW look for an override -- resolving one has side
    # effects (it spends a grant and writes a receipt), so it must never run on
    # a path that was going to be allowed anyway.
    if override_authorization(
        context=f"direct full-suite pytest invocation on host '{host}'"
    ):
        return
    message = full_suite_host_violation_message(
        host=host,
        target_hostname=target_hostname,
        additional_target_hostnames=additional_hostnames,
        override_authorized=False,
    )
    assert message is not None

    pytest.exit(message, returncode=1)


def main() -> None:
    """Expose the private reader to the repository-neutral shell library."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--workspace-config-root", action="store_true")
    mode.add_argument("--private-table", metavar="RELATIVE_PATH")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    root = _workspace_config_root(os.environ)
    if args.workspace_config_root:
        if root is None:
            raise SystemExit(1)
        print(root)
        return
    try:
        table = _private_table_text(os.environ, args.private_table)
    except HostTableError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    if args.status:
        if table is None:
            print(
                f"deployment host table NOT CONFIGURED: {root}/{args.private_table} is absent -- "
                "set ONEX_WORKSPACE_CONFIG_ROOT to a root holding the declared table"
            )
        else:
            print(f"deployment host table: {root}/{args.private_table}")
    elif table is None:
        raise SystemExit(2)
    else:
        print(table, end="")


if __name__ == "__main__":
    main()
