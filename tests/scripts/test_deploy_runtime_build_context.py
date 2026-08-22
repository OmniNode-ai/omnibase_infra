# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Deploy-runtime regression coverage for Docker build context paths."""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-runtime.sh"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile.runtime"
DOCKER_DIR = REPO_ROOT / "docker"

# Matches a COPY directive's argument list. We discard `--from=<stage>` lines
# (those copy from a prior build stage, not the host build context) and any
# remaining `--flag` tokens (e.g. `--chown=...`), then keep the source operands
# that pull from the workspace/ tree.
_COPY_LINE_RE = re.compile(r"^COPY\s+(?P<args>.+)$", re.MULTILINE)


def _dockerfile_workspace_copy_sources() -> list[str]:
    """Every Dockerfile.runtime COPY source that pulls from the workspace/ tree.

    The deployed build context is assembled by deploy-runtime.sh's sync_files;
    each of these paths must be rsynced (or generated) into that context or the
    workspace-mode `docker build` fails with "failed to calculate checksum ...:
    not found" (the OMN-12987 regression). This list is derived from the live
    Dockerfile so a future COPY workspace/<x> without a matching rsync is caught.
    """
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    sources: list[str] = []
    for match in _COPY_LINE_RE.finditer(dockerfile):
        tokens = match.group("args").split()
        # `--from=<stage>` copies from a build stage, never the host context.
        if any(tok.startswith("--from=") for tok in tokens):
            continue
        # Drop flag tokens (--chown=, --link, ...); the last operand is the
        # destination, everything before it is a build-context source.
        operands = [tok for tok in tokens if not tok.startswith("--")]
        sources.extend(src for src in operands[:-1] if src.startswith("workspace/"))
    return sources


def _dockerfile_config_copy_sources() -> list[str]:
    """Every Dockerfile.runtime COPY source that pulls from the config/ tree.

    Same rationale as `_dockerfile_workspace_copy_sources` (OMN-12987), applied
    to config/ (OMN-15696): sync_files() must rsync config/ into the deployed
    build context or a COPY config/<x> fails workspace-mode `docker build` with
    "failed to calculate checksum ...: not found".
    """
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    sources: list[str] = []
    for match in _COPY_LINE_RE.finditer(dockerfile):
        tokens = match.group("args").split()
        if any(tok.startswith("--from=") for tok in tokens):
            continue
        operands = [tok for tok in tokens if not tok.startswith("--")]
        sources.extend(src for src in operands[:-1] if src.startswith("config/"))
    return sources


@pytest.mark.unit
def test_deploy_runtime_syncs_runtime_dockerfile_copy_sources() -> None:
    """deploy-runtime.sh must ship paths copied by Dockerfile.runtime."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    required_sources = (
        "workspace/sibling-repos/",
        "scripts/runtime_build/compute_workspace_provenance.py",
    )
    for source in required_sources:
        assert source in dockerfile

    assert '"${repo_root}/workspace/sibling-repos/"' in deploy_script
    assert '"${repo_root}/scripts/runtime_build/"' in deploy_script


@pytest.mark.unit
def test_deploy_runtime_stages_every_workspace_copy_source() -> None:
    """Every `COPY workspace/<x>` in Dockerfile.runtime must be staged.

    Regression guard for OMN-12987: Dockerfile.runtime COPYs
    workspace/sibling-pin-comparison.json (and workspace/sibling-repos/), but an
    earlier deploy-runtime.sh only rsynced workspace/sibling-repos/ into the
    deployed build context. The root-level comparison file was never carried
    over, so every workspace-mode `docker build` failed with "failed to
    calculate checksum of ref ...:/workspace/sibling-pin-comparison.json: not
    found". The dev compose build masked this because it runs from the repo root
    where the committed placeholder exists.

    This test derives the workspace/ COPY sources from the live Dockerfile and
    asserts deploy-runtime.sh references each as an rsync source for the deployed
    context, so a future Dockerfile COPY without a matching rsync fails CI.
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    workspace_sources = _dockerfile_workspace_copy_sources()
    # The Dockerfile must at minimum COPY the two known workspace/ paths; a regex
    # that silently matched nothing would make this guard vacuously pass.
    assert "workspace/sibling-repos/" in workspace_sources
    assert "workspace/sibling-pin-comparison.json" in workspace_sources

    missing: list[str] = []
    for source in workspace_sources:
        # A directory source (trailing slash) and a file source both appear in
        # deploy-runtime.sh as an rsync argument quoted under ${repo_root}.
        staged = f'"${{repo_root}}/{source}"' in deploy_script
        if not staged:
            missing.append(source)

    assert not missing, (
        "Dockerfile.runtime COPYs these workspace/ paths but deploy-runtime.sh "
        f"does not stage them into the deployed build context: {missing}. Add an "
        "rsync of each into sync_files() or workspace-mode `docker build` will "
        "fail with 'failed to calculate checksum ...: not found' (OMN-12987)."
    )


@pytest.mark.unit
def test_deploy_runtime_stages_every_config_copy_source() -> None:
    """Every `COPY config/<x>` in Dockerfile.runtime must be staged.

    Regression guard for OMN-15696: Dockerfile.runtime COPYs
    config/runner_fleet.yaml (added by OMN-15676), but sync_files() never
    rsynced config/ into the deployed build context, so any --force redeploy
    or cold bring-up that recreates deployed/<version>/ failed the image build
    with "failed to calculate checksum of ref ...:/config/runner_fleet.yaml:
    not found" -- the same COPY-without-matching-rsync class OMN-12987 fixed
    for workspace/.

    This test derives the config/ COPY sources from the live Dockerfile and
    asserts deploy-runtime.sh stages each -- either via an exact-file rsync
    argument, or by rsyncing the containing config/ directory -- so a future
    Dockerfile COPY without a matching rsync fails CI.
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    config_sources = _dockerfile_config_copy_sources()
    # The Dockerfile must at minimum COPY the known runner_fleet.yaml path; a
    # regex that silently matched nothing would make this guard vacuously pass.
    assert "config/runner_fleet.yaml" in config_sources

    directory_staged = '"${repo_root}/config/"' in deploy_script

    missing: list[str] = []
    for source in config_sources:
        staged = directory_staged or f'"${{repo_root}}/{source}"' in deploy_script
        if not staged:
            missing.append(source)

    assert not missing, (
        "Dockerfile.runtime COPYs these config/ paths but deploy-runtime.sh "
        f"does not stage them into the deployed build context: {missing}. Add an "
        "rsync of each (or of config/ as a whole) into sync_files() or "
        "workspace-mode `docker build` will fail with 'failed to calculate "
        "checksum ...: not found' (OMN-15696)."
    )


# =============================================================================
# General build-context parity: every Dockerfile the deploy path can build,
# every COPY/ADD source it references, matched against every rsync rule
# deploy-runtime.sh's sync_files() actually stages (OMN-16103).
#
# The scoped tests above (workspace/, config/) only fired because someone
# hand-listed the prefix to check. OMN-16103 was a `COPY scripts/<x>` that no
# prefix-scoped test covered: docker/Dockerfile.runtime's
# `COPY scripts/seed-keycloak-clients.py /app/scripts/seed-keycloak-clients.py`
# COPY is a *continuation-wrapped* multi-line directive, and sync_files()'s
# scripts/ rsync is an --include allowlist (not a directory-wide sync), so the
# file silently never reached the deployed build context. Every .201
# git-ref redeploy then failed at `docker build` with "failed to calculate
# checksum of ref ...:/app/scripts/seed-keycloak-clients.py: not found"
# (or the equivalent COPY-source-missing error) before a single container
# started.
#
# This block derives, from the live source, an actual coverage model instead
# of a hand-maintained prefix list, so the *next* out-of-manifest COPY (any
# prefix, any directive, single-line or continuation-wrapped) fails this test
# in CI rather than failing a live host deploy.
# =============================================================================

_COMPOSE_FILENAME_RE = re.compile(r"docker-compose\.[\w.-]+\.yml")
_DOCKERFILE_ENTRY_RE = re.compile(r"^\s*dockerfile:\s*(\S+)\s*$", re.MULTILINE)
_COPY_OR_ADD_RE = re.compile(r"^(?:COPY|ADD)\s+(?P<args>.+)$", re.MULTILINE)


def _join_line_continuations(text: str) -> str:
    """Join `\\`-continued lines into one logical line each.

    Shared join logic for both bash (deploy-runtime.sh) and Dockerfile syntax
    -- both use a trailing backslash for line continuation. Without this, a
    multi-line `COPY --chown=... \\\n    src \\\n    dst` (or the equivalent
    wrapped `rsync ... \\\n    src dst`) is invisible to a single-line regex,
    which is exactly how the OMN-16103 COPY escaped the scoped tests above.
    """
    joined_lines: list[str] = []
    pending = ""
    for raw_line in text.split("\n"):
        line = f"{pending} {raw_line.strip()}" if pending else raw_line
        pending = ""
        stripped = line.rstrip()
        if stripped.endswith("\\") and not stripped.endswith("\\\\"):
            pending = stripped[:-1].rstrip()
            continue
        joined_lines.append(line)
    if pending:
        joined_lines.append(pending)
    return "\n".join(joined_lines)


def _dockerfiles_built_by_deploy_path() -> list[Path]:
    """Every Dockerfile reachable from a deploy-runtime.sh compose invocation.

    deploy-runtime.sh always layers docker-compose.infra.yml plus exactly one
    lane overlay (resolve_compose_file_args()); rather than re-implement that
    lane-selection logic here (and drift from it), this scans every
    docker-compose.*.yml filename literally referenced anywhere in
    deploy-runtime.sh (comments included -- every real overlay is named in a
    comment near resolve_compose_file_args()/resolve_lane_overlay_filename())
    and collects the `dockerfile:` build source each one declares. A regex
    that matched nothing would silently vacuous-pass the parity test below,
    so this is asserted non-empty.
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    filenames = sorted(set(_COMPOSE_FILENAME_RE.findall(deploy_script)))

    dockerfiles: set[Path] = set()
    for filename in filenames:
        compose_path = DOCKER_DIR / filename
        if not compose_path.is_file():
            continue
        compose_text = compose_path.read_text(encoding="utf-8")
        for match in _DOCKERFILE_ENTRY_RE.finditer(compose_text):
            # Every compose file in this repo builds with `context: ..`
            # (repo root); the dockerfile: value is a repo-root-relative path.
            dockerfiles.add(REPO_ROOT / match.group(1))

    return sorted(p for p in dockerfiles if p.is_file())


def _dockerfile_copy_add_sources(dockerfile_text: str) -> list[str]:
    """Every non-stage, non-absolute COPY/ADD source in a Dockerfile.

    Handles both single-line and `\\`-continued multi-line directives.
    `--from=<stage>` sources copy from a prior build stage, never the host
    build context, and are excluded. Absolute paths and URL sources (ADD
    supports remote URLs) are excluded -- those never come from the rsynced
    build context either.
    """
    joined = _join_line_continuations(dockerfile_text)
    sources: list[str] = []
    for match in _COPY_OR_ADD_RE.finditer(joined):
        tokens = match.group("args").split()
        if any(tok.startswith("--from=") for tok in tokens):
            continue
        operands = [tok for tok in tokens if not tok.startswith("--")]
        if len(operands) < 2:
            continue
        for src in operands[:-1]:
            if src.startswith("/") or "://" in src:
                continue
            sources.append(src)
    return sources


def _for_loop_variable_values(deploy_script: str) -> dict[str, list[str]]:
    """Resolve simple `for VAR in word1 word2; do` loop bodies.

    sync_files() syncs README.md/LICENSE via `for f in README.md LICENSE; do
    rsync -a "${repo_root}/${f}" ...; done` rather than one rsync per file.
    This lets the rsync-rule extractor below expand `${f}` into the literal
    words the loop actually iterates, instead of treating it as an
    unresolvable variable reference.
    """
    values: dict[str, list[str]] = {}
    for match in re.finditer(r"for\s+(\w+)\s+in\s+([^;]+);\s*do", deploy_script):
        values[match.group(1)] = shlex.split(match.group(2))
    return values


class _RsyncCoverageRule:
    """One rsync invocation's (source, include/exclude filter) coverage."""

    def __init__(
        self,
        src_rel: str,
        is_dir: bool,
        includes: list[str],
        excludes: list[str],
    ) -> None:
        self.src_rel = src_rel.strip("/")
        self.is_dir = is_dir
        self.includes = includes
        self.excludes = excludes

    def covers(self, path_rel: str) -> bool:
        path_rel = path_rel.strip("/")
        if not self.is_dir:
            return path_rel == self.src_rel
        if path_rel != self.src_rel and not path_rel.startswith(f"{self.src_rel}/"):
            return False
        sub = path_rel[len(self.src_rel) :].lstrip("/")
        # An --include allowlist (paired with a trailing --exclude='*', the
        # pattern scripts/ uses) covers ONLY the listed relative paths -- NOT
        # the whole directory. This is the exact distinction that let
        # scripts/seed-keycloak-clients.py silently miss: scripts/runtime_build/
        # is a full directory sync while scripts/ itself is an allowlist, and
        # a coverage check that treated both as "scripts/ is synced" would
        # stay vacuously green.
        if self.includes:
            return any(fnmatch.fnmatch(sub, pattern) for pattern in self.includes)
        if self.excludes:
            if any(
                fnmatch.fnmatch(f"/{sub}", pattern)
                or fnmatch.fnmatch(sub, pattern.lstrip("/"))
                for pattern in self.excludes
            ):
                return False
        return True


def _rsync_manifest_rules(deploy_script: str) -> list[_RsyncCoverageRule]:
    """Every rsync rule deploy-runtime.sh stages from `${repo_root}` sources.

    Parses every logical `rsync ...` invocation in the script (after joining
    line continuations), keeping only source operands anchored at
    `${repo_root}/` -- deploy-runtime.sh also rsyncs migration-tree snapshots
    from unrelated `${src_tree}`/`${snapshot_dir}` vars (line ~1542/1565);
    those aren't repo_root-anchored and are naturally excluded rather than
    needing an explicit denylist.
    """
    joined = _join_line_continuations(deploy_script)
    loop_vars = _for_loop_variable_values(deploy_script)
    rules: list[_RsyncCoverageRule] = []

    for line in joined.split("\n"):
        stripped = line.strip()
        if not (stripped.startswith("rsync ") or stripped == "rsync"):
            continue
        try:
            tokens = shlex.split(stripped)[1:]  # drop the leading "rsync"
        except ValueError:
            continue

        includes: list[str] = []
        excludes: list[str] = []
        paths: list[str] = []
        for tok in tokens:
            if tok.startswith("--include="):
                includes.append(tok[len("--include=") :])
            elif tok.startswith("--exclude="):
                excludes.append(tok[len("--exclude=") :])
            elif tok.startswith("-"):
                continue
            else:
                paths.append(tok)

        if len(paths) < 2:
            continue
        srcs = paths[:-1]  # last path operand is always the destination

        for src in srcs:
            if "${repo_root}/" not in src:
                continue
            rel = src.split("${repo_root}/", 1)[1]
            if "${f}" in rel and "f" in loop_vars:
                for value in loop_vars["f"]:
                    resolved = rel.replace("${f}", value)
                    rules.append(
                        _RsyncCoverageRule(
                            resolved, resolved.endswith("/"), includes, excludes
                        )
                    )
                continue
            if "$" in rel:
                # An unresolved variable reference this parser doesn't model
                # (none exist today) -- skip rather than guess at coverage.
                continue
            rules.append(_RsyncCoverageRule(rel, rel.endswith("/"), includes, excludes))

    return rules


@pytest.mark.unit
def test_deploy_runtime_rsync_manifest_covers_every_dockerfile_copy_source() -> None:
    """Every Dockerfile the deploy path builds must have its COPY/ADD sources
    covered by deploy-runtime.sh's rsync manifest (OMN-16103).

    General build-context parity guard, superseding the need to hand-add a
    prefix-scoped test (like the workspace/ and config/ tests above) every
    time a new COPY namespace appears. Derives both sides from live source:
    the Dockerfile set from every compose file deploy-runtime.sh can invoke,
    and the coverage model from every rsync rule (including --include
    allowlists, which only cover their listed paths -- not their whole
    source directory) in sync_files(). A future `COPY <path>` with no
    matching rsync now fails this test in CI instead of failing every
    subsequent host `--execute` redeploy at the Docker build step.
    """
    dockerfiles = _dockerfiles_built_by_deploy_path()
    # A discovery regex that matched nothing would make this test vacuously
    # pass -- assert we found at least the known runtime Dockerfile.
    assert DOCKERFILE in dockerfiles, (
        f"Dockerfile discovery found {dockerfiles!r}, missing the known "
        f"{DOCKERFILE}. The compose-filename or dockerfile: regex likely "
        "drifted from the live docker-compose.*.yml files."
    )

    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")
    rules = _rsync_manifest_rules(deploy_script)
    assert rules, "rsync rule extraction found nothing -- parser likely broken."

    failures: dict[str, list[str]] = {}
    for dockerfile in dockerfiles:
        sources = _dockerfile_copy_add_sources(dockerfile.read_text(encoding="utf-8"))
        missing = [s for s in sources if not any(rule.covers(s) for rule in rules)]
        if missing:
            failures[str(dockerfile.relative_to(REPO_ROOT))] = missing

    assert not failures, (
        "These Dockerfile COPY/ADD sources are not covered by any rsync rule "
        f"in deploy-runtime.sh's sync_files(): {failures}. Every .201 "
        "git-ref redeploy building this image will fail at `docker build` "
        "with a COPY-source-not-found error. Add a matching rsync (or "
        "--include entry, if the containing directory uses an allowlist "
        "sync) to sync_files()."
    )


def _init_git_repo(path: Path, marker: str) -> str:
    path.mkdir(parents=True)
    (path / "marker.txt").write_text(marker, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "marker.txt"], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=deploy-runtime-test",
            "-c",
            "user.email=deploy-runtime-test@example.com",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        cwd=path,
        check=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_fake_docker(bin_dir: Path) -> None:
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env sh
set -eu
if [ "$1" = "compose" ] && [ "$2" = "version" ] && [ "$3" = "--short" ]; then
  printf '2.27.0\\n'
  exit 0
fi
printf 'unexpected docker invocation: %s\\n' "$*" >&2
exit 1
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)


@pytest.mark.unit
def test_workspace_printed_build_command_uses_operator_omni_home(
    tmp_path: Path,
) -> None:
    """Operator OMNI_HOME must survive ~/.omnibase/.env during workspace builds."""
    operator_home = tmp_path / "operator" / "omni_home"
    env_home_root = tmp_path / "env-file" / "omni_home"
    operator_omnimarket_sha = _init_git_repo(
        operator_home / "omnimarket", "operator-root"
    )
    env_omnimarket_sha = _init_git_repo(env_home_root / "omnimarket", "env-root")

    home = tmp_path / "home"
    omnibase_dir = home / ".omnibase"
    omnibase_dir.mkdir(parents=True)
    (omnibase_dir / ".env").write_text(
        f"INFRA_HOST=127.0.0.1\nOMNI_HOME={env_home_root}\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_docker(bin_dir)

    env = os.environ.copy()
    env.update(
        {
            "BUILD_SOURCE": "workspace",
            "HOME": str(home),
            "OMNI_HOME": str(operator_home),
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        }
    )
    env.pop("EXPECTED_BUILD_SOURCE", None)

    result = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "--print-compose-cmd"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"--build-arg OMNI_HOME={operator_home}" in result.stdout
    assert "--build-arg BUILD_SOURCE=workspace" in result.stdout
    assert "--build-arg EXPECTED_BUILD_SOURCE=workspace" in result.stdout
    assert f"--build-arg OMNIMARKET_REF={operator_omnimarket_sha}" in result.stdout
    assert env_omnimarket_sha not in result.stdout


@pytest.mark.unit
def test_deploy_runtime_stages_workspace_and_passes_omni_home_arg() -> None:
    """Workspace mode must stage sibling repos and pass OMNI_HOME to Dockerfile."""
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'stage_workspace_if_needed "${repo_root}"' in deploy_script
    assert '--build-arg "BUILD_SOURCE=${build_source}"' in deploy_script
    assert (
        '--build-arg "EXPECTED_BUILD_SOURCE=${expected_build_source}"' in deploy_script
    )
    assert '--build-arg "OMNI_HOME=${omni_home}"' in deploy_script


@pytest.mark.unit
def test_deploy_runtime_runs_sibling_lock_pin_preflight() -> None:
    """Workspace staging must run the OMN-12987 lock-pin preflight before build.

    Recurrence guard: the 2026-06-11 stability crash shipped a 13-day-stale
    infra 0.37.0 because the build ignored the omnimarket dev lock. The deploy
    script must invoke check_sibling_lock_pins after staging and abort on
    failure.
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    # The preflight is called from stage_workspace_if_needed after staging.
    assert 'check_sibling_lock_pins "${repo_root}" "${omni_home}"' in deploy_script
    # The preflight function references the guard script and aborts on failure.
    assert "scripts/runtime_build/check_sibling_lock_pins.py" in deploy_script
    assert "Refusing to build a stale image." in deploy_script


@pytest.mark.unit
def test_deploy_runtime_uses_current_lock_pin_preflight_interface() -> None:
    """The preflight caller must match check_sibling_lock_pins.py's current CLI.

    Regression guard: OMN-12977/12987 replaced the original ``--provenance-out``
    flag with ``--lock`` (required pin authority), repeatable ``--repo
    PACKAGE=PATH`` (the canonical clones the build vendors), and ``--output``
    (where to write the comparison JSON). The deploy-runtime.sh caller was left
    pinned to the removed ``--provenance-out`` flag, so EVERY workspace
    ``--execute`` deploy failed at argparse (``the following arguments are
    required: --lock``) before any build started. This test pins the corrected
    interface so the stale-flag invocation can never come back.
    """
    deploy_script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    # Removed flag must be gone — its presence is the exact regression we hit.
    assert "--provenance-out" not in deploy_script

    # Current required + supported flags must be wired into the guard_args.
    assert '--lock "${lock_path}"' in deploy_script
    assert "--repo " in deploy_script
    assert '--output "${provenance_out}"' in deploy_script

    # The consuming repo's omnimarket uv.lock is the pin authority, and every
    # vendored sibling must be passed as a --repo PACKAGE=PATH entry.
    assert 'lock_path="${omni_home}/omnimarket/uv.lock"' in deploy_script
    for package in (
        "omnibase-infra",
        "omnibase-core",
        "omnibase-spi",
        "omnibase-compat",
    ):
        assert f'--repo "{package}=' in deploy_script

    # OMN-16296 removed onex_change_control from the runtime image and from
    # check_sibling_lock_pins.py's DEFAULT_PACKAGE_REPO_DIRS (commit 018c64a14),
    # but left this preflight's guard_args pinned to the removed package,
    # which fails argparse validation
    # (``error: argument --repo: unknown package 'onex-change-control'``) on
    # every workspace-mode deploy. OMN-16390 fixes the residual. Pin the
    # absence so a future re-add has to be deliberate.
    assert '--repo "onex-change-control=' not in deploy_script

    # The OMN-12977 operator override must be honored via --allow-drift.
    assert "ALLOW_SIBLING_PIN_DRIFT" in deploy_script
    assert "--allow-drift" in deploy_script


@pytest.mark.unit
def test_stage_workspace_emits_build_sha_marker() -> None:
    """stage_workspace.sh must record each sibling's HEAD SHA (OMN-12987).

    rsync drops .git, so without a .build-sha marker the staged tree has no
    recoverable SHA and the lock-pin preflight cannot verify the vendored commit.
    """
    stage_script = (
        DEPLOY_SCRIPT.parent / "runtime_build" / "stage_workspace.sh"
    ).read_text(encoding="utf-8")

    # OMN-13030 refactored the SHA capture to a variable (reused for the
    # per-repo VCS provenance manifest), so assert the marker is still written
    # from the resolved HEAD SHA. OMN-14900 scoped the probe with
    # `-c safe.directory=${src}` so a uid-mismatched invoker (the deploy
    # runner container) is never rejected with "dubious ownership".
    assert 'git -c "safe.directory=${src}" -C "${src}" rev-parse HEAD' in stage_script
    assert 'echo "${vcs_ref}" > "${dst}/.build-sha"' in stage_script
