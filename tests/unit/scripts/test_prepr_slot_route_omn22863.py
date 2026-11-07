# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-PR slot route proves an omnimarket-only pull request (OMN-22863).

Two defects held every omnimarket pull request at the lab proof:

* AC1: the slot route (``prepr_verify_lane.sh``) built the infra runtime only, so
  an omnimarket proof on a slot named no container that runs the package and the
  identity check read ``composed-containers-missing``. The route now layers the
  omnimarket tree over the slot's runtime image, before the slot starts, and the
  proof reads the slot's own containers that run that image.
* AC2: a slot lease whose owner is dead was refused for ever, because the
  reaper's teardown demanded a running gateway to re-prove an offboard that an
  earlier run had already proven and whose gateway was gone.

The shell functions run against a fake ``docker``; the lease code runs for real.
"""

from __future__ import annotations

import importlib.util
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from tests.helpers.prepr_pool_overlay import pool_overlay_env

ROOT = Path(__file__).resolve().parents[3]
PROVE_SH = ROOT / "scripts/runtime_build/prepr_pool_prove.sh"
VERIFY_SH = ROOT / "scripts/runtime_build/prepr_verify_lane.sh"
TEARDOWN = ROOT / "scripts/runtime_build/prepr_teardown_slot.py"

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("pool_overlay_env")]

SLOT = 2
PROJECT = f"omnibase-infra-prepr-{SLOT}"
KERNEL_M = f"omninode-prepr-{SLOT}-runtime"
KERNEL_E = f"omninode-prepr-{SLOT}-runtime-effects"
WORKER = f"omninode-prepr-{SLOT}-runtime-worker"
SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")


def _function(path: Path, name: str) -> str:
    script = path.read_text(encoding="utf-8")
    start = script.index(f"{name}() {{")
    end = script.index("\n}\n", start) + 3
    return script[start:end]


def _fake_docker(tmp: Path, body: str) -> Path:
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "docker"
    exe.write_text(f'#!/bin/bash\necho "docker $*" >> "{tmp}/calls.txt"\n{body}\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    (tmp / "calls.txt").write_text("")
    return bin_dir


def _bash(script: str, bin_dir: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(bin_dir), **env},
        capture_output=True,
        text=True,
        check=False,
    )


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t.invalid",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    )
    return done.stdout.strip()


def _market_tree(tmp: Path) -> Path:
    tree = tmp / "omnimarket"
    tree.mkdir()
    _git(tree, "init", "-q", "-b", "dev")
    (tree / "Dockerfile.runtime").write_text("ARG INFRA_RUNTIME_IMAGE\n")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-qm", "market")
    return tree


# --- AC1: the slot route builds the composed image before the identity check ---------


def test_the_slot_route_layers_omnimarket_over_the_slot_runtime_before_it_starts() -> (
    None
):
    verify = VERIFY_SH.read_text(encoding="utf-8")
    assert "--composed-omnimarket" in verify
    call = verify.index('if [[ -n "${COMPOSED_OMNIMARKET}" ]]; then')
    assert verify.index("build --progress=plain") < call, "the slot images exist first"
    assert call < verify.index("up -d --no-deps --no-build"), "built before it starts"
    assert call < verify.index('log "rendering and verifying'), "before the render"
    # inside the pool build lock: a composed layer is a build
    assert call < verify.index("\npool_build_lock_release\n")


def test_the_pool_hands_the_omnimarket_tree_to_the_slot_route() -> None:
    prove = PROVE_SH.read_text(encoding="utf-8")
    build = prove[prove.index("\nbuild)") :]
    slot_branch = build[: build.index("export DEPLOY_SOURCE_REFS_OUT")]
    assert '--composed-omnimarket "$R/omnimarket"' in slot_branch
    assert "market_under_test" in slot_branch
    assert "slot_composed_containers" in slot_branch


def _composed_block() -> str:
    """The route's composed-layer block, as the script spells it."""
    text = VERIFY_SH.read_text(encoding="utf-8")
    start = text.index('if [[ -n "${COMPOSED_OMNIMARKET}" ]]; then')
    return text[start : text.index("\nfi\n", start) + 4]


def _run_block(
    tmp_path: Path, market: Path, bin_dir: Path
) -> subprocess.CompletedProcess[str]:
    script = (
        "set -euo pipefail\n"
        'fail() { echo "FAIL $*" >&2; exit "$1"; }\n'
        "EXIT_BUILD_FAILED=10\n"
        f"COMPOSE_PROJECT={PROJECT}\nCOMPOSED_OMNIMARKET={market}\n"
        f"SLOT_BUILD_SERVICES=({' '.join(SERVICES)})\n" + _composed_block()
    )
    return _bash(script, bin_dir)


_BUILD_BODY = 'if [ "$1" = build ]; then cat > "${TMP_TAR:-/dev/null}"; fi'


def test_the_composed_layer_replaces_every_slot_service_image(tmp_path: Path) -> None:
    market = _market_tree(tmp_path)
    commit = _git(market, "rev-parse", "HEAD")
    bin_dir = _fake_docker(tmp_path, f"{_BUILD_BODY}\nexit 0")
    done = _run_block(tmp_path, market, bin_dir)
    assert done.returncode == 0, done.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    build = next(c for c in calls if c.startswith("docker build"))
    assert f"INFRA_RUNTIME_IMAGE={PROJECT}-omninode-runtime:latest" in build
    assert f"OMNIMARKET_VCS_REF={commit}" in build
    # teardown finds the image by the project label, so the layer carries it
    assert f"--label com.docker.compose.project={PROJECT}" in build
    tags = [c for c in calls if c.startswith("docker tag ")]
    assert tags == [
        f"docker tag {PROJECT}-omnimarket-runtime:latest {PROJECT}-{svc}:latest"
        for svc in SERVICES
    ]
    assert calls.index(build) < calls.index(tags[0]), "built before it is tagged"


def test_the_layer_builds_the_commit_and_never_a_working_tree(tmp_path: Path) -> None:
    market = _market_tree(tmp_path)
    (market / "Dockerfile.runtime").write_text("uncommitted edit\n")
    bin_dir = _fake_docker(tmp_path, f"{_BUILD_BODY}\nexit 0")
    done = _bash(
        "set -euo pipefail\n"
        'fail() { exit "$1"; }\nEXIT_BUILD_FAILED=10\n'
        f"COMPOSE_PROJECT={PROJECT}\nCOMPOSED_OMNIMARKET={market}\n"
        f"SLOT_BUILD_SERVICES=({' '.join(SERVICES)})\n" + _composed_block(),
        bin_dir,
        TMP_TAR=str(tmp_path / "ctx.tar"),
    )
    assert done.returncode == 0, done.stderr
    with tarfile.open(tmp_path / "ctx.tar") as ctx:
        member = ctx.extractfile("Dockerfile.runtime")
        assert member is not None
        assert member.read() == b"ARG INFRA_RUNTIME_IMAGE\n"


def test_a_tree_that_is_not_a_worktree_is_refused_before_any_tag(
    tmp_path: Path,
) -> None:
    bin_dir = _fake_docker(tmp_path, f"{_BUILD_BODY}\nexit 0")
    not_a_tree = tmp_path / "plain"
    not_a_tree.mkdir()
    done = _run_block(tmp_path, not_a_tree, bin_dir)
    assert done.returncode == 10, done.stderr
    assert "docker tag" not in (tmp_path / "calls.txt").read_text()


def test_a_slot_service_with_no_built_image_is_not_silently_skipped(
    tmp_path: Path,
) -> None:
    market = _market_tree(tmp_path)
    bin_dir = _fake_docker(
        tmp_path,
        f"""{_BUILD_BODY}
if [ "$1 $2" = "image inspect" ] && [ "${{@: -1}}" = "{PROJECT}-runtime-worker:latest" ]; then exit 1; fi
exit 0""",
    )
    done = _run_block(tmp_path, market, bin_dir)
    assert done.returncode == 10, done.stderr
    assert "runtime-worker" in done.stderr


def _containers_host(tmp: Path, on_composed: list[str], others: list[str]) -> Path:
    rows = {
        **dict.fromkeys(on_composed, "sha256:composed"),
        **dict.fromkeys(others, "sha256:infra"),
    }
    table = "\n".join(f"{name} {image}" for name, image in rows.items())
    return _fake_docker(
        tmp,
        f"""case "$1" in
  image) echo sha256:composed ;;
  ps) cut -d' ' -f1 <<'T'
{table}
T
  ;;
  inspect) c="${{@: -1}}"; grep "^$c " <<'T' | cut -d' ' -f2
{table}
T
  ;;
esac""",
    )


def _containers(tmp: Path, bin_dir: Path) -> subprocess.CompletedProcess[str]:
    work = tmp / "work"
    work.mkdir()
    script = (
        f"W={work}; P={PROJECT}; M={KERNEL_M}; E={KERNEL_E}\n"
        + _function(PROVE_SH, "slot_composed_containers")
        + "slot_composed_containers\n"
    )
    return _bash(script, bin_dir)


def test_the_proof_reads_the_slot_containers_that_run_the_composed_image(
    tmp_path: Path,
) -> None:
    on = [KERNEL_M, KERNEL_E, WORKER]
    done = _containers(tmp_path, _containers_host(tmp_path, on, ["postgres-x"]))
    assert done.returncode == 0, done.stdout + done.stderr
    listed = (tmp_path / "work" / "composed-containers").read_text().split()
    assert sorted(listed) == sorted(on), "only containers on the composed image"


@pytest.mark.parametrize("kernel", [KERNEL_M, KERNEL_E])
def test_a_slot_whose_kernel_is_not_on_the_composed_image_names_no_list(
    tmp_path: Path, kernel: str
) -> None:
    on = [c for c in (KERNEL_M, KERNEL_E, WORKER) if c != kernel]
    done = _containers(tmp_path, _containers_host(tmp_path, on, [kernel]))
    assert done.returncode == 1, done.stdout
    assert "lacks the kernel" in done.stdout
    assert not (tmp_path / "work" / "composed-containers").exists()


# --- AC2: a dead holder's lease is reclaimed by the pool's own rule -------------------


@pytest.fixture
def reaper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(TEARDOWN.parent))
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    spec = importlib.util.spec_from_file_location("slot_route_omn22863", TEARDOWN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plant_dead_holder(reaper: Any, root: Path) -> None:
    record = reaper.claim(
        root, 1, "dead-run", "standalone-1695065", str(reaper.STAGING_PARENT / "slot-1")
    )
    record["expires"] = time.time() - 1
    record["owner_pid"] = 99999999
    record["owner_start"] = "gone"
    reaper.write_record(root, 1, record)


def _teardown_of_a_destroyed_slot(
    reaper: Any, staging: Path, *, proof: bool, gateway: bool
) -> Any:
    """The slot after its gateway was offboarded and every container removed."""
    state = staging / "tenant-state"
    state.mkdir(parents=True)
    (state / "credential.env").write_text("placeholder")
    if proof:
        (state / "credential.env.revoked.json").write_text("{}")
    commands: list[list[str]] = []

    def runner(argv: list[str], env: Any) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        if argv[:3] == ["docker", "ps", "-a"]:
            return subprocess.CompletedProcess(argv, 0, "abc\n" if gateway else "", "")
        if argv[:2] == ["docker", "ps"]:  # a shared server, looked up by label
            return subprocess.CompletedProcess(argv, 0, "shared-server\n", "")
        if "revoke" in argv:
            return subprocess.CompletedProcess(
                argv, 1, "", "FATAL: no running onex-api container"
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    td = reaper._Teardown(reaper.selectors_for(reaper.policy.SLOTS[1]), runner, {})
    td.commands = commands
    return td


def test_a_dead_holder_lease_is_reclaimed_and_the_next_run_takes_the_slot(
    reaper: Any, tmp_path: Path
) -> None:
    root = tmp_path / "leases"
    root.mkdir()
    _plant_dead_holder(reaper, root)
    with pytest.raises(reaper.LeaseRefusedError):
        reaper.claim(
            root, 1, "next-run", "next", str(reaper.STAGING_PARENT / "slot-1")
        )  # the refusal measured on the lab host, before the reaper acts
    staging = tmp_path / "slot-1"
    td = _teardown_of_a_destroyed_slot(reaper, staging, proof=True, gateway=False)

    def teardown(slot: int, path: str) -> bool:
        planned = {
            "containers": [],
            "groups": [],
            "topics": [],
            "volumes": [],
            "images": [],
        }
        td.destroy(planned, staging)
        return not td.errors

    verdict = reaper.reap_slot(root, 1, teardown=teardown)
    assert verdict["verdict"] == "REAPED", (verdict, td.errors)
    assert not staging.exists()
    taken = reaper.claim(
        root, 1, "next-run", "next", str(reaper.STAGING_PARENT / "slot-1")
    )
    assert taken["holder"] == "next"


def test_without_the_offboard_proof_the_dead_holder_is_still_not_reclaimed(
    reaper: Any, tmp_path: Path
) -> None:
    root = tmp_path / "leases"
    root.mkdir()
    _plant_dead_holder(reaper, root)
    staging = tmp_path / "slot-1"
    td = _teardown_of_a_destroyed_slot(reaper, staging, proof=False, gateway=False)

    def teardown(slot: int, path: str) -> bool:
        td.destroy(
            {"containers": [], "groups": [], "topics": [], "volumes": [], "images": []},
            staging,
        )
        return not td.errors

    assert reaper.reap_slot(root, 1, teardown=teardown)["verdict"] == "INCOMPLETE"
    assert staging.exists(), "the credential is kept for a rerun"
    with pytest.raises(reaper.LeaseRefusedError):
        reaper.claim(root, 1, "next-run", "next", str(reaper.STAGING_PARENT / "slot-1"))


def test_a_present_gateway_is_still_offboarded_even_with_an_old_proof(
    reaper: Any, tmp_path: Path
) -> None:
    staging = tmp_path / "slot-1"
    td = _teardown_of_a_destroyed_slot(reaper, staging, proof=True, gateway=True)
    assert td.revoke_tenant(staging) is False
    assert any("revoke" in c for c in td.commands), "the live gateway is asked"
    assert td.errors


def test_the_reaper_snapshot_carries_the_migrations_its_database_drop_scans(
    reaper: Any, tmp_path: Path
) -> None:
    out = reaper.install_reaper(tmp_path / "prefix", tmp_path / "units", enable=False)
    release = Path(out["helper"]).parents[2]
    shipped = {p.name for p in (release / "docker/migrations/forward").glob("*.sql")}
    tracked = {p.name for p in (ROOT / "docker/migrations/forward").glob("*.sql")}
    assert shipped and shipped == tracked
