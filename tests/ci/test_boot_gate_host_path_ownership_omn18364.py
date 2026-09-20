# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18364 — the boot gate prepares its hostPath dirs with a derived owner.

WHAT BROKE. `k8s/onex-lab` binds the lane's Postgres to a static hostPath
PersistentVolume (OMN-18186, so a pod replacement stops destroying the
database). OMN-18765 then gave the pod `runAsNonRoot: true` / `runAsUser: 70`
to match the namespace's restricted Pod Security Standard. Both are correct.
Together, on a FRESH cluster, they are fatal: `hostPath` is the one volume type
the kubelet does not apply `fsGroup` to, so `type: DirectoryOrCreate` yields a
root-owned 0755 directory and uid 70 cannot write in it.

Measured 2026-09-19 from the boot gate's own diagnostics artifact
(`candidate-boot-gate-35459326286`):

    mkdir: can't create directory '/var/lib/postgresql/data/pgdata':
    Permission denied

with `Restart Count: 5`, `Back-off restarting failed container postgres`, and
the Deployment blowing its 180s `progressDeadlineSeconds`. Every
`deliver-dev-candidate-to-staging` run from 09:29Z onward that reached the boot
gate died at `error: deployment "onex-lab-postgres" exceeded its progress
deadline`. The persistent lab lane was unaffected because its directory already
existed owned by 70 — which is exactly why this went unnoticed.

The namespace enforces `restricted`, so the manifest cannot chown its own
volume (a root initContainer is refused at admission). The directory must
arrive already owned. These tests pin that the owner is DERIVED from the render
and that every unresolvable shape fails closed rather than silently preparing
nothing.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "boot_gate.py"
_spec = importlib.util.spec_from_file_location("boot_gate_omn18364", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
boot_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boot_gate)


HOST_PATH = "/var/lib/onex-lab/postgres-data"


def _persistent_volume(claim: str | None = "onex-lab-postgres-data") -> dict[str, Any]:
    spec: dict[str, Any] = {
        "hostPath": {"path": HOST_PATH, "type": "DirectoryOrCreate"}
    }
    if claim is not None:
        spec["claimRef"] = {"name": claim, "namespace": "onex-dev"}
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolume",
        "metadata": {"name": "onex-lab-postgres-data"},
        "spec": spec,
    }


def _claim() -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {"name": "onex-lab-postgres-data", "namespace": "onex-dev"},
        "spec": {"volumeName": "onex-lab-postgres-data"},
    }


def _deployment(
    name: str = "onex-lab-postgres",
    *,
    run_as_user: int | None = 70,
    run_as_group: int | None = 70,
    fs_group: int | None = 70,
    claim: str = "onex-lab-postgres-data",
) -> dict[str, Any]:
    security: dict[str, Any] = {}
    if run_as_user is not None:
        security["runAsUser"] = run_as_user
    if run_as_group is not None:
        security["runAsGroup"] = run_as_group
    if fs_group is not None:
        security["fsGroup"] = fs_group
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": "onex-dev"},
        "spec": {
            "template": {
                "spec": {
                    "securityContext": security,
                    "volumes": [
                        {"name": "data", "persistentVolumeClaim": {"claimName": claim}}
                    ],
                }
            }
        },
    }


def _lane() -> list[dict[str, Any]]:
    return [_persistent_volume(), _claim(), _deployment()]


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def test_owner_is_derived_from_the_rendered_pod_security_context() -> None:
    """The live shape: uid/gid come off the pod that mounts the claim."""
    assert boot_gate.resolve_host_path_plans(_lane()) == [(HOST_PATH, 70, 70, "0700")]


def test_a_changed_uid_in_the_render_moves_the_prepared_ownership() -> None:
    """Derivation, not restatement.

    If the number were hardcoded in the gate, this case would still return 70
    and the pod would crash-loop exactly as it did before the fix.
    """
    documents = [
        _persistent_volume(),
        _claim(),
        _deployment(run_as_user=1001, run_as_group=1001),
    ]
    assert boot_gate.resolve_host_path_plans(documents) == [
        (HOST_PATH, 1001, 1001, "0700")
    ]


def test_group_falls_back_to_fs_group_when_run_as_group_is_absent() -> None:
    documents = [
        _persistent_volume(),
        _claim(),
        _deployment(run_as_group=None, fs_group=70),
    ]
    assert boot_gate.resolve_host_path_plans(documents) == [(HOST_PATH, 70, 70, "0700")]


def test_claim_is_resolved_through_the_pvc_when_the_pv_names_no_claim_ref() -> None:
    documents = [_persistent_volume(claim=None), _claim(), _deployment()]
    assert boot_gate.resolve_host_path_plans(documents) == [(HOST_PATH, 70, 70, "0700")]


# ---------------------------------------------------------------------------
# fail-closed controls
# ---------------------------------------------------------------------------
def test_empty_render_fails_closed() -> None:
    """A step that prepared nothing must not look like a step that worked."""
    with pytest.raises(boot_gate.HostPathOwnershipError, match="no hostPath"):
        boot_gate.resolve_host_path_plans([])


def test_host_path_volume_no_workload_mounts_fails_closed() -> None:
    with pytest.raises(boot_gate.HostPathOwnershipError, match="no Deployment"):
        boot_gate.resolve_host_path_plans([_persistent_volume(), _claim()])


def test_pod_without_run_as_user_fails_closed() -> None:
    documents = [
        _persistent_volume(),
        _claim(),
        _deployment(run_as_user=None),
    ]
    with pytest.raises(boot_gate.HostPathOwnershipError, match="no runAsUser"):
        boot_gate.resolve_host_path_plans(documents)


def test_pod_without_any_group_fails_closed() -> None:
    documents = [
        _persistent_volume(),
        _claim(),
        _deployment(run_as_group=None, fs_group=None),
    ]
    with pytest.raises(
        boot_gate.HostPathOwnershipError, match="neither runAsGroup nor fsGroup"
    ):
        boot_gate.resolve_host_path_plans(documents)


def test_two_workloads_disagreeing_about_the_uid_fails_closed() -> None:
    documents = [
        _persistent_volume(),
        _claim(),
        _deployment(name="a", run_as_user=70, run_as_group=70),
        _deployment(name="b", run_as_user=1001, run_as_group=1001),
    ]
    with pytest.raises(boot_gate.HostPathOwnershipError, match="two uids"):
        boot_gate.resolve_host_path_plans(documents)


# ---------------------------------------------------------------------------
# execution + readback
# ---------------------------------------------------------------------------
class _Runner:
    """Records commands; answers `stat` with whatever ownership it is given."""

    def __init__(self, stat_output: str, *, install_rc: int = 0, stat_rc: int = 0):
        self.stat_output = stat_output
        self.install_rc = install_rc
        self.stat_rc = stat_rc
        self.calls: list[list[str]] = []

    def __call__(
        self, node: str, command: list[str]
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        if command[0] == "install":
            return subprocess.CompletedProcess(command, self.install_rc, "", "")
        return subprocess.CompletedProcess(
            command, self.stat_rc, self.stat_output + "\n", ""
        )


def _render_file(tmp_path: Path) -> Path:
    path = tmp_path / "onex-lab-render.yaml"
    path.write_text(yaml.safe_dump_all(_lane(), sort_keys=False))
    return path


def test_prepare_installs_the_directory_owned_and_reads_it_back(tmp_path: Path) -> None:
    runner = _Runner("70:70:700")
    rc = boot_gate.prepare_host_paths(
        _render_file(tmp_path), "onex-lab-boot-gate-control-plane", runner=runner
    )
    assert rc == 0
    assert runner.calls[0] == [
        "install",
        "-d",
        "-m",
        "0700",
        "-o",
        "70",
        "-g",
        "70",
        HOST_PATH,
    ]
    assert runner.calls[1] == ["stat", "-c", "%u:%g:%a", HOST_PATH]


def test_prepare_fails_when_the_readback_disagrees(tmp_path: Path) -> None:
    """`install` exiting 0 is not proof the directory is owned as asked.

    A filesystem that accepts the command and refuses the chown would
    otherwise hand the pod the same permission error minutes later, which is
    the whole failure this step removes.
    """
    runner = _Runner("0:0:755")
    rc = boot_gate.prepare_host_paths(_render_file(tmp_path), "node", runner=runner)
    assert rc == 1


def test_prepare_fails_when_the_install_fails(tmp_path: Path) -> None:
    runner = _Runner("70:70:700", install_rc=1)
    rc = boot_gate.prepare_host_paths(_render_file(tmp_path), "node", runner=runner)
    assert rc == 1


def test_prepare_fails_when_the_readback_command_fails(tmp_path: Path) -> None:
    runner = _Runner("", stat_rc=1)
    rc = boot_gate.prepare_host_paths(_render_file(tmp_path), "node", runner=runner)
    assert rc == 1


def test_prepare_reports_an_unresolvable_render_rather_than_preparing_nothing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    runner = _Runner("70:70:700")
    assert boot_gate.prepare_host_paths(path, "node", runner=runner) == 1
    assert runner.calls == []


# ---------------------------------------------------------------------------
# the seam cannot be reached from argv
# ---------------------------------------------------------------------------
def test_no_entrypoint_exposes_an_option_to_skip_or_force_the_preparation() -> None:
    """No `--force`, no `--skip`, no way to assert ownership from the caller.

    Same posture as the prod gate's health seam: the runner is a keyword-only
    argument on the function, so adding a CLI escape hatch is a red test rather
    than a review catch.
    """
    source = _MODULE_PATH.read_text()
    prep_block = source.split('"prepare-host-paths"', 1)[1].split(
        "wait = sub.add_parser", 1
    )[0]
    declared = {
        line.split('"')[1]
        for line in prep_block.splitlines()
        if "add_argument(" in line and '"' in line
    }
    assert declared == {"--render", "--node", "--mode"}, declared
    for forbidden in ("--force", "--skip", "--uid", "--gid", "--assume-owned"):
        assert forbidden not in prep_block
