# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot refuses runtime contracts mounted from another revision than the image (OMN-25856).

The dev runtime services bind-mount ``../contracts:/app/contracts:ro`` from the
host clone over the image copy. A queued build of an older ref (0f84c5c8d)
booted with the clone's newer contracts (8d75451d6: ``boot_subscriptions.policy:
gate_required``) and failed. The image stamps a sha256 manifest of its own
``contracts/runtime`` files at build; the entrypoint's guard compares the files
the container sees to it and aborts with ``CONTRACTS_REVISION_MISMATCH`` before
the kernel starts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_contracts_revision_guard as guard
from omnibase_infra.runtime import entrypoint_preflight

pytestmark = [pytest.mark.unit]

_OLD = 'boot_subscriptions:\n  policy: "in_use"\n'
_NEW = 'boot_subscriptions:\n  policy: "gate_required"\n'
_DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"


def _runtime(root: Path, files: dict[str, str]) -> Path:
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    for name, text in files.items():
        (runtime / name).write_text(text)
    return runtime


def _stamp(runtime: Path, stamp: Path) -> None:
    """The manifest the Dockerfile writes: ``sha256sum`` lines, ``./``-relative."""
    stamp.write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(runtime)}\n"
            for p in sorted(runtime.rglob("*"))
            if p.is_file()
        )
    )


def _env(runtime: Path, stamp: Path) -> dict[str, str]:
    return {
        "ONEX_RUNTIME_CONTRACTS_DIR": str(runtime),
        "ONEX_CONTRACTS_STAMP_PATH": str(stamp),
    }


def test_matching_contracts_pass(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "image", {"runtime_config.yaml": _OLD})
    stamp = tmp_path / "stamp.sha256"
    _stamp(runtime, stamp)

    assert guard.refuse_mismatched_contracts(_env(runtime, stamp)) == 0


def test_newer_mounted_contracts_on_older_image_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The 0f84c5c8d image with the 8d75451d6 contracts."""
    image = _runtime(tmp_path / "image", {"runtime_config.yaml": _OLD})
    stamp = tmp_path / "stamp.sha256"
    _stamp(image, stamp)
    mounted = _runtime(tmp_path / "host", {"runtime_config.yaml": _NEW})

    rc = guard.refuse_mismatched_contracts(_env(mounted, stamp))

    assert rc == 1
    err = capsys.readouterr().err
    assert "CONTRACTS_REVISION_MISMATCH" in err
    assert "runtime_config.yaml" in err


def test_a_mounted_file_the_image_never_shipped_is_refused(tmp_path: Path) -> None:
    image = _runtime(tmp_path / "image", {"runtime_config.yaml": _OLD})
    stamp = tmp_path / "stamp.sha256"
    _stamp(image, stamp)
    mounted = _runtime(
        tmp_path / "host", {"runtime_config.yaml": _OLD, "added_later.yaml": "x: 1\n"}
    )

    assert guard.refuse_mismatched_contracts(_env(mounted, stamp)) == 1


def test_a_file_the_mount_does_not_carry_is_not_judged(tmp_path: Path) -> None:
    image = _runtime(
        tmp_path / "image", {"runtime_config.yaml": _OLD, "core_only.yaml": "y: 1\n"}
    )
    stamp = tmp_path / "stamp.sha256"
    _stamp(image, stamp)
    mounted = _runtime(tmp_path / "host", {"runtime_config.yaml": _OLD})

    assert guard.refuse_mismatched_contracts(_env(mounted, stamp)) == 0


def test_an_unset_stamp_path_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert guard.refuse_mismatched_contracts({}) == 1
    assert "CONTRACTS_REVISION_MISMATCH" in capsys.readouterr().err


def test_a_missing_stamp_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _runtime(tmp_path / "image", {"runtime_config.yaml": _OLD})

    rc = guard.refuse_mismatched_contracts(_env(runtime, tmp_path / "absent.sha256"))

    assert rc == 1
    assert "CONTRACTS_REVISION_MISMATCH" in capsys.readouterr().err


def test_run_preflight_refuses_a_mismatch_before_any_other_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _runtime(tmp_path / "image", {"runtime_config.yaml": _OLD})
    stamp = tmp_path / "stamp.sha256"
    _stamp(image, stamp)
    mounted = _runtime(tmp_path / "host", {"runtime_config.yaml": _NEW})
    reached: list[str] = []
    monkeypatch.setattr(
        entrypoint_preflight,
        "stamp_all_fingerprints",
        lambda _env: reached.append("stamp") or 0,
    )

    rc = entrypoint_preflight.run_preflight(_env(mounted, stamp))

    assert rc != 0
    assert reached == []


def test_the_dockerfile_stamps_the_manifest_outside_the_mounted_path() -> None:
    dockerfile = (_DOCKER_DIR / "Dockerfile.runtime").read_text()

    assert (
        "ONEX_CONTRACTS_STAMP_PATH=/app/config/contracts_runtime.sha256" in dockerfile
    )
    assert "> /app/config/contracts_runtime.sha256" in dockerfile
