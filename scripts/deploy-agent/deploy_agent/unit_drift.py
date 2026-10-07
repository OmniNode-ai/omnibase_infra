# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Detect installed units that no longer match their tracked source (OMN-20037).

A tracked unit's PATH was fixed while its installed copy stayed stale, leaving
the agent failing every job for roughly two days. Report that gap explicitly;
self-update may repair only the agent's own plain-file unit at a job boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

INDETERMINATE_REASON = "no unit-drift provider is configured"


class EnumUnitStatus(StrEnum):
    """Observed relationship between a declared unit and its installed copy."""

    OK = "OK"
    DRIFT = "DRIFT"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ModelUnitEntry(BaseModel):
    """One host-scoped unit and its source of expected bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    tracked: str
    installed: str
    hosts: list[str]
    render_script: str | None = None
    render_args: list[str] = Field(default_factory=list)

    @field_validator("tracked", "render_script")
    @classmethod
    def repo_relative(cls, value: str | None) -> str | None:
        """Keep declared sources relative to the repository root."""
        if value is not None and (
            not value or Path(value).is_absolute() or ".." in Path(value).parts
        ):
            raise ValueError("source paths must be repo-root-relative")
        return value


class ModelUnitResult(BaseModel):
    """A read-only observation; absent hashes mean no bytes were read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    status: EnumUnitStatus
    installed_path: Path
    tracked_sha256: str | None = None
    installed_sha256: str | None = None


class _ModelManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    units: list[ModelUnitEntry]


def load_manifest(path: Path) -> list[ModelUnitEntry]:
    """Validate the whole manifest, including its top-level keys."""
    return _ModelManifest.model_validate(yaml.safe_load(path.read_text())).units


def _hostname(hostname: str) -> str:
    return hostname.lower().removesuffix(".local")


def _installed_path(
    entry: ModelUnitEntry, home: Path, *, resolve_registry: bool = True
) -> Path:
    if entry.installed == "~":
        return home
    if entry.installed.startswith("~/"):
        return home / entry.installed[2:]
    if entry.installed.startswith("{registry_root}/") and resolve_registry:
        registry_root = os.environ.get("UNIT_DRIFT_REGISTRY_ROOT", "")
        if not registry_root or not Path(registry_root).is_absolute():
            raise ValueError(
                "registry-bound installed paths require absolute UNIT_DRIFT_REGISTRY_ROOT"
            )
        return Path(registry_root) / entry.installed.removeprefix("{registry_root}/")
    return Path(entry.installed)


def _tracked_bytes(entry: ModelUnitEntry, repo_root: Path, home: Path) -> bytes:
    if entry.render_script is None:
        return (repo_root / entry.tracked).read_bytes()
    return subprocess.run(
        [
            "bash",
            str(repo_root / entry.render_script),
            *(arg.replace("{home}", str(home)) for arg in entry.render_args),
        ],
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout


def check_units(
    entries: Sequence[ModelUnitEntry],
    *,
    repo_root: Path,
    hostname: str,
    home: Path,
    overrides: Mapping[str, Path] | None = None,
) -> list[ModelUnitResult]:
    """Compare raw bytes on this host without modifying either copy."""
    results: list[ModelUnitResult] = []
    for entry in entries:
        applicable = _hostname(hostname) in {_hostname(host) for host in entry.hosts}
        installed = (overrides or {}).get(entry.name)
        if installed is None:
            installed = _installed_path(entry, home, resolve_registry=applicable)
        if not applicable:
            results.append(
                ModelUnitResult(
                    name=entry.name,
                    status=EnumUnitStatus.NOT_APPLICABLE,
                    installed_path=installed,
                )
            )
            continue
        tracked = _tracked_bytes(entry, repo_root, home)
        try:
            actual = installed.read_bytes()
        except FileNotFoundError:
            actual = None
        status = (
            EnumUnitStatus.MISSING
            if actual is None
            else EnumUnitStatus.OK
            if actual == tracked
            else EnumUnitStatus.DRIFT
        )
        results.append(
            ModelUnitResult(
                name=entry.name,
                status=status,
                installed_path=installed,
                tracked_sha256=hashlib.sha256(tracked).hexdigest()[:12],
                installed_sha256=(
                    hashlib.sha256(actual).hexdigest()[:12]
                    if actual is not None
                    else None
                ),
            )
        )
    return results


def has_drift(results: Sequence[ModelUnitResult]) -> bool:
    """Missing declared units are drift too."""
    return any(
        result.status in (EnumUnitStatus.DRIFT, EnumUnitStatus.MISSING)
        for result in results
    )


def own_unit_name_from_cgroup(text: str) -> str | None:
    """Read the innermost service from the unified cgroup membership."""
    for line in text.splitlines():
        if line.startswith("0::"):
            return next(
                (
                    part
                    for part in reversed(line[3:].split("/"))
                    if part.endswith(".service")
                ),
                None,
            )
    return None


def sync_own_unit(
    entries: Sequence[ModelUnitEntry],
    *,
    unit_name: str | None,
    repo_root: Path,
    hostname: str,
    home: Path,
    runner: Callable[[list[str]], int],
    stamp: str,
) -> bool:
    """Back up and atomically replace our drifting unit, then reload, never restart."""
    if unit_name is None:
        return False
    entry = next((entry for entry in entries if entry.name == unit_name), None)
    if entry is None:
        return False
    result = check_units([entry], repo_root=repo_root, hostname=hostname, home=home)[0]
    if result.status is not EnumUnitStatus.DRIFT:
        return False
    if entry.render_script is not None:
        raise ValueError("render entries cannot be installed by sync_own_unit")
    tracked = _tracked_bytes(entry, repo_root, home)
    installed = result.installed_path
    shutil.copy2(installed, f"{installed}.bak-unit-drift-{stamp}")
    with tempfile.NamedTemporaryFile(
        dir=installed.parent, prefix=f".{installed.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(tracked)
            handle.flush()
            shutil.copymode(installed, temporary)
            os.replace(temporary, installed)  # noqa: PTH105 -- explicit atomic replacement
        finally:
            temporary.unlink(missing_ok=True)
    if runner(["systemctl", "--user", "daemon-reload"]) != 0:
        raise RuntimeError("systemctl --user daemon-reload failed")
    return True


def report(results: Sequence[ModelUnitResult], hostname: str) -> dict[str, object]:
    """Return a JSON-serializable observation, including non-applicable entries."""
    return {
        "hostname": hostname,
        "drift": has_drift(results),
        "units": [result.model_dump(mode="json") for result in results],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Report drift for the manifest; a missing installed file also exits one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--hostname", default=socket.gethostname())
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--installed-override", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    overrides: dict[str, Path] = {}
    for override in args.installed_override:
        name, separator, path = override.partition("=")
        if not separator or not name or not path:
            parser.error("--installed-override must be NAME=PATH")
        overrides[name] = Path(path)
    results = check_units(
        load_manifest(args.manifest),
        repo_root=args.repo_root,
        hostname=args.hostname,
        home=args.home,
        overrides=overrides,
    )
    if args.json:
        print(json.dumps(report(results, args.hostname)))
    else:
        for result in results:
            print(
                f"UNIT-DRIFT {result.status.value} {result.name} "
                f"installed={result.installed_path}"
            )
        counts = {
            status: sum(r.status == status for r in results)
            for status in EnumUnitStatus
        }
        print(
            f"UNIT-DRIFT-SUMMARY drift={counts[EnumUnitStatus.DRIFT]} "
            f"missing={counts[EnumUnitStatus.MISSING]} ok={counts[EnumUnitStatus.OK]}"
        )
    return int(has_drift(results))


if __name__ == "__main__":
    raise SystemExit(main())
