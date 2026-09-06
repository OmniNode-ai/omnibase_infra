# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Build a local, dev-only Market replacement without deploying it [OMN-17991].

The base and all source inputs are immutable. Offline build tools must already
exist in the invoking uv environment; this helper never bootstraps dependencies,
calls the ordinary deployment path, publishes an image, or replaces shared tags.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import sys
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from uuid import uuid4

from compute_workspace_provenance import _content_parity_diff, _hash_tree
from deploy_source_ref import (
    RepoRefSelection,
    _assert_owned_clean_target,
    _git,
    _resolve_commit,
    validate_immutable_selections,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scoped_effects_deploy import _IMAGE_CONFIG_FIELDS, _run_command
from scoped_effects_plan import ImageId, SourceRepo, SourceSha

CommandRunner = Callable[[list[str], str | None, int], str]
SIBLINGS = ("omnibase_core", "omnibase_compat", "omnimarket")


class ModelEffectsCandidatePlan(BaseModel):
    """Only a derived local artifact is authorized, never a service mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1"]
    ticket_id: Annotated[str, Field(pattern=r"^OMN-[1-9][0-9]*$")]
    reason: Annotated[str, Field(min_length=10, max_length=1000)]
    base_image_id: ImageId
    source_root: Path
    source_pins: dict[SourceRepo, SourceSha]
    infra_source_root: Path
    infra_source_sha: SourceSha

    @model_validator(mode="after")
    def check_boundary(self) -> Self:
        if set(self.source_pins) != set(SIBLINGS):
            raise ValueError("source_pins must cover exactly the three staged siblings")
        if len(self.reason.strip()) < 10:
            raise ValueError("reason must contain a concrete build justification")
        for path in (self.source_root, self.infra_source_root):
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("source paths must be absolute and traversal-free")
        return self


def validate_sources(plan: ModelEffectsCandidatePlan) -> dict[str, str]:
    """Read actual clean source bytes; no fetch, checkout, reset or clean."""
    selections = (
        *validate_immutable_selections(
            {name: str(plan.source_root / name) for name in SIBLINGS},
            dict(plan.source_pins),
        ),
        RepoRefSelection(
            "omnibase_infra", plan.infra_source_root.resolve(), plan.infra_source_sha
        ),
    )
    if len({row.path for row in selections}) != len(selections):
        raise ValueError("each source must have its own distinct clone")
    digests: dict[str, str] = {}
    for row in selections:
        _assert_owned_clean_target(row)
        if Path(_git(row.path, "rev-parse", "--show-toplevel")).resolve() != row.path:
            raise ValueError(f"source path is not a clone root: {row.repo}")
        if (
            _resolve_commit(row.path, row.ref) != row.ref
            or _git(row.path, "rev-parse", "HEAD") != row.ref
        ):
            raise ValueError(
                f"source clone is not at its selected immutable SHA: {row.repo}"
            )
        package = row.path / "src" / row.repo
        if not package.is_dir() or any(
            path.is_symlink() for path in package.rglob("*")
        ):
            raise ValueError(f"source package absent or symlinked: {row.repo}")
        digests[row.repo] = _hash_tree(package)
    return digests


def _image(runner: CommandRunner, reference: str, expected_id: str) -> dict[str, Any]:
    values = json.loads(runner(["docker", "image", "inspect", reference], None, 30))
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
        or values[0].get("Id") != expected_id
    ):
        raise RuntimeError("local base/candidate image identity mismatch")
    value: dict[str, Any] = values[0]
    return value


def _probe(runner: CommandRunner, image: str) -> dict[str, Any]:
    probe = (
        Path(__file__).with_name("scoped_image_probe.py").read_text(encoding="utf-8")
    )
    script = (
        "import importlib.metadata, json\nfrom pathlib import Path\n"
        f"namespace = {{'__name__': 'candidate_image_probe'}}\nexec({probe!r}, namespace)\n"
        "evidence = namespace['collect_image_evidence']()\n"
        "evidence['package_digests'] = {name: namespace['tree_digest'](Path(str("
        "importlib.metadata.distribution(name).locate_file(name)))) "
        "for name in ('omnibase_core', 'omnibase_compat', 'omnimarket', 'omnibase_infra')}\n"
        "print(json.dumps(evidence, sort_keys=True))\n"
    )
    result = json.loads(
        runner(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--network",
                "none",
                "--read-only",
                "--entrypoint",
                "uv",
                image,
                "run",
                "--no-project",
                "--no-sync",
                "--python",
                "/app/.venv/bin/python",
                "python",
                "-B",
                "-",
            ],
            script,
            120,
        )
    )
    if not isinstance(result, dict) or not isinstance(
        result.get("package_digests"), dict
    ):
        raise RuntimeError("image probe returned no package-content evidence")
    return result


def _snapshot_market(source: Path, destination: Path) -> None:
    """Snapshot only tracked regular files, never ambient secrets or symlinks."""
    for name in _git(source, "ls-files", "-z").split("\0"):
        if not name:
            continue
        relative = Path(name)
        path = source / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(
                "Market source snapshot contains an unsupported tracked path"
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def validate_wheel(wheel: Path, source_package: Path, extracted: Path) -> str:
    """Reject foreign payloads, then prove wheel package bytes equal pinned source."""
    metadata_roots: set[str] = set()
    names: set[str] = set()
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or not path.parts
                or item.filename in names
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("wheel contains an unsafe or duplicated path")
            names.add(item.filename)
            first = path.parts[0]
            is_metadata = first.startswith("omnimarket-") and first.endswith(
                ".dist-info"
            )
            if first != "omnimarket" and not is_metadata:
                raise ValueError("wheel changes files outside the Market distribution")
            if is_metadata:
                metadata_roots.add(first)
            if not item.is_dir():
                target = extracted / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
    if len(metadata_roots) != 1:
        raise ValueError("wheel must contain exactly one Market metadata directory")
    metadata_root = next(iter(metadata_roots))
    metadata = (extracted / metadata_root / "METADATA").read_text(encoding="utf-8")
    if "\nName: omnimarket\n" not in "\n" + metadata:
        raise ValueError("wheel distribution identity is not omnimarket")
    differing = _content_parity_diff(source_package, extracted / "omnimarket")
    if differing:
        raise ValueError(
            f"wheel payload differs from pinned Market source: {differing[:10]}"
        )
    return metadata_root


# Runs only within the isolated Docker build. It imports the canonical hash
# implementation, not its workspace-only main/local-install provenance policy.
_INSTALLER = r"""
import hashlib, importlib.metadata, json, os, stat, subprocess, sys
from pathlib import Path
sys.path.insert(0, '/candidate/proof-tools')
from compute_workspace_provenance import _hash_tree

root = Path('/app/.venv')
payload = Path('/candidate')
manifest = json.loads((payload / 'manifest.json').read_text())
market = importlib.metadata.distribution('omnimarket')
site = Path(market.locate_file(''))
allowed_roots = [site / 'omnimarket', site / manifest.pop('wheel_metadata_root')]
allowed_scripts = set()
if not market.files:
    raise RuntimeError('base Market has no installed file inventory')
for rel in market.files:
    path = Path(market.locate_file(rel)).absolute()
    if any(part.endswith('.dist-info') for part in rel.parts):
        allowed_roots.append(site / rel.parts[0])
    if path.parent.resolve() == (root / 'bin').resolve():
        allowed_scripts.add(path.resolve())

def unchanged_venv_digest():
    result = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if any(path == prefix or prefix in path.parents for prefix in allowed_roots):
            continue
        if path.resolve() in allowed_scripts:
            continue
        info = path.lstat()
        result.update(str(path.relative_to(root)).encode())
        result.update(str(stat.S_IMODE(info.st_mode)).encode())
        if path.is_symlink():
            result.update(b'link:' + os.readlink(path).encode())
        elif path.is_file():
            result.update(b'file:' + path.read_bytes())
        else:
            result.update(b'directory')
    return result.hexdigest()

before = unchanged_venv_digest()
subprocess.run([
    '/usr/local/bin/uv', 'pip', 'install', '--python', '/app/.venv/bin/python',
    '--offline', '--no-deps', '--no-sources', '--no-cache', '--link-mode=copy',
    '--reinstall-package', 'omnimarket', str(payload / manifest.pop('wheel_name')),
], check=True)
if unchanged_venv_digest() != before:
    raise RuntimeError('Market install changed other installed environment bytes')
for proof in manifest['proofs']:
    dist = importlib.metadata.distribution(proof['package'])
    actual = _hash_tree(Path(dist.locate_file(proof['repo'])))
    if actual != proof['staged_package_digest']:
        raise RuntimeError('installed package differs from source: ' + proof['repo'])
    proof['installed_package_digest'] = actual
    proof['status'] = 'verified'
manifest['unchanged_venv_fingerprint'] = before
Path('/app/build-provenance.json').write_text(json.dumps(manifest, sort_keys=True) + '\n')
"""


def _manifest(
    plan: ModelEffectsCandidatePlan, digests: dict[str, str]
) -> dict[str, Any]:
    return {
        "build_source": "scoped-derived",
        "base_image_id": plan.base_image_id,
        "infra_vcs_ref": plan.infra_source_sha,
        "per_repo_vcs_provenance": {
            "siblings": {
                name: {"vcs_ref": pin, "vcs_dirty": False, "vcs_branch": "HEAD"}
                for name, pin in plan.source_pins.items()
            }
        },
        "proofs": [
            {
                "repo": name,
                "package": name.replace("_", "-"),
                "staged_package_digest": digest,
                "status": "pending",
                "origin": "replacement" if name == "omnimarket" else "base-equivalence",
            }
            for name, digest in digests.items()
        ],
    }


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_candidate(
    plan: ModelEffectsCandidatePlan,
    output: Path,
    *,
    execute: bool = False,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    digests = validate_sources(plan)
    base = _image(runner, plan.base_image_id, plan.base_image_id)
    if not execute:
        return {
            "status": "PREVIEW",
            "base_image_id": plan.base_image_id,
            "source_digests": digests,
        }
    # No hidden networked backend install. uv build uses this already-provisioned
    # environment with isolation disabled, with its backend version recorded.
    try:
        backend = importlib.metadata.version("hatchling")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "offline hatchling build backend is not provisioned; no bootstrap attempted"
        ) from exc
    if output.exists() or output.is_symlink():
        raise ValueError("output directory already exists")
    output = output.resolve()
    if not output.parent.is_dir() or output.parent.stat().st_uid != os.geteuid():
        raise ValueError(
            "output must be a new directory beneath an owned existing parent"
        )
    if any(
        root == output or root in output.parents
        for root in (plan.source_root.resolve(), plan.infra_source_root.resolve())
    ):
        raise ValueError("candidate output must not be inside a source clone root")
    baseline = _probe(runner, plan.base_image_id)
    infra_ref = baseline.get("infra_source_ref")
    if (
        not isinstance(infra_ref, str)
        or not re.fullmatch(r"(?:[0-9a-f]{12}|[0-9a-f]{40})", infra_ref)
        or _resolve_commit(plan.infra_source_root, infra_ref) != plan.infra_source_sha
    ):
        raise RuntimeError(
            "baseline infra lineage differs from the selected full source SHA"
        )
    for name in ("omnibase_core", "omnibase_compat", "omnibase_infra"):
        if baseline["package_digests"].get(name) != digests[name]:
            raise RuntimeError(
                f"base installed {name} does not match its pinned source"
            )
    output.mkdir(mode=0o700)
    receipt: dict[str, Any] = {
        "schema_version": "1",
        "ticket_id": plan.ticket_id,
        "status": "REFUSED",
        "base_image_id": plan.base_image_id,
        "source_pins": dict(plan.source_pins),
        "infra_source_sha": plan.infra_source_sha,
        "reason": plan.reason,
        "started_at": datetime.now(UTC).isoformat(),
        "artifact_actions": [],
        "build_toolchain": {"python": sys.version, "hatchling": backend},
    }
    _write(output / "plan.json", plan.model_dump(mode="json"))
    try:
        context = output / "context"
        payload = context / "payload"
        payload.mkdir(parents=True)
        source = output / "market-source"
        _snapshot_market(plan.source_root / "omnimarket", source)
        runner(
            [
                "uv",
                "build",
                "--offline",
                "--no-build-isolation",
                "--no-sources",
                "--no-python-downloads",
                "--python",
                sys.executable,
                "--wheel",
                "--out-dir",
                str(payload),
                str(source),
            ],
            None,
            300,
        )
        wheels = list(payload.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("offline build must produce exactly one Market wheel")
        wheel = wheels[0]
        metadata_root = validate_wheel(
            wheel, source / "src/omnimarket", output / "wheel-proof"
        )
        if validate_sources(plan) != digests:
            raise RuntimeError("source inputs changed during wheel construction")
        manifest = _manifest(plan, digests)
        manifest.update(
            {
                "wheel_name": wheel.name,
                "wheel_metadata_root": metadata_root,
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            }
        )
        _write(payload / "manifest.json", manifest)
        (payload / "install_market.py").write_text(_INSTALLER, encoding="utf-8")
        proof_tools = payload / "proof-tools"
        proof_tools.mkdir()
        for name in ("compute_workspace_provenance.py", "resolve_workspace_pins.py"):
            shutil.copy2(Path(__file__).with_name(name), proof_tools / name)
        suffix = f"{plan.ticket_id.lower()}-{uuid4().hex}"
        base_alias = f"onex-effects-candidate-base:{suffix}"
        candidate_tag = f"onex-effects-candidate:{suffix}"
        for tag in (base_alias, candidate_tag):
            if runner(
                [
                    "docker",
                    "image",
                    "ls",
                    "--filter",
                    f"reference={tag}",
                    "--format",
                    "{{.ID}}",
                ],
                None,
                30,
            ).strip():
                raise RuntimeError("unique candidate artifact tag already exists")
        receipt["artifact_actions"].append(
            {"action": "tag-base", "tag": base_alias, "image_id": plan.base_image_id}
        )
        _write(output / "candidate-receipt.json", receipt)
        runner(["docker", "image", "tag", plan.base_image_id, base_alias], None, 30)
        _image(runner, base_alias, plan.base_image_id)
        (context / "Dockerfile").write_text(
            f"FROM {base_alias}\n"
            "RUN --network=none --mount=type=bind,source=payload,target=/candidate,readonly "
            "/usr/local/bin/uv run --no-project --no-sync --python /app/.venv/bin/python "
            "python -B /candidate/install_market.py\n"
            'LABEL com.omninode.build_source="scoped-derived" '
            'com.omninode.promotion_class="stability-candidate" '
            'com.omninode.non_main_lineage="true" '
            'com.omninode.allowed_lane="dev" '
            f'com.omninode.base_image_id="{plan.base_image_id}"\n',
            encoding="utf-8",
        )
        image_file = output / "candidate-image.id"
        receipt["artifact_actions"].append(
            {"action": "build-candidate", "tag": candidate_tag}
        )
        _write(output / "candidate-receipt.json", receipt)
        runner(
            [
                "docker",
                "build",
                "--pull=false",
                "--network=none",
                "--iidfile",
                str(image_file),
                "--tag",
                candidate_tag,
                str(context),
            ],
            None,
            900,
        )
        candidate_id = image_file.read_text(encoding="utf-8").strip()
        if (
            len(candidate_id) != 71
            or not candidate_id.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in candidate_id[7:])
        ):
            raise RuntimeError("build did not return an immutable candidate image ID")
        _image(runner, base_alias, plan.base_image_id)
        candidate = _image(runner, candidate_tag, candidate_id)
        expected_labels = {
            "com.omninode.build_source": "scoped-derived",
            "com.omninode.promotion_class": "stability-candidate",
            "com.omninode.non_main_lineage": "true",
            "com.omninode.allowed_lane": "dev",
            "com.omninode.base_image_id": plan.base_image_id,
        }
        labels = candidate.get("Config", {}).get("Labels", {}) or {}
        if any(labels.get(key) != value for key, value in expected_labels.items()):
            raise RuntimeError(
                "candidate lacks truthful dev-only derived lineage labels"
            )
        base_layers = base.get("RootFS", {}).get("Layers")
        layers = candidate.get("RootFS", {}).get("Layers")
        if not base_layers or not layers or layers[: len(base_layers)] != base_layers:
            raise RuntimeError(
                "candidate does not derive from the exact baseline layers"
            )
        if any(
            base.get("Config", {}).get(key) != candidate.get("Config", {}).get(key)
            for key in _IMAGE_CONFIG_FIELDS
        ):
            raise RuntimeError("candidate changed baseline runtime image configuration")
        after = _probe(runner, candidate_id)
        if (
            after["package_digests"] != digests
            or after.get("source_pins") != plan.source_pins
            or after.get("base_image_id") != plan.base_image_id
        ):
            raise RuntimeError(
                "candidate installed source proof does not match the plan"
            )
        for key in (
            "dependency_fingerprint",
            "shared_runtime_fingerprint",
            "required_topics",
        ):
            if after.get(key) != baseline.get(key):
                raise RuntimeError(f"candidate changed baseline {key}")
        receipt.update(
            {
                "status": "BUILT",
                "candidate_image_id": candidate_id,
                "candidate_tag": candidate_tag,
                "base_alias": base_alias,
                "wheel_sha256": manifest["wheel_sha256"],
                "baseline": baseline,
                "candidate": after,
            }
        )
        return receipt
    except Exception as exc:
        receipt["failure"] = str(exc)
        raise
    finally:
        receipt["completed_at"] = datetime.now(UTC).isoformat()
        _write(output / "candidate-receipt.json", receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        plan = ModelEffectsCandidatePlan.model_validate_json(
            args.plan.read_text(encoding="utf-8")
        )
        result = build_candidate(plan, args.output, execute=args.execute)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"candidate build REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
