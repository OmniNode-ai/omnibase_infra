# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read installed image content without importing or starting runtime packages.

Sent on stdin to an isolated, network-disabled image inspection process. Output
contains hashes and source IDs only, never configuration or environment values.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import re
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

SIBLINGS = ("omnibase_core", "omnibase_compat", "omnimarket")
TOPIC = re.compile(r"\bonex\.(?:evt|cmd|intent)\.[a-z0-9_-]+\.[a-z0-9_.-]+\.v[0-9]+\b")
# Dockerfile.runtime installs the image's own project editable from /app
# (`uv sync`), so its bytes live under /app/src, not in site-packages. That one
# install is attested by its source tree; every other editable still refuses.
IMAGE_PROJECT_ROOT = Path("/app")
IMAGE_PROJECT_DISTRIBUTION = "omnibase-infra"


def _is_image_project(name: str, direct_url: str | None) -> bool:
    """True for the image's own editable project; refuse any other editable."""
    if not direct_url:
        return False
    value = json.loads(direct_url)
    if not value.get("dir_info", {}).get("editable"):
        return False
    if (
        name != IMAGE_PROJECT_DISTRIBUTION
        or value.get("url") != IMAGE_PROJECT_ROOT.as_uri()
    ):
        raise ValueError(f"editable dependency cannot be attested: {name}")
    return True


def installed_package_root(name: str) -> Path:
    """Locate installed package bytes, including the image's editable project."""
    dist = importlib.metadata.distribution(name)
    normalized = dist.metadata["Name"].lower().replace("_", "-")
    if _is_image_project(normalized, dist.read_text("direct_url.json")):
        root = IMAGE_PROJECT_ROOT / "src" / name
    else:
        root = Path(str(dist.locate_file(name)))
    if not root.is_dir():
        raise ValueError(f"required installed package tree absent: {name}")
    return root


def tree_digest(root: Path) -> str:
    """Match the existing workspace provenance package-content algorithm."""
    digest = hashlib.sha256()
    if not root.is_dir():
        raise ValueError(f"required installed package tree absent: {root.name}")
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or any(
            part in {".git", "__pycache__", ".venv"} or part.endswith(".egg-info")
            for part in rel.parts
        ):
            continue
        digest.update(str(rel).encode())
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def dependency_digest(distributions: Iterable[importlib.metadata.Distribution]) -> str:
    """Hash actual non-Market distribution content, not just version labels.

    Install-location metadata and bytecode are excluded. Runtime-relevant
    metadata (including entry points) and all payload files remain included.
    Unrecorded distributions, and editables other than the image's own
    project (hashed from its source tree), refuse explicitly.
    """
    digest = hashlib.sha256()
    named = sorted(
        (
            (dist.metadata["Name"].lower().replace("_", "-"), dist)
            for dist in distributions
        ),
        key=lambda pair: pair[0],
    )
    if not named or len({name for name, _ in named}) != len(named):
        raise ValueError("installed distribution inventory absent or duplicated")
    for name, dist in named:
        if name == "omnimarket":
            continue
        editable = _is_image_project(name, dist.read_text("direct_url.json"))
        digest.update(json.dumps([name, dist.version]).encode())
        if editable:
            source = IMAGE_PROJECT_ROOT / "src" / name.replace("-", "_")
            digest.update(tree_digest(source).encode())
        # Distribution.files may silently omit missing payload files. Read the
        # wheel inventory itself so missing files fail rather than disappear.
        record = dist.read_text("RECORD")
        if not record:
            raise ValueError(f"dependency file inventory absent: {name}")
        rows = list(csv.reader(io.StringIO(record)))
        if not rows or any(len(row) != 3 or not row[0] for row in rows):
            raise ValueError(f"malformed dependency file inventory: {name}")
        paths = [Path(row[0]) for row in rows]
        if len(set(paths)) != len(paths):
            raise ValueError(f"duplicated dependency file inventory: {name}")
        for rel in sorted(paths):
            installation_metadata = any(
                part.endswith(".dist-info") for part in rel.parts
            ) and rel.name in {"RECORD", "direct_url.json", "INSTALLER", "REQUESTED"}
            if (
                installation_metadata
                or "__pycache__" in rel.parts
                or rel.suffix in {".pyc", ".pyo"}
            ):
                continue
            path = Path(str(dist.locate_file(rel)))
            digest.update(str(rel).encode())
            if not path.is_file():
                # strip_runtime_entry_points (Dockerfile.runtime) deletes a
                # legacy distribution's entry points without rewriting RECORD.
                if rel.name == "entry_points.txt" and rel.parent.suffix == ".dist-info":
                    digest.update(b"\0stripped")
                    continue
                raise ValueError(f"recorded dependency payload absent: {name}:{rel}")
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
    return digest.hexdigest()


def source_pins(app: Path) -> dict[str, str]:
    """Validate existing provenance against current installed package bytes."""
    manifest: dict[str, Any] = json.loads(
        (app / "build-provenance.json").read_text(encoding="utf-8")
    )
    if manifest.get("build_source") == "release":
        return {}
    if manifest.get("build_source") not in {"workspace", "scoped-derived"}:
        raise ValueError("unsupported or missing image build provenance")
    siblings = manifest["per_repo_vcs_provenance"]["siblings"]
    proofs = manifest["proofs"]
    pins: dict[str, str] = {}
    for name in SIBLINGS:
        row = siblings[name]
        sha = row["vcs_ref"]
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"non-immutable image source: {name}")
        if row.get("vcs_dirty") is not False:
            raise ValueError(f"unclean image source: {name}")
        matching = [proof for proof in proofs if proof.get("repo") == name]
        if len(matching) != 1 or matching[0].get("status") != "verified":
            raise ValueError(f"missing/ambiguous package content proof: {name}")
        proof = matching[0]
        actual = tree_digest(installed_package_root(name))
        if not (
            actual
            == proof["installed_package_digest"]
            == proof["staged_package_digest"]
        ):
            raise ValueError(f"installed content differs from source proof: {name}")
        pins[name] = sha
    return pins


def shared_evidence(app: Path) -> tuple[str, list[str]]:
    """Fingerprint shared boot/activation material and declared topic names.

    The declared set is a conservative admission check, not a claim to enumerate
    every dynamically synthesized topic or every external reader.
    """
    import yaml

    digest = hashlib.sha256()
    topics: set[str] = set()
    for name in ("contracts", "config", "entrypoint-runtime.sh"):
        root = app / name
        if not root.exists():
            raise ValueError(f"required shared runtime material absent: {name}")
        files = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in files:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            content = path.read_bytes()
            digest.update(str(path.relative_to(app)).encode())
            digest.update(content)
            if path.suffix in {".yaml", ".yml", ".json"}:
                topics.update(TOPIC.findall(content.decode("utf-8")))
    for name in ("omnibase_core", "omnibase_infra", "omnibase_compat", "omnimarket"):
        dist = importlib.metadata.distribution(name)
        package = installed_package_root(name)
        if name == "omnimarket":
            # Plugin/CLI entry-point changes can activate work even when the
            # topic-name set is unchanged. Market payload is the allowed change;
            # its activation/ownership configuration is not.
            for metadata in ("entry_points.txt", "METADATA"):
                digest.update(metadata.encode())
                digest.update((dist.read_text(metadata) or "").encode())
        for path in sorted(package.rglob("*")):
            if not path.is_file() or path.suffix not in {".yaml", ".yml", ".json"}:
                continue
            text_content = path.read_text(encoding="utf-8")
            topics.update(TOPIC.findall(text_content))
            if name != "omnimarket":
                continue  # Already covered by the dependency payload digest.
            digest.update(str(path.relative_to(package)).encode())
            if path.name == "contract.yaml":
                contract = yaml.safe_load(text_content)
                if not isinstance(contract, dict):
                    raise ValueError("invalid installed Market node contract")
                activation = {
                    key: value
                    for key, value in contract.items()
                    if key not in {"contract_version", "node_version", "description"}
                }
                # Ticket attribution is not a bootstrap requirement. Keep all
                # other metadata, including unknown future activation fields.
                if isinstance(activation.get("metadata"), dict):
                    activation["metadata"] = {
                        key: value
                        for key, value in activation["metadata"].items()
                        if key != "related_tickets"
                    }
                digest.update(
                    json.dumps(
                        _without_descriptions(activation), sort_keys=True
                    ).encode()
                )
            elif path.name == "metadata.yaml":
                # Descriptive/version metadata can accompany the approved code
                # change. Activation metadata remains part of the comparison.
                metadata_value = yaml.safe_load(text_content)
                if not isinstance(metadata_value, dict):
                    raise ValueError("invalid installed Market node metadata")
                metadata_value = {
                    key: value
                    for key, value in metadata_value.items()
                    if key not in {"version", "description"}
                }
                digest.update(
                    json.dumps(
                        _without_descriptions(metadata_value), sort_keys=True
                    ).encode()
                )
            else:
                digest.update(text_content.encode())
    if not topics:
        raise ValueError("declared runtime topic universe is empty")
    declared = sorted(topics)
    digest.update(json.dumps(declared, separators=(",", ":")).encode())
    return digest.hexdigest(), declared


def _without_descriptions(value: Any) -> Any:
    """Ignore documentation text, never routing/profile/configuration fields."""
    if isinstance(value, dict):
        return {
            key: _without_descriptions(item)
            for key, item in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    if isinstance(value, (date, datetime)):
        # YAML timestamps are typed values, not JSON strings; retain both
        # type and value in the canonical fingerprint.
        return {"yaml_scalar_type": type(value).__name__, "value": value.isoformat()}
    return value


def collect_image_evidence(app: Path = Path("/app")) -> dict[str, object]:
    """Collect immutable content evidence; performs no external I/O or imports."""
    shared, topics = shared_evidence(app)
    manifest = json.loads((app / "build-provenance.json").read_text(encoding="utf-8"))
    infra_ref = manifest.get("infra_vcs_ref")
    if not isinstance(infra_ref, str) or not re.fullmatch(
        r"[0-9a-f]{12,40}", infra_ref
    ):
        raise ValueError("missing immutable infrastructure source provenance")
    return {
        "source_pins": source_pins(app),
        "dependency_fingerprint": dependency_digest(importlib.metadata.distributions()),
        "shared_runtime_fingerprint": shared,
        "required_topics": topics,
        "infra_source_ref": infra_ref,
        "base_image_id": manifest.get("base_image_id"),
        "market_payload_fingerprint": tree_digest(installed_package_root("omnimarket")),
    }


if __name__ == "__main__":
    print(json.dumps(collect_image_evidence(), sort_keys=True))
