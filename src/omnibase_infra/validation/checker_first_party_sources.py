#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Classify where each first-party package of a built environment came from.

OMN-17427 stage 0: the dev lane boots its composed runtime on dev-merge tags and
on nothing else. ``first_party_sources.yaml`` (beside this file) names the
first-party packages and the tag shape; this module reads what an environment
really installed (PEP 610 ``direct_url.json`` beside each ``*.dist-info``, plus
the staging record ``stage_workspace.sh`` writes for an install from a staged
local tree) and names every package whose source is not a dev-merge tag:

* ``sha``            a git install pinned to a commit
* ``branch``         a git install of a branch, or a staged tree on a branch
* ``release``        a release tag (``vX.Y.Z``) or a downloaded archive
* ``exact-version``  an index install, which is whatever ``==X.Y.Z`` resolved
* ``unbound``        a local install with no staging record to bind it to a tag
* ``tag-mismatch``   a tag whose twelve hex digits are not the installed commit
* ``absent``         a package this build requires and did not install

Two entry points, both usable without importing the rest of ``omnibase_infra``
(load the file by path where the package is not installed):

    stamp(staging_record, expected_refs, build_context)
        write the dev-merge tag each staged tree is checked out at into the record
    run_audit(staging_record, require=..., record_into=...)
        classify the installed environment; return 1 on any finding

It reads only the standard library plus PyYAML, which the image already has.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

CONTRACT_PATH = Path(__file__).resolve().parent / "first_party_sources.yaml"
DEV_TAG_REFSPEC = "dev-*"

KIND_DEV_MERGE_TAG = "dev-merge-tag"
KIND_SHA = "sha"
KIND_BRANCH = "branch"
KIND_RELEASE = "release"
KIND_EXACT_VERSION = "exact-version"
KIND_UNBOUND = "unbound"
KIND_TAG_MISMATCH = "tag-mismatch"
KIND_ABSENT = "absent"

_SHA = re.compile(r"^[0-9a-f]{7,40}$")
# Dots belong to one suffix segment. Only '+' and '-' separate later segments,
# so a run of dots cannot be partitioned between nested repetitions.
_RELEASE = re.compile(
    r"^v?[0-9]+\.[0-9]+(?:[.+-][0-9A-Za-z.]+(?:[+-][0-9A-Za-z.]+)*)?$"
)


@dataclass(frozen=True)
class FirstPartyPackage:
    distribution: str
    repo: str
    install: str


@dataclass(frozen=True)
class FirstPartyContract:
    policy: str
    tag_pattern: re.Pattern[str]
    packages: tuple[FirstPartyPackage, ...]

    def package(self, distribution: str) -> FirstPartyPackage:
        wanted = normalize(distribution)
        for package in self.packages:
            if normalize(package.distribution) == wanted:
                return package
        raise KeyError(distribution)


@dataclass(frozen=True)
class InstalledSource:
    """What one first-party distribution was installed from."""

    distribution: str
    repo: str
    kind: str
    version: str = ""
    tag: str = ""
    commit: str = ""
    install_url: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.kind == KIND_DEV_MERGE_TAG

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_contract(path: Path = CONTRACT_PATH) -> FirstPartyContract:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("policy") != KIND_DEV_MERGE_TAG:
        raise ValueError(f"{path}: policy must be {KIND_DEV_MERGE_TAG!r}")
    packages = tuple(
        FirstPartyPackage(
            distribution=str(item["distribution"]),
            repo=str(item["repo"]),
            install=str(item["install"]),
        )
        for item in raw["packages"]
    )
    return FirstPartyContract(
        policy=str(raw["policy"]),
        tag_pattern=re.compile(str(raw["tag_pattern"])),
        packages=packages,
    )


def newest_dev_merge_tag(names: Iterable[str], contract: FirstPartyContract) -> str:
    """The newest tag of ``names`` that has the dev-merge shape ('' when none).

    The stamp is the commit time in UTC, so a plain sort by stamp, then name,
    puts the newest last.
    """
    tags = [name for name in names if contract.tag_pattern.match(name)]
    if not tags:
        return ""
    return max(tags, key=lambda name: (name.split("-")[1], name))


def _read_direct_url(dist: importlib.metadata.Distribution) -> dict[str, object] | None:
    text = dist.read_text("direct_url.json")
    if text is None:
        return None
    parsed = json.loads(text)
    return parsed if isinstance(parsed, dict) else None


def _staging_record(
    package: FirstPartyPackage, staging: Mapping[str, object]
) -> Mapping[str, object] | None:
    section = "build_context" if package.install == "build-context" else "siblings"
    records = staging.get(section, {})
    record = records.get(package.repo) if isinstance(records, Mapping) else None
    return record if isinstance(record, Mapping) else None


def _classify_revision(
    rev: str, commit: str, contract: FirstPartyContract
) -> tuple[str, str]:
    """(kind, detail) for a git requested revision and the commit it resolved to."""
    match = contract.tag_pattern.match(rev)
    if match is not None:
        if commit and not commit.startswith(match["sha12"]):
            return (
                KIND_TAG_MISMATCH,
                f"tag {rev} names {match['sha12']}, installed {commit[:12]}",
            )
        return KIND_DEV_MERGE_TAG, ""
    if _SHA.match(rev):
        return KIND_SHA, f"pinned to commit {rev}"
    if _RELEASE.match(rev):
        return KIND_RELEASE, f"release tag {rev}"
    if not rev:
        return KIND_BRANCH, "no revision requested: the default branch head"
    return KIND_BRANCH, f"branch or non-dev-merge ref {rev}"


def _classify_staged(
    record: Mapping[str, object] | None, contract: FirstPartyContract
) -> tuple[str, str, str, str]:
    """(kind, tag, commit, detail) for an install from a staged local tree."""
    if record is None:
        return (
            KIND_UNBOUND,
            "",
            "",
            "no staging record binds this local install to a ref",
        )
    commit = str(record.get("vcs_ref", ""))
    tag = str(record.get("dev_merge_tag", ""))
    if record.get("vcs_dirty"):
        return KIND_BRANCH, tag, commit, "the staged tree was dirty"
    if tag:
        kind, detail = _classify_revision(tag, commit, contract)
        return kind, tag, commit, detail
    branch = str(record.get("vcs_branch", ""))
    if branch and branch != "HEAD":
        return KIND_BRANCH, "", commit, f"staged from branch {branch}"
    return KIND_SHA, "", commit, f"staged at commit {commit[:12]} with no dev-merge tag"


def classify_distribution(
    package: FirstPartyPackage,
    dist: importlib.metadata.Distribution,
    staging: Mapping[str, object],
    contract: FirstPartyContract,
) -> InstalledSource:
    version = dist.version
    direct_url = _read_direct_url(dist)
    if direct_url is None:
        return InstalledSource(
            package.distribution,
            package.repo,
            KIND_EXACT_VERSION,
            version=version,
            detail="installed from a package index, so only a version names it",
        )
    url = str(direct_url.get("url", ""))
    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, dict):
        rev = str(vcs_info.get("requested_revision", ""))
        commit = str(vcs_info.get("commit_id", ""))
        kind, detail = _classify_revision(rev, commit, contract)
        tag = rev if kind == KIND_DEV_MERGE_TAG else ""
        return InstalledSource(
            package.distribution, package.repo, kind, version, tag, commit, url, detail
        )
    if "archive_info" in direct_url:
        return InstalledSource(
            package.distribution,
            package.repo,
            KIND_RELEASE,
            version=version,
            install_url=url,
            detail="installed from a downloaded archive",
        )
    record = _staging_record(package, staging)
    kind, tag, commit, detail = _classify_staged(record, contract)
    return InstalledSource(
        package.distribution, package.repo, kind, version, tag, commit, url, detail
    )


def audit_environment(
    contract: FirstPartyContract,
    staging: Mapping[str, object],
    *,
    search_paths: Sequence[str] | None = None,
    required: Iterable[str] = (),
) -> list[InstalledSource]:
    """One ``InstalledSource`` per first-party distribution found, plus an
    ``absent`` row for each ``required`` distribution that is not installed.
    """
    dists: dict[str, importlib.metadata.Distribution] = {}
    found_dists = (
        importlib.metadata.distributions()
        if search_paths is None
        else importlib.metadata.distributions(path=list(search_paths))
    )
    for dist in found_dists:
        name = dist.metadata["Name"]
        if name:
            dists.setdefault(normalize(name), dist)
    needed = {normalize(name) for name in required}
    found: list[InstalledSource] = []
    for package in contract.packages:
        key = normalize(package.distribution)
        installed = dists.get(key)
        if installed is None:
            if key in needed:
                found.append(
                    InstalledSource(
                        package.distribution,
                        package.repo,
                        KIND_ABSENT,
                        detail="required by this build and not installed",
                    )
                )
            continue
        found.append(classify_distribution(package, installed, staging, contract))
    return found


def violations(sources: Iterable[InstalledSource]) -> list[InstalledSource]:
    return [source for source in sources if not source.ok]


def describe(source: InstalledSource) -> str:
    where = source.tag or source.commit[:12] or source.version or "-"
    return f"{source.distribution}: {source.kind} ({where}) {source.detail}".rstrip()


def _git(tree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={tree}", "-C", str(tree), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _build_context_record(
    tree: Path, contract: FirstPartyContract
) -> dict[str, object]:
    """The staging record of the build context (omnibase_infra itself)."""
    tags = _git(tree, "tag", "--points-at", "HEAD", "--list", DEV_TAG_REFSPEC)
    dirty = _git(tree, "status", "--porcelain", "--", ".", ":!workspace")
    return {
        "vcs_ref": _git(tree, "rev-parse", "HEAD"),
        "vcs_dirty": bool(dirty),
        "vcs_branch": _git(tree, "rev-parse", "--abbrev-ref", "HEAD"),
        "dev_merge_tag": newest_dev_merge_tag(tags.split(), contract),
    }


def stamp(
    staging_record: Path,
    expected_refs: Path,
    build_context: Path,
    contract_path: Path = CONTRACT_PATH,
) -> dict[str, object]:
    """Write each tree's dev-merge tag, and the build context, into the record.

    A sibling's tag is the ref RT-1 checked it out at when that ref is a
    dev-merge tag (``deploy_source_ref.py --follow-dev-tag``); a sibling checked
    out at anything else records none, which the audit refuses.
    """
    contract = load_contract(contract_path)
    record: dict[str, object] = json.loads(staging_record.read_text(encoding="utf-8"))
    rows = json.loads(expected_refs.read_text(encoding="utf-8")).get("repos", {})
    siblings = record.get("siblings", {})
    if isinstance(siblings, dict):
        for repo, entry in siblings.items():
            ref = str(rows.get(repo, {}).get("ref", ""))
            entry["dev_merge_tag"] = ref if contract.tag_pattern.match(ref) else ""
    record["build_context"] = {
        "omnibase_infra": _build_context_record(build_context, contract)
    }
    staging_record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def run_audit(
    staging_record: Path,
    *,
    require: Sequence[str] = (),
    record_into: Path | None = None,
    search_paths: Sequence[str] | None = None,
    contract_path: Path = CONTRACT_PATH,
) -> int:
    """Audit the installed environment; print one line per package; 1 on a finding.

    ``staging_record`` is the staging record or a build-provenance.json (it
    carries the record under ``per_repo_vcs_provenance``), so a running container
    is audited with ``staging_record=/app/build-provenance.json``. With
    ``record_into`` the classification is written into that build-provenance.json.
    """
    contract = load_contract(contract_path)
    staging = json.loads(staging_record.read_text(encoding="utf-8"))
    staging = staging.get("per_repo_vcs_provenance", staging)
    sources = audit_environment(
        contract, staging, search_paths=search_paths, required=require
    )
    if record_into is not None:
        manifest = json.loads(record_into.read_text(encoding="utf-8"))
        manifest["first_party_sources"] = [source.to_dict() for source in sources]
        manifest.setdefault("per_repo_vcs_provenance", {})["build_context"] = (
            staging.get("build_context", {})
        )
        record_into.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for source in sources:
        line = ("ok   " if source.ok else "FAIL ") + describe(source)
        sys.stdout.write(line + "\n")
    bad = violations(sources)
    if bad:
        sys.stderr.write(
            f"{len(bad)} first-party package(s) not resolved from a dev-merge tag\n"
        )
    return 1 if bad else 0
