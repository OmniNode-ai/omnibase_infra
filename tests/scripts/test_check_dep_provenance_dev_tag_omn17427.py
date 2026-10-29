# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-merge lineage decisions and a real file-served git mirror."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/check_dep_provenance.py"
_FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "dep_provenance_dev_tag_omn17427.json"
)
SHA = "123456789abc" + "0" * 28
TAG = f"dev-20261029143607-{SHA[:12]}"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("dev_tag_provenance", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def manifest(tag=TAG, *, key="tag", version="0.47.58"):
    return (
        '[project]\nname="example"\nversion="0.0.0"\n'
        f'dependencies=["omnibase-core=={version}"]\n'
        "[tool.uv.sources]\n"
        'omnibase-core={git="https://github.com/OmniNode-ai/omnibase_core.git",'
        f'{key}="{tag}"}}\n'
    )


def lock_text(tag=TAG, sha=SHA, version="0.47.58"):
    return (
        '[[package]]\nname="omnibase-core"\n'
        f'version="{version}"\n'
        f'source={{git="https://github.com/OmniNode-ai/omnibase_core.git?tag={tag}#{sha}"}}\n'
    )


@pytest.mark.parametrize(
    "case", json.loads(_FIXTURES.read_text()), ids=lambda c: c["name"]
)
def test_pure_decision_paired_with_positive_control(mod, case):
    facts = mod.DevMergeTagFacts(True, SHA, "0.47.58", True, "fixture")
    lock = mod.DevMergeLockEntry(
        "0.47.58", f"https://github.com/OmniNode-ai/omnibase_core.git?tag={TAG}#{SHA}"
    )
    decide = mod.dev_merge_tag_violations
    assert (
        decide("omnibase-core", TAG, "0.47.58", facts, lock, lock_required=True) == []
    )
    kwargs = {
        "package": "omnibase-core",
        "tag": TAG,
        "declared_version": "0.47.58",
        "facts": facts,
        "lock_entry": lock,
        "lock_required": True,
    }
    if "facts" in case:
        kwargs["facts"] = replace(facts, **case["facts"])
    if "declared_version" in case:
        kwargs["declared_version"] = case["declared_version"]
    if "lock" in case:
        kwargs["lock_entry"] = replace(lock, **case["lock"])
    if case.get("missing_lock"):
        kwargs["lock_entry"] = None
    violations = decide(**kwargs)
    assert violations, case
    assert all("omnibase-core" in v and TAG in v for v in violations)
    assert any(case["condition"] in v for v in violations)


def test_each_failed_condition_has_its_own_line(mod):
    facts = mod.DevMergeTagFacts(True, "f" * 40, "0.47.58", False, "fixture")
    violations = mod.dev_merge_tag_violations(
        "omnibase-core", TAG, "0.47.57", facts, None, lock_required=True
    )
    assert len(violations) == 4


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rev", TAG),
        ("branch", TAG),
        ("tag", "dev-2026-abc"),
        ("tag", "dev-20261029143607-ABCDEF123456"),
        ("tag", "dev-20261029143607-123456789ab"),
        ("tag", "dev-20261029143607-123456789abcd"),
        ("tag", "v0.47.57"),
    ],
)
def test_other_shapes_keep_the_release_tree_check(mod, key, value):
    facts = mod.DevMergeTagFacts(True, SHA, "0.47.58", True, "fixture")
    dev_calls = []

    def dev_resolver(repo, tag):
        dev_calls.append((repo, tag))
        return facts

    assert (
        mod.find_lineage_violations(
            manifest(), resolve_dev_tag=dev_resolver, lock_text=lock_text()
        )
        == []
    )
    dev_calls.clear()
    calls = []

    def release_resolver(repo, ref):
        calls.append(ref)
        return ("pin" if ref == value else "release"), "fixture"

    assert mod.find_lineage_violations(
        manifest(value, key=key), resolve=release_resolver, resolve_dev_tag=dev_resolver
    )
    assert dev_calls == []
    assert calls == [value, "v0.47.58"]
    # Its release-path twin changes only the resolved content.
    assert (
        mod.find_lineage_violations(
            manifest(value, key=key),
            resolve=lambda *_: ("same", "fixture"),
            resolve_dev_tag=dev_resolver,
        )
        == []
    )


def test_range_is_not_admitted_by_tag_path(mod):
    good = manifest()

    def resolver(*_):
        return mod.DevMergeTagFacts(True, SHA, "0.47.58", True, "fixture")

    assert mod.find_lineage_violations(good, resolve_dev_tag=resolver) == []
    assert (
        "non-exact"
        in mod.find_lineage_violations(
            good.replace("==0.47.58", ">=0.47.58"), resolve_dev_tag=resolver
        )[0]
    )


def test_lock_parsing_and_missing_entry_fail_closed(mod):
    def resolver(*_):
        return mod.DevMergeTagFacts(True, SHA, "0.47.58", True, "fixture")

    assert (
        mod.find_lineage_violations(
            manifest(), resolve_dev_tag=resolver, lock_text=lock_text()
        )
        == []
    )
    for bad in [
        "package = []",
        "[[invalid",
        lock_text().replace("omnibase-core", "other"),
    ]:
        assert mod.find_lineage_violations(
            manifest(), resolve_dev_tag=resolver, lock_text=bad
        )


@pytest.fixture
def real_repo(tmp_path):
    root = tmp_path / "origin"
    root.mkdir()
    env = scrub_git_location_env()
    env.update(
        GIT_AUTHOR_NAME="Fixture",
        GIT_AUTHOR_EMAIL="fixture@example.invalid",
        GIT_COMMITTER_NAME="Fixture",
        GIT_COMMITTER_EMAIL="fixture@example.invalid",
    )

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            env=scrub_git_location_env(env),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    git("init", "-b", "dev")
    (root / "pyproject.toml").write_text(
        '[project]\nname="omnibase-core"\nversion="0.47.58"\n'
    )
    git("add", "pyproject.toml")
    git("commit", "-m", "first")
    first = git("rev-parse", "HEAD")
    tag = f"dev-20261029143607-{first[:12]}"
    git("tag", "-a", tag, "-m", "annotated dev merge")
    bad_suffix = "dev-20261029143607-000000000000"
    git("tag", bad_suffix)
    (root / "second").write_text("second")
    git("add", "second")
    git("commit", "-m", "second")
    git("switch", "-c", "feature", first)
    (root / "feature").write_text("feature")
    git("add", "feature")
    git("commit", "-m", "feature")
    feature_sha = git("rev-parse", "HEAD")
    feature_tag = f"dev-20261029143608-{feature_sha[:12]}"
    git("tag", feature_tag)
    git("switch", "dev")
    return root.as_uri(), first, tag, feature_tag, bad_suffix


def test_real_git_mirror_and_paired_failures(mod, real_repo):
    url, sha, tag, off_dev, bad_suffix = real_repo

    def resolve(_repo, ref):
        return mod.resolve_dev_merge_tag_via_mirror(
            "omnibase_core", ref, mirror_url=url
        )

    facts = resolve("omnibase_core", tag)
    assert facts.commit == sha  # annotated tag was peeled
    assert facts.on_dev is True  # first of two commits, not merely dev HEAD
    assert (
        mod.find_lineage_violations(
            manifest(tag), resolve_dev_tag=resolve, lock_text=lock_text(tag, sha)
        )
        == []
    )
    for invalid_tag, condition in [
        ("dev-20261029143607-ffffffffffff", "tag exists"),
        (off_dev, "reachable from dev"),
        (bad_suffix, "suffix"),
    ]:
        # Twin is the real valid-tag case above; exactly the tag under test differs.
        failures = mod.find_lineage_violations(
            manifest(invalid_tag), resolve_dev_tag=resolve
        )
        assert any(condition in v for v in failures), failures
    for declared, lock, condition in [
        ("0.47.57", lock_text(tag, sha), "declared version"),
        ("0.47.58", lock_text(tag, sha, "0.47.57"), "lock version"),
        ("0.47.58", lock_text(tag, "f" * 40), "lock commit"),
    ]:
        failures = mod.find_lineage_violations(
            manifest(tag, version=declared), resolve_dev_tag=resolve, lock_text=lock
        )
        assert any(condition in v for v in failures), failures


def test_hermetic_falls_back_only_when_mirror_cannot_answer(mod, monkeypatch):
    good = mod.DevMergeTagFacts(True, SHA, "0.47.58", True, "mirror")
    monkeypatch.setattr(mod, "resolve_dev_merge_tag_via_mirror", lambda *_: good)

    def no_api(*_):
        pytest.fail("complete mirror must not call API")

    monkeypatch.setattr(mod, "resolve_dev_merge_tag", no_api)
    assert mod.resolve_dev_merge_tag_hermetic("omnibase_core", TAG) == good
    partial = replace(good, on_dev=None)
    monkeypatch.setattr(mod, "resolve_dev_merge_tag_via_mirror", lambda *_: partial)
    monkeypatch.setattr(mod, "resolve_dev_merge_tag", lambda *_: good)
    resolved = mod.resolve_dev_merge_tag_hermetic("omnibase_core", TAG)
    assert replace(resolved, detail="mirror") == good


@pytest.mark.parametrize(
    ("status", "on_dev"),
    [
        ("ahead", True),
        ("identical", True),
        ("behind", False),
        ("diverged", False),
        ("unexpected", None),
    ],
)
def test_rest_compare_direction_and_annotated_peeling(mod, monkeypatch, status, on_dev):
    import base64

    calls = []

    def api(url):
        calls.append(url)
        if "/git/ref/tags/" in url:
            return 200, {"object": {"type": "tag", "sha": "a" * 40}}, "ok"
        if "/git/tags/" in url:
            return 200, {"object": {"type": "commit", "sha": SHA}}, "ok"
        if "/contents/" in url:
            return (
                200,
                {
                    "encoding": "base64",
                    "content": base64.b64encode(
                        b'[project]\nversion="0.47.58"\n'
                    ).decode(),
                },
                "ok",
            )
        assert f"/compare/{SHA}...dev" in url
        return 200, {"status": status}, "ok"

    monkeypatch.setattr(mod, "_api_get", api)
    facts = mod.resolve_dev_merge_tag("omnibase_core", TAG)
    assert facts.commit == SHA
    assert facts.version == "0.47.58"
    assert facts.on_dev is on_dev


def test_unreachable_mirror_is_undetermined(mod):
    facts = mod.resolve_dev_merge_tag_via_mirror(
        "omnibase_core", TAG, mirror_url="file:///nonexistent/omn17427.git"
    )
    assert facts.tag_exists is None
    assert (
        mod.dev_merge_tag_violations("omnibase-core", TAG, "0.47.58", facts, None)[
            0
        ].find("UNDETERMINED lineage")
        >= 0
    )


def test_offline_shape_admits_dev_merge_tag_without_token(mod):
    assert mod.find_violations(manifest()) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [("rev", TAG), ("branch", TAG), ("tag", "v0.47.58"), ("tag", "dev-2026-abc")],
)
def test_other_offline_shapes_need_existing_escape_token(mod, key, value):
    text = manifest(value, key=key)
    assert mod.find_violations(text)
    token = "# raw-override-" + "ok: OMN-17427 until=2026-11-12"
    assert mod.find_violations(text.rstrip() + " " + token + "\n") == []


@pytest.mark.parametrize("key", ["rev", "branch"])
@pytest.mark.parametrize("value", [TAG, ""])
def test_mixed_tag_sources_are_not_sanctioned(mod, key, value):
    text = manifest().replace(f'tag="{TAG}"', f'tag="{TAG}",{key}="{value}"')
    assert mod.find_violations(text)
    token = "# raw-override-" + "ok: OMN-17427 until=2026-11-12"
    assert mod.find_violations(text.rstrip() + " " + token + "\n") == []
