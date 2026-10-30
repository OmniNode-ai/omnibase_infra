# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427 stage 0: every first-party package resolves from a dev-merge tag.

The dev lane boots its composed runtime on dev-merge tags and on nothing else.
``first_party_sources.py`` reads what a built environment installed and names
every first-party package that came from a sha, a branch, a release or an exact
version. These tests build throwaway environments (a directory of
``*.dist-info`` with the PEP 610 ``direct_url.json`` a real install leaves),
prove a clean environment passes, and plant each violation in turn to prove the
audit refuses it. A check that never fails proves nothing, so the planted cases
are the point of the file.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_BUILD = REPO_ROOT / "scripts" / "runtime_build"

pytestmark = pytest.mark.unit


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "validation"
    / "checker_first_party_sources.py"
)
fps = _load("checker_first_party_sources_under_test", CHECKER)
CONTRACT = fps.load_contract()

# Four real tags of the shape the tag workflows write, with the commit each
# names (the first twelve hex digits of the commit are the tag's suffix).
COMMITS = {
    "omnibase_core": "66d35d2283c3" + "a" * 28,
    "omnibase_spi": "f79fd4dce71e" + "b" * 28,
    "omnibase_compat": "0123456789ab" + "c" * 28,
    "omnimarket": "2b214bdc0c13" + "d" * 28,
    "omnibase_infra": "62fb83e4bd0c" + "e" * 28,
}
TAGS = {
    "omnibase_core": "dev-20261030152335-66d35d2283c3",
    "omnibase_spi": "dev-20261030141310-f79fd4dce71e",
    "omnibase_compat": "dev-20261030153000-0123456789ab",
    "omnimarket": "dev-20261030152625-2b214bdc0c13",
    "omnibase_infra": "dev-20261030140409-62fb83e4bd0c",
}
DISTRIBUTIONS = {
    "omnibase_core": "omnibase-core",
    "omnibase_spi": "omnibase-spi",
    "omnibase_compat": "omnibase-compat",
    "omnimarket": "omnimarket",
    "omnibase_infra": "omnibase-infra",
}


def _record(repo: str, **override: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "vcs_ref": COMMITS[repo],
        "vcs_dirty": False,
        "vcs_branch": "HEAD",
        "dev_merge_tag": TAGS[repo],
    }
    record.update(override)
    return record


def _provenance(**records: dict[str, Any]) -> dict[str, Any]:
    siblings = {
        repo: _record(repo)
        for repo in ("omnibase_core", "omnibase_spi", "omnibase_compat", "omnimarket")
    }
    context = {"omnibase_infra": _record("omnibase_infra")}
    for repo, record in records.items():
        (context if repo == "omnibase_infra" else siblings)[repo] = record
    return {"siblings": siblings, "build_context": context}


def _local(repo: str) -> dict[str, Any]:
    return {"url": f"file:///workspace/sibling-repos/{repo}", "dir_info": {}}


def _vcs(repo: str, revision: str, commit: str | None = None) -> dict[str, Any]:
    return {
        "url": f"https://github.com/OmniNode-ai/{repo}.git",
        "vcs_info": {
            "vcs": "git",
            "requested_revision": revision,
            "commit_id": commit if commit is not None else COMMITS[repo],
        },
    }


def _clean_environment() -> dict[str, dict[str, Any] | None]:
    """Direct URLs of an image built the way the dev lane builds it."""
    return {
        "omnibase_core": _local("omnibase_core"),
        "omnibase_spi": _local("omnibase_spi"),
        "omnibase_compat": _local("omnibase_compat"),
        "omnimarket": {"url": "file:///opt/omnimarket", "dir_info": {}},
        "omnibase_infra": {"url": "file:///app", "dir_info": {"editable": True}},
    }


def _install(site: Path, direct_urls: dict[str, dict[str, Any] | None]) -> None:
    for repo, direct_url in direct_urls.items():
        dist = DISTRIBUTIONS[repo]
        info = site / f"{dist.replace('-', '_')}-1.0.0.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist}\nVersion: 1.0.0\n", encoding="utf-8"
        )
        if direct_url is not None:
            (info / "direct_url.json").write_text(
                json.dumps(direct_url), encoding="utf-8"
            )


def _audit(
    tmp_path: Path,
    direct_urls: dict[str, dict[str, Any] | None],
    provenance: dict[str, Any] | None = None,
    required: tuple[str, ...] = (),
) -> list[Any]:
    site = tmp_path / "site-packages"
    site.mkdir()
    _install(site, direct_urls)
    return fps.audit_environment(
        CONTRACT,
        provenance if provenance is not None else _provenance(),
        search_paths=[str(site)],
        required=required,
    )


def test_a_clean_dev_lane_environment_passes(tmp_path: Path) -> None:
    sources = _audit(tmp_path, _clean_environment())
    assert {s.repo for s in sources} == set(DISTRIBUTIONS)
    assert fps.violations(sources) == []
    by_repo = {s.repo: s for s in sources}
    for repo, source in by_repo.items():
        assert source.kind == "dev-merge-tag", repo
        assert source.tag == TAGS[repo], repo
        assert source.commit == COMMITS[repo], repo


def test_an_install_from_a_git_tag_passes_and_records_the_tag(tmp_path: Path) -> None:
    urls = _clean_environment()
    urls["omnibase_core"] = _vcs("omnibase_core", TAGS["omnibase_core"])
    (core,) = [s for s in _audit(tmp_path, urls) if s.repo == "omnibase_core"]
    assert core.ok
    assert core.tag == TAGS["omnibase_core"]
    assert core.commit == COMMITS["omnibase_core"]


# Each case plants one violation into an otherwise clean environment and names
# the kind the audit must report. The kinds are the ones the ticket names.
PLANTED = [
    pytest.param(
        "omnibase_core",
        _vcs("omnibase_core", COMMITS["omnibase_core"]),
        None,
        "sha",
        id="core-git-sha",
    ),
    pytest.param(
        "omnibase_spi",
        _vcs("omnibase_spi", "dev"),
        None,
        "branch",
        id="spi-git-branch",
    ),
    pytest.param(
        "omnibase_compat",
        _vcs("omnibase_compat", "v0.5.7"),
        None,
        "release",
        id="compat-release-tag",
    ),
    pytest.param(
        "omnibase_compat",
        {
            "url": "https://github.com/x/omnibase_compat/archive/v0.5.5.tar.gz",
            "archive_info": {"hashes": {}},
        },
        None,
        "release",
        id="compat-release-archive",
    ),
    pytest.param(
        "omnibase_spi",
        None,
        None,
        "exact-version",
        id="spi-index-exact-version",
    ),
    pytest.param(
        "omnibase_core",
        _vcs("omnibase_core", TAGS["omnibase_core"], commit="f" * 40),
        None,
        "tag-mismatch",
        id="core-tag-names-another-commit",
    ),
    pytest.param(
        "omnimarket",
        {"url": "file:///opt/omnimarket", "dir_info": {}},
        _record("omnimarket", dev_merge_tag=""),
        "sha",
        id="omnimarket-staged-at-a-sha",
    ),
    pytest.param(
        "omnimarket",
        {"url": "file:///opt/omnimarket", "dir_info": {}},
        _record("omnimarket", dev_merge_tag="", vcs_branch="dev"),
        "branch",
        id="omnimarket-staged-on-a-branch",
    ),
    pytest.param(
        "omnibase_compat",
        _local("omnibase_compat"),
        _record("omnibase_compat", vcs_dirty=True),
        "branch",
        id="compat-staged-from-a-dirty-tree",
    ),
    pytest.param(
        "omnibase_core",
        _local("omnibase_core"),
        _record("omnibase_core", dev_merge_tag="v0.47.58"),
        "release",
        id="core-staged-at-a-release-tag",
    ),
    pytest.param(
        "omnibase_infra",
        {"url": "file:///app", "dir_info": {"editable": True}},
        _record("omnibase_infra", dev_merge_tag=""),
        "sha",
        id="infra-built-from-a-sha",
    ),
    pytest.param(
        "omnibase_spi",
        _local("omnibase_spi"),
        _record("omnibase_spi", dev_merge_tag=TAGS["omnibase_core"]),
        "tag-mismatch",
        id="spi-staged-record-carries-another-repos-tag",
    ),
]


@pytest.mark.parametrize(("repo", "direct_url", "record", "kind"), PLANTED)
def test_a_planted_violation_is_refused(
    tmp_path: Path,
    repo: str,
    direct_url: dict[str, Any] | None,
    record: dict[str, Any] | None,
    kind: str,
) -> None:
    urls = _clean_environment()
    urls[repo] = direct_url
    staging = _provenance(**({repo: record} if record is not None else {}))
    bad = fps.violations(_audit(tmp_path, urls, staging))
    assert [(s.repo, s.kind) for s in bad] == [(repo, kind)], [
        fps.describe(s) for s in bad
    ]


def test_a_local_install_with_no_staging_record_is_unbound(tmp_path: Path) -> None:
    staging = _provenance()
    del staging["siblings"]["omnibase_spi"]
    bad = fps.violations(_audit(tmp_path, _clean_environment(), staging))
    assert [(s.repo, s.kind) for s in bad] == [("omnibase_spi", "unbound")]


def test_a_required_package_that_is_not_installed_is_reported(tmp_path: Path) -> None:
    urls = _clean_environment()
    del urls["omnibase_spi"]
    sources = _audit(tmp_path, urls, required=("omnibase-spi",))
    assert [(s.repo, s.kind) for s in fps.violations(sources)] == [
        ("omnibase_spi", "absent")
    ]


def test_a_package_the_build_does_not_require_may_be_absent(tmp_path: Path) -> None:
    urls = _clean_environment()
    del urls["omnimarket"]
    assert fps.violations(_audit(tmp_path, urls, required=("omnibase-core",))) == []


def _run_audit(
    site: Path, record: Path, *extra: Path | None
) -> subprocess.CompletedProcess[str]:
    """Call run_audit the way the Dockerfile does: from a fresh interpreter."""
    record_into = extra[0] if extra else None
    code = (
        "import sys; from pathlib import Path; "
        "import importlib.util as u; "
        f"s = u.spec_from_file_location('c', {str(CHECKER)!r}); "
        "m = u.module_from_spec(s); sys.modules['c'] = m; s.loader.exec_module(m); "
        f"sys.exit(m.run_audit(Path({str(record)!r}), require=['omnibase-spi'], "
        f"search_paths=[{str(site)!r}], "
        f"record_into={('Path(' + repr(str(record_into)) + ')') if record_into else 'None'}))"
    )
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def test_the_audit_command_exits_nonzero_for_a_violation_and_zero_for_a_clean_one(
    tmp_path: Path,
) -> None:
    clean_site = tmp_path / "clean"
    clean_site.mkdir()
    _install(clean_site, _clean_environment())
    planted_site = tmp_path / "planted"
    planted_site.mkdir()
    urls = _clean_environment()
    urls["omnibase_spi"] = None
    _install(planted_site, urls)
    record = tmp_path / "sibling-vcs-provenance.json"
    record.write_text(json.dumps(_provenance()), encoding="utf-8")

    clean = _run_audit(clean_site, record)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    planted = _run_audit(planted_site, record)
    assert planted.returncode == 1
    assert "FAIL omnibase-spi: exact-version" in planted.stdout


def test_the_audit_records_its_classification_into_the_build_provenance(
    tmp_path: Path,
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    _install(site, _clean_environment())
    record = tmp_path / "sibling-vcs-provenance.json"
    record.write_text(json.dumps(_provenance()), encoding="utf-8")
    manifest = tmp_path / "build-provenance.json"
    manifest.write_text(
        json.dumps({"build_source": "workspace", "per_repo_vcs_provenance": {}}),
        encoding="utf-8",
    )

    done = _run_audit(site, record, manifest)
    assert done.returncode == 0, done.stdout + done.stderr
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert {
        entry["repo"]: entry["kind"] for entry in written["first_party_sources"]
    } == dict.fromkeys(DISTRIBUTIONS, "dev-merge-tag")
    assert (
        written["per_repo_vcs_provenance"]["build_context"]
        == _provenance()["build_context"]
    )

    # A running container is audited from the manifest alone.
    manifest_only = _run_audit(site, manifest)
    assert manifest_only.returncode == 1  # the manifest holds no sibling records
    assert "unbound" in manifest_only.stdout


def _commit_repo(path: Path) -> str:
    env = {
        **scrub_git_location_env(os.environ),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t.t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t.t",
        "GIT_AUTHOR_DATE": "2026-10-30T15:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-10-30T15:00:00+00:00",
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
            env=scrub_git_location_env(env),
        ).stdout.strip()

    path.mkdir(parents=True)
    git("init", "-q", "-b", "dev")
    (path / "f.txt").write_text("x\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    return git("rev-parse", "HEAD")


def test_stamp_records_the_tag_each_tree_was_checked_out_at(tmp_path: Path) -> None:
    infra = tmp_path / "infra"
    sha = _commit_repo(infra)
    infra_tag = f"dev-20261030150000-{sha[:12]}"
    subprocess.run(
        ["git", "-C", str(infra), "tag", infra_tag],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )
    record = tmp_path / "record.json"
    record.write_text(
        json.dumps(
            {
                "siblings": {
                    "omnibase_core": {
                        "vcs_ref": "a" * 40,
                        "vcs_dirty": False,
                        "vcs_branch": "HEAD",
                    },
                    "omnibase_spi": {
                        "vcs_ref": "b" * 40,
                        "vcs_dirty": False,
                        "vcs_branch": "HEAD",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    refs = tmp_path / "refs.json"
    refs.write_text(
        json.dumps(
            {
                "repos": {
                    "omnibase_core": {"ref": "dev-20261030152335-" + "a" * 12},
                    "omnibase_spi": {"ref": "origin/dev"},
                }
            }
        ),
        encoding="utf-8",
    )

    # stage_workspace.sh calls stamp from a fresh interpreter that loads the file
    # by path (the deploy host has no omnibase_infra installed); run that program.
    stage = (RUNTIME_BUILD / "stage_workspace.sh").read_text(encoding="utf-8")
    line = next(line for line in stage.splitlines() if "m.stamp(" in line)
    program = shlex.split(line.strip())[2]
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(CHECKER),
            str(record),
            str(refs),
            str(infra),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    stamped = json.loads(record.read_text(encoding="utf-8"))
    assert stamped["siblings"]["omnibase_core"]["dev_merge_tag"] == (
        "dev-20261030152335-" + "a" * 12
    )
    # a sibling staged at a branch ref records no tag, which the audit refuses
    assert stamped["siblings"]["omnibase_spi"]["dev_merge_tag"] == ""
    context = stamped["build_context"]["omnibase_infra"]
    assert context["vcs_ref"] == sha
    assert context["dev_merge_tag"] == infra_tag
    assert context["vcs_dirty"] is False


def test_newest_dev_merge_tag_is_the_latest_stamp_and_ignores_other_names() -> None:
    names = [
        "v0.5.7",
        "dev-20261030091909-cff20073f3ac",
        "dev-20261030141310-f79fd4dce71e",
        "dev-20261030034714-0e0b63afe98d",
        "dev-latest",
        "dev-2026103014131-f79fd4dce71e",
    ]
    assert (
        fps.newest_dev_merge_tag(names, CONTRACT) == "dev-20261030141310-f79fd4dce71e"
    )
    assert fps.newest_dev_merge_tag(["v0.5.7", "main"], CONTRACT) == ""


# -- the contract agrees with the places that spell the same sets ------------


def _manifest_array(name: str) -> list[str]:
    script = f'source "{RUNTIME_BUILD / "sibling_clone_manifest.sh"}"; printf "%s\\n" "${{{name}[@]}}"'
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    return result.stdout.split()


def test_contract_packages_are_the_sibling_clone_manifest() -> None:
    manifest = _manifest_array("SIBLING_CLONE_MANIFEST")
    dists = _manifest_array("SIBLING_CLONE_MANIFEST_DIST_NAMES")
    assert {p.repo: p.distribution for p in CONTRACT.packages} == dict(
        zip(manifest, dists, strict=True)
    )
    vendored = _manifest_array("SIBLING_VENDORED_REPOS")
    assert sorted(p.repo for p in CONTRACT.packages if p.install == "vendored") == (
        sorted(vendored)
    )
    assert [p.repo for p in CONTRACT.packages if p.install == "build-context"] == [
        "omnibase_infra"
    ]


def test_every_vendored_sibling_is_staged_at_its_dev_merge_tag() -> None:
    stage = (RUNTIME_BUILD / "stage_workspace.sh").read_text(encoding="utf-8")
    assert 'for repo in "${SIBLING_REPOS[@]}"; do' in stage
    assert 'checkout_args+=(--follow-dev-tag "${repo}")' in stage
    assert "m.stamp(" in stage
    assert 'SIBLING_REPOS=("${SIBLING_VENDORED_REPOS[@]}")' in stage


def test_the_tag_pattern_is_the_one_the_checkout_and_the_workflows_use() -> None:
    deploy_source_ref = _load(
        "deploy_source_ref_for_pattern", RUNTIME_BUILD / "deploy_source_ref.py"
    )
    assert CONTRACT.tag_pattern.pattern == deploy_source_ref.DEV_TAG_PATTERN.pattern
    workflow = (REPO_ROOT / ".github" / "workflows" / "dev-merge-tag.yml").read_text(
        encoding="utf-8"
    )
    assert 'TAG="dev-${STAMP}-${SHA12}"' in workflow


def test_the_image_installs_every_workspace_package_from_the_staged_tree() -> None:
    provenance = _load(
        "compute_workspace_provenance_for_set",
        RUNTIME_BUILD / "compute_workspace_provenance.py",
    )
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile.runtime").read_text(
        encoding="utf-8"
    )
    for repo, dist in provenance.WORKSPACE_PACKAGES.items():
        assert f'"{dist} @ file:///workspace/sibling-repos/{repo}"' in dockerfile, repo
    assert "for repo in omnibase_core omnibase_spi omnibase_compat; do" in dockerfile
    # the audit that fails the build, and the packages it requires
    audit = dockerfile.split("import run_audit;", 1)[1].split("fi\n", 1)[0]
    required = audit.split("]))'", 1)[1].replace("\\", " ").replace(";", " ").split()
    assert required == [*provenance.WORKSPACE_PACKAGES.values(), "omnibase-infra"]
