# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21928: a pre-PR slot may audit its target from a branch; nothing else relaxes.

``first_party_sources.yaml`` declares named build contexts. The dev lane's context
accepts a dev-merge tag and nothing else. The pre-PR slot's context also accepts
the TARGET package (the one installed from the build context) staged from a
branch, because a slot builds the unmerged branch under test. A vendored sibling
needs a dev-merge tag in every context. These tests build throwaway environments
(a directory of ``*.dist-info`` with the PEP 610 ``direct_url.json`` a real
install leaves) and run the audit in each context, with the data file edited in
place of the code for the declared-context cases.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER = REPO_ROOT / "src/omnibase_infra/validation/checker_first_party_sources.py"
POLICY = CHECKER.with_name("first_party_sources.yaml")

pytestmark = pytest.mark.unit


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fps = _load("checker_first_party_sources_omn21928", CHECKER)
CONTRACT = fps.load_contract()
SLOT = CONTRACT.pre_pr_slot_context
DEV = CONTRACT.default_context

COMMITS = {
    "omnibase_core": "66d35d2283c3" + "a" * 28,
    "omnibase_spi": "f79fd4dce71e" + "b" * 28,
    "omnimarket": "2b214bdc0c13" + "d" * 28,
    "omnibase_infra": "62fb83e4bd0c" + "e" * 28,
}
TAGS = {
    "omnibase_core": "dev-20261030152335-66d35d2283c3",
    "omnibase_spi": "dev-20261030141310-f79fd4dce71e",
    "omnimarket": "dev-20261030152625-2b214bdc0c13",
}
DISTRIBUTIONS = {
    "omnibase_core": "omnibase-core",
    "omnibase_spi": "omnibase-spi",
    "omnimarket": "omnimarket",
    "omnibase_infra": "omnibase-infra",
}
BRANCH = "omn-18087-t6-carry"


def _slot_record(repo: str) -> dict[str, Any]:
    """What the pre-PR slot entrypoint records for ``repo``."""
    if repo == "omnibase_infra":
        return {
            "vcs_ref": COMMITS[repo],
            "vcs_dirty": False,
            "vcs_branch": BRANCH,
            "dev_merge_tag": "",
        }
    return {
        "vcs_ref": COMMITS[repo],
        "vcs_dirty": False,
        "vcs_branch": "HEAD",
        "dev_merge_tag": TAGS[repo],
    }


def _provenance(**override: dict[str, Any]) -> dict[str, Any]:
    siblings = {
        repo: _slot_record(repo)
        for repo in ("omnibase_core", "omnibase_spi", "omnimarket")
    }
    target = {"omnibase_infra": _slot_record("omnibase_infra")}
    for repo, record in override.items():
        (target if repo == "omnibase_infra" else siblings)[repo] = record
    return {"siblings": siblings, "build_context": target}


def _site(tmp_path: Path) -> Path:
    site = tmp_path / "site-packages"
    site.mkdir(parents=True)
    for repo, dist in DISTRIBUTIONS.items():
        info = site / f"{dist.replace('-', '_')}-1.0.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist}\nVersion: 1.0.0\n", encoding="utf-8"
        )
        (info / "direct_url.json").write_text(
            json.dumps({"url": f"file:///workspace/{repo}", "dir_info": {}}),
            encoding="utf-8",
        )
    return site


def _audit(
    tmp_path: Path,
    provenance: dict[str, Any],
    context: str | None,
    contract: Any = CONTRACT,
) -> dict[str, Any]:
    sources = fps.audit_environment(
        contract,
        provenance,
        search_paths=[str(_site(tmp_path))],
        required=tuple(DISTRIBUTIONS.values()),
        context=context,
    )
    return {source.repo: source for source in sources}


def _write_policy(tmp_path: Path, edit: Any) -> Path:
    data = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    edit(data)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "first_party_sources.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _context(data: dict[str, Any], name: str) -> dict[str, Any]:
    (context,) = [c for c in data["build_contexts"] if c["name"] == name]
    return context


def test_first_party_slot_context_positive_control_a_tagged_target_passes_both(
    tmp_path: Path,
) -> None:
    tagged = {
        **_slot_record("omnibase_infra"),
        "dev_merge_tag": "dev-20261030140409-62fb83e4bd0c",
    }
    for context in (DEV, SLOT):
        found = _audit(tmp_path / context, _provenance(omnibase_infra=tagged), context)
        assert found["omnibase_infra"].kind == "dev-merge-tag", context
        assert fps.violations(found.values()) == [], context


def test_first_party_slot_context_negative_control(tmp_path: Path) -> None:
    """The same branch-staged target record: accepted in the slot, refused on the dev lane."""
    provenance = _provenance()
    slot = _audit(tmp_path / "slot", provenance, SLOT)
    assert slot["omnibase_infra"].kind == "branch"
    assert slot["omnibase_infra"].accepted_in == SLOT
    assert slot["omnibase_infra"].ok
    assert fps.violations(slot.values()) == []

    for context in (DEV, None):
        dev = _audit(tmp_path / f"dev-{context}", provenance, context)
        assert dev["omnibase_infra"].kind == "branch", context
        assert dev["omnibase_infra"].accepted_in == "", context
        assert not dev["omnibase_infra"].ok, context
        assert [s.repo for s in fps.violations(dev.values())] == ["omnibase_infra"]


def test_first_party_slot_context_negative_control_run_audit_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    staging = tmp_path / "sibling-vcs-provenance.json"
    staging.write_text(json.dumps(_provenance()), encoding="utf-8")
    site = str(_site(tmp_path))
    require = tuple(DISTRIBUTIONS.values())
    manifest = tmp_path / "build-provenance.json"
    manifest.write_text("{}", encoding="utf-8")

    assert (
        fps.run_audit(
            staging,
            require=require,
            search_paths=[site],
            record_into=manifest,
            context=SLOT,
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "ok   omnibase-infra: branch" in out
    assert f"[accepted in the {SLOT} build context]" in out
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert recorded["first_party_build_context"] == SLOT

    for context in ("", DEV):
        assert (
            fps.run_audit(
                staging, require=require, search_paths=[site], context=context
            )
            == 1
        )
        out = capsys.readouterr()
        assert "FAIL omnibase-infra: branch" in out.out
        assert f"(build context {DEV})" in out.err


def test_sibling_branch_still_fails_on_slot(tmp_path: Path) -> None:
    for repo in ("omnibase_core", "omnibase_spi", "omnimarket"):
        from_branch = {
            "vcs_ref": COMMITS[repo],
            "vcs_dirty": False,
            "vcs_branch": "dev",
            "dev_merge_tag": "",
        }
        for context in (SLOT, DEV):
            found = _audit(
                tmp_path / f"{repo}-{context}",
                _provenance(**{repo: from_branch}),
                context,
            )
            assert found[repo].kind == "branch", (repo, context)
            assert not found[repo].ok, (repo, context)
            assert found[repo].accepted_in == "", (repo, context)
            refused = {s.repo for s in fps.violations(found.values())}
            # The dev lane also refuses the branch-staged target; the slot does not.
            assert refused == {repo} | (
                {"omnibase_infra"} if context == DEV else set()
            ), (repo, context)


def test_sibling_branch_still_fails_on_slot_even_if_the_data_names_the_sibling_kind(
    tmp_path: Path,
) -> None:
    """A context relaxes the target only: every context accepting ``branch`` leaves a sibling refused."""
    path = _write_policy(
        tmp_path,
        lambda data: [
            context.update(target_accepts=["branch", "sha"])
            for context in data["build_contexts"]
        ],
    )
    contract = fps.load_contract(path)
    branch = {
        "vcs_ref": COMMITS["omnibase_core"],
        "vcs_dirty": False,
        "vcs_branch": "dev",
        "dev_merge_tag": "",
    }
    for context in (SLOT, DEV):
        found = _audit(
            tmp_path / context, _provenance(omnibase_core=branch), context, contract
        )
        assert not found["omnibase_core"].ok, context
        assert found["omnibase_infra"].ok, context


def test_declared_context_shipped_data_relaxes_only_the_slot_target() -> None:
    assert DEV == "dev-lane"
    assert SLOT == "pre-pr-slot"
    assert CONTRACT.context(DEV).target_accepts == ()
    assert CONTRACT.context(SLOT).target_accepts == ("branch",)


def test_declared_context_name_comes_from_the_data(tmp_path: Path) -> None:
    def rename(data: dict[str, Any]) -> None:
        _context(data, "pre-pr-slot")["name"] = "branch-under-test"
        data["pre_pr_slot_context"] = "branch-under-test"

    contract = fps.load_contract(_write_policy(tmp_path, rename))
    assert contract.pre_pr_slot_context == "branch-under-test"
    renamed = _audit(
        tmp_path / "renamed", _provenance(), contract.pre_pr_slot_context, contract
    )
    assert renamed["omnibase_infra"].ok
    assert renamed["omnibase_infra"].accepted_in == "branch-under-test"
    with pytest.raises(KeyError):
        _audit(tmp_path / "old-name", _provenance(), "pre-pr-slot", contract)


def test_declared_context_target_rule_comes_from_the_data(tmp_path: Path) -> None:
    provenance = _provenance()

    def close_slot(data: dict[str, Any]) -> None:
        _context(data, "pre-pr-slot")["target_accepts"] = []

    def open_dev(data: dict[str, Any]) -> None:
        _context(data, "dev-lane")["target_accepts"] = ["branch"]

    closed = fps.load_contract(_write_policy(tmp_path / "closed", close_slot))
    found = _audit(tmp_path / "closed-run", provenance, SLOT, closed)
    assert found["omnibase_infra"].kind == "branch"
    assert not found["omnibase_infra"].ok

    opened = fps.load_contract(_write_policy(tmp_path / "opened", open_dev))
    found = _audit(tmp_path / "opened-run", provenance, DEV, opened)
    assert found["omnibase_infra"].ok
    assert found["omnibase_infra"].accepted_in == DEV

    # Control: the shipped data gives the opposite answer for both contexts.
    shipped = _audit(tmp_path / "shipped-slot", provenance, SLOT)
    assert shipped["omnibase_infra"].ok
    assert not _audit(tmp_path / "shipped-dev", provenance, DEV)["omnibase_infra"].ok


def test_declared_context_unknown_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        _audit(tmp_path, _provenance(), "not-a-declared-context")


@pytest.mark.parametrize("kind", ["release", "exact-version", "unbound", "absent"])
def test_declared_context_cannot_relax_a_kind_that_is_never_acceptable(
    tmp_path: Path, kind: str
) -> None:
    path = _write_policy(
        tmp_path,
        lambda data: _context(data, "pre-pr-slot").update(target_accepts=[kind]),
    )
    with pytest.raises(ValueError, match="can be relaxed"):
        fps.load_contract(path)


def test_declared_context_default_must_name_a_declared_context(tmp_path: Path) -> None:
    path = _write_policy(
        tmp_path, lambda data: data.update(default_context="undeclared")
    )
    with pytest.raises(ValueError, match="not in build_contexts"):
        fps.load_contract(path)


def test_declared_context_the_image_audit_reads_the_build_arg_and_defaults_to_dev() -> (
    None
):
    dockerfile = (REPO_ROOT / "docker/Dockerfile.runtime").read_text(encoding="utf-8")
    assert 'ARG FIRST_PARTY_BUILD_CONTEXT=""' in dockerfile
    assert 'context=os.environ.get("FIRST_PARTY_BUILD_CONTEXT", "")' in dockerfile
    # The empty default is the policy's default (dev-lane) context.
    assert CONTRACT.default_context == DEV
