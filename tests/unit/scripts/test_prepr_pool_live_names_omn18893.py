# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""changed_path_live names what the runtime logs, and waives what never runs (OMN-18893).

A runtime logs a node's consumer group and its ``node=`` field from the contract's
``name``, which need not be the node directory's name (``node_host_reconcile_compute``
logs as ``host_reconcile_compute``), so a derived live term that only carries the
directory name can never match. And a node whose contract declares no topics and no
runtime profile (a local-transport node) runs in no runtime at all, so demanding a
runtime log line from it made its receipt unpassable. These tests drive the host's
``derive_live_grep`` / ``derive_live_not_applicable`` functions and the pool judge.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_BASE = Path(__file__).with_name("test_prepr_pool_pr_head_receipt_omn19566.py")


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load("prepr_pool_receipt_helpers_omn18893", _BASE)
pool = base.pool

RUNTIME_CONTRACT = """\
name: {name}
node_type: compute
event_bus:
  subscribe_topics:
    - onex.cmd.x.request.v1
  publish_topics:
    - onex.evt.x.done.v1
"""
LOCAL_CONTRACT = """\
name: "{name}"
node_type: COMPUTE_GENERIC
runtime_profiles: []
event_bus:
  subscribe_topics: []
  publish_topics: []
descriptor:
  runtime_profiles: []
"""
PROFILE_ONLY_CONTRACT = """\
name: {name}
runtime_profiles:
  - runtime_main
event_bus:
  subscribe_topics: []
  publish_topics: []
"""


def _repo_with_nodes(tmp_path: Path, contracts: dict[str, str]) -> Path:
    """A repo whose PR touches every node in ``contracts`` (dir name -> contract)."""
    root = tmp_path / "root"
    repo = root / "omnibase_infra"
    nodes = repo / "src" / "omnibase_infra" / "nodes"
    nodes.mkdir(parents=True)
    (nodes / "__init__.py").write_text("", encoding="utf-8")
    for node in contracts:
        (nodes / node).mkdir()
        (nodes / node / "contract.yaml").write_text("name: placeholder\n", "utf-8")
    base._git(repo, "init", "-q", "-b", "dev")
    base._git(repo, "add", "-A")
    base._git(repo, "commit", "-qm", "base")
    base._git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    base._git(repo, "switch", "-q", "-c", "pr")
    for node, contract in contracts.items():
        (nodes / node / "contract.yaml").write_text(
            contract.format(name=node.removeprefix("node_")), encoding="utf-8"
        )
    base._git(repo, "add", "-A")
    base._git(repo, "commit", "-qm", "head")
    return root


def _host(root: Path, function: str) -> str:
    """Run one host-side derivation function over the repo and return its stdout."""
    return subprocess.run(
        [
            "bash",
            "-c",
            f'R="{root}"\n{base._derivation_script(function)}{function} omnibase_infra',
        ],
        env=scrub_git_location_env(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _count(term: str, line: str) -> str:
    return subprocess.run(
        ["grep", "-cE", term],
        input=line + "\n",
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def test_the_derived_term_matches_the_contract_name_the_runtime_logs(
    tmp_path: Path,
) -> None:
    root = _repo_with_nodes(tmp_path, {"node_host_reconcile_compute": RUNTIME_CONTRACT})
    (term,) = _host(root, "derive_live_grep").split()
    for line, expected in (
        # the consumer group and the node= field carry the contract name
        ("consumer_group=local.omnimarket.host_reconcile_compute.consume.1.0.0", "1"),
        ("[INFO] omnibase_infra.host_reconcile_compute.consume: node started", "1"),
        # the directory name still matches its own loggers
        ("omnibase_infra.nodes.node_host_reconcile_compute.handlers.h: ran", "1"),
        # discovery and look-alikes stay dead
        ("Discovered contract: host_reconcile_compute (COMPUTE) from x", "0"),
        ("consumer_group=local.omnimarket.host_reconcile_compute_more.consume", "0"),
        ("consumer_group=local.omnimarket.other_host_reconcile_compute.consume", "0"),
    ):
        assert _count(term, line) == expected, (term, line)


def test_the_term_is_the_directory_name_alone_when_the_contract_name_is_the_same(
    tmp_path: Path,
) -> None:
    root = _repo_with_nodes(tmp_path, {"node_kept": RUNTIME_CONTRACT})
    (term,) = _host(root, "derive_live_grep").split()
    assert term == "[.](node_kept|kept)[.]"
    root = _repo_with_nodes(
        tmp_path / "same",
        {"node_kept": "name: node_kept\nevent_bus:\n  subscribe_topics: [t]\n"},
    )
    (term,) = _host(root, "derive_live_grep").split()
    assert term == pool.LIVE_TERM_DOT + "node_kept" + pool.LIVE_TERM_DOT


def test_a_node_with_no_topics_and_no_runtime_profile_is_not_a_live_term(
    tmp_path: Path,
) -> None:
    root = _repo_with_nodes(
        tmp_path,
        {
            "node_pin_bump_compute": LOCAL_CONTRACT,
            "node_runs_compute": RUNTIME_CONTRACT,
            "node_profile_only": PROFILE_ONLY_CONTRACT,
        },
    )
    terms = _host(root, "derive_live_grep").split()
    assert sorted(terms) == sorted(
        [
            "[.](node_runs_compute|runs_compute)[.]",
            "[.](node_profile_only|profile_only)[.]",
        ]
    )
    assert _host(root, "derive_live_not_applicable").split() == [
        "node_pin_bump_compute"
    ]


def test_an_unreadable_contract_is_never_waived(tmp_path: Path) -> None:
    root = _repo_with_nodes(tmp_path, {"node_broken": "name: [unclosed\n"})
    assert _host(root, "derive_live_not_applicable").split() == []
    assert len(_host(root, "derive_live_grep").split()) == 1


def test_a_contract_with_a_topic_under_any_event_bus_key_is_never_waived(
    tmp_path: Path,
) -> None:
    contract = "name: {name}\nevent_bus:\n  subscribe_topics: []\n  dlq_topics: [t]\n"
    root = _repo_with_nodes(tmp_path, {"node_dlq_only": contract})
    assert _host(root, "derive_live_not_applicable").split() == []


NA_NODE = "src/omnibase_infra/nodes/node_pin_bump_compute/contract.yaml"
NA_LINE = (
    "live-grep not-applicable: node_pin_bump_compute "
    "(contract declares no topics and no runtime profile)\n"
)


def test_a_changed_local_node_is_not_applicable_not_a_failure(tmp_path: Path) -> None:
    live = (
        "live-grep derived from the diff: none (every changed node is local)\n"
        + NA_LINE
    )
    code, text, events = base._infra_run(tmp_path, base._infra_clone(NA_NODE), live)
    assert code == pool.EXIT_PASS, text
    assert events[0]["verifier_token"] == "ACCEPTED", text
    assert "changed_path_live" in events[0]["mandatory_checks"]
    (receipt,) = (tmp_path / "receipts").glob("*.receipt.json")
    (check,) = [
        c
        for c in json.loads(receipt.read_text(encoding="utf-8"))["checks"]
        if c["name"] == "changed_path_live"
    ]
    assert check["ok"] is True
    assert "not-applicable: node_pin_bump_compute" in check["evidence"]


def test_a_local_node_does_not_excuse_a_runtime_node_that_never_ran(
    tmp_path: Path,
) -> None:
    term = pool.LIVE_TERM_DOT + "node_runtime_error_triage_effect" + pool.LIVE_TERM_DOT
    live = (
        f"live-grep derived from the diff: {term} \n{NA_LINE}"
        f"  omnibase-infra-local-omninode-runtime mentions {term}: 0\n"
        f"  omnibase-infra-local-runtime-effects mentions {term}: 0\n"
    )
    clone = base._infra_clone(NA_NODE, base.NODE)
    code, text, events = base._infra_run(tmp_path, clone, live)
    assert code == pool.EXIT_FAIL, text
    assert events[0]["failing_checks"] == ["changed_path_live"]
    assert "no runtime log mentions: node_runtime_error_triage_effect\n" in text + "\n"


def test_a_local_node_beside_a_live_runtime_node_is_accepted(tmp_path: Path) -> None:
    term = pool.LIVE_TERM_DOT + "node_runtime_error_triage_effect" + pool.LIVE_TERM_DOT
    live = (
        f"live-grep derived from the diff: {term} \n{NA_LINE}"
        f"  omnibase-infra-local-omninode-runtime mentions {term}: 5\n"
        f"  omnibase-infra-local-runtime-effects mentions {term}: 0\n"
    )
    clone = base._infra_clone(NA_NODE, base.NODE)
    code, text, events = base._infra_run(tmp_path, clone, live)
    assert code == pool.EXIT_PASS, text
    assert events[0]["verifier_token"] == "ACCEPTED", text


def test_an_alternation_term_is_named_by_its_directory_in_the_failure(
    tmp_path: Path,
) -> None:
    term = (
        pool.LIVE_TERM_DOT
        + "(node_runtime_error_triage_effect|runtime_error_triage_effect)"
        + pool.LIVE_TERM_DOT
    )
    live = (
        f"  omnibase-infra-local-omninode-runtime mentions {term}: 0\n"
        f"  omnibase-infra-local-runtime-effects mentions {term}: 0\n"
    )
    code, text, _ = base._infra_run(tmp_path, base._infra_clone(base.NODE), live)
    assert code == pool.EXIT_FAIL
    assert "no runtime log mentions: node_runtime_error_triage_effect\n" in text + "\n"
