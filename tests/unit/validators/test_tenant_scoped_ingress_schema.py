# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the tenant_scoped_ingress opt-in gate (OMN-14360 / OMN-14208)."""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.validators.tenant_scoped_ingress_schema import (
    load_allowlist,
    main,
    validate_file,
)

_PREFIXED_TOPIC = "tenant-acme.onex.cmd.omnimarket.delegate-skill.v1"
_BARE_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _contract(*, flag: bool, topics: list[str], node_id: str = "node_foo") -> str:
    topic_lines = "\n".join(f"    - {t}" for t in topics) or "    []"
    if topics:
        subscribe = "  subscribe_topics:\n" + topic_lines
    else:
        subscribe = "  subscribe_topics: []"
    return (
        f"name: {node_id}\n"
        "node_type: EFFECT_GENERIC\n"
        "event_bus:\n"
        f"  tenant_scoped_ingress: {'true' if flag else 'false'}\n"
        f"{subscribe}\n"
    )


def _allowlist(node_id: str, seam_test: str) -> dict[str, str]:
    return {node_id: seam_test}


def test_flag_true_all_prefixed_with_allowlist_passes(tmp_path: Path) -> None:
    f = _write(
        tmp_path, "contract.yaml", _contract(flag=True, topics=[_PREFIXED_TOPIC])
    )
    findings = validate_file(f, _allowlist("node_foo", "tests/integration/test_x.py"))
    assert findings == []


def test_flag_true_bare_topic_fails(tmp_path: Path) -> None:
    f = _write(tmp_path, "contract.yaml", _contract(flag=True, topics=[_BARE_TOPIC]))
    findings = validate_file(f, _allowlist("node_foo", "tests/integration/test_x.py"))
    kinds = {fd.kind for fd in findings}
    assert "BARE_OR_MIXED_TOPIC" in kinds
    assert any(_BARE_TOPIC in fd.detail for fd in findings)


def test_flag_true_mixed_topics_flags_only_bare(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "contract.yaml",
        _contract(flag=True, topics=[_PREFIXED_TOPIC, _BARE_TOPIC]),
    )
    findings = validate_file(f, _allowlist("node_foo", "tests/integration/test_x.py"))
    bare = [fd for fd in findings if fd.kind == "BARE_OR_MIXED_TOPIC"]
    assert len(bare) == 1
    assert _BARE_TOPIC in bare[0].detail


def test_flag_true_missing_allowlist_fails(tmp_path: Path) -> None:
    f = _write(
        tmp_path, "contract.yaml", _contract(flag=True, topics=[_PREFIXED_TOPIC])
    )
    findings = validate_file(f, {})  # empty allowlist
    assert {fd.kind for fd in findings} == {"MISSING_SEAM_TEST"}


def test_flag_true_empty_allowlist_seam_test_fails(tmp_path: Path) -> None:
    f = _write(
        tmp_path, "contract.yaml", _contract(flag=True, topics=[_PREFIXED_TOPIC])
    )
    # node present but seam_test is empty string — still fails closed.
    findings = validate_file(f, _allowlist("node_foo", ""))
    assert {fd.kind for fd in findings} == {"MISSING_SEAM_TEST"}


def test_flag_true_no_subscribe_topics_is_no_op_optin(tmp_path: Path) -> None:
    f = _write(tmp_path, "contract.yaml", _contract(flag=True, topics=[]))
    findings = validate_file(f, _allowlist("node_foo", "tests/integration/test_x.py"))
    assert {fd.kind for fd in findings} == {"NO_SUBSCRIBE_TOPICS"}


def test_flag_false_is_ignored(tmp_path: Path) -> None:
    f = _write(tmp_path, "contract.yaml", _contract(flag=False, topics=[_BARE_TOPIC]))
    # No opt-in → gate does not apply even with a bare topic and no allowlist.
    assert validate_file(f, {}) == []


def test_flag_absent_is_ignored(tmp_path: Path) -> None:
    body = f"name: node_bar\nevent_bus:\n  subscribe_topics:\n    - {_BARE_TOPIC}\n"
    f = _write(tmp_path, "contract.yaml", body)
    assert validate_file(f, {}) == []


def test_truthy_string_flag_does_not_optin(tmp_path: Path) -> None:
    # Only an explicit boolean True opts in; a string "true" must NOT.
    body = (
        "name: node_baz\n"
        "event_bus:\n"
        '  tenant_scoped_ingress: "yes"\n'
        "  subscribe_topics:\n"
        f"    - {_BARE_TOPIC}\n"
    )
    f = _write(tmp_path, "contract.yaml", body)
    assert validate_file(f, {}) == []


def test_load_allowlist_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_allowlist(tmp_path / "does_not_exist.yaml") == {}


def test_load_allowlist_parses_entries(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "allowlist.yaml",
        (
            "allowlist:\n"
            "  - node_id: node_foo\n"
            "    seam_test: tests/integration/test_foo.py\n"
            "  - node_id: node_bar\n"
            "    seam_test: tests/integration/test_bar.py\n"
        ),
    )
    mapping = load_allowlist(p)
    assert mapping == {
        "node_foo": "tests/integration/test_foo.py",
        "node_bar": "tests/integration/test_bar.py",
    }


def test_main_exit_1_on_violation(tmp_path: Path) -> None:
    node_dir = tmp_path / "node_foo"
    node_dir.mkdir()
    _write(node_dir, "contract.yaml", _contract(flag=True, topics=[_BARE_TOPIC]))
    empty_allowlist = _write(tmp_path, "allowlist.yaml", "allowlist: []\n")
    rc = main([str(node_dir), "--allowlist", str(empty_allowlist)])
    assert rc == 1


def test_main_exit_0_when_clean(tmp_path: Path) -> None:
    node_dir = tmp_path / "node_foo"
    node_dir.mkdir()
    _write(node_dir, "contract.yaml", _contract(flag=True, topics=[_PREFIXED_TOPIC]))
    allowlist = _write(
        tmp_path,
        "allowlist.yaml",
        (
            "allowlist:\n"
            "  - node_id: node_foo\n"
            "    seam_test: tests/integration/test_tenant_stamp_seam_omn14208.py\n"
        ),
    )
    rc = main([str(node_dir), "--allowlist", str(allowlist)])
    assert rc == 0
