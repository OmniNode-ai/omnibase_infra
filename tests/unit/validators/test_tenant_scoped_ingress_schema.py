# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the tenant_scoped_ingress opt-in gate (OMN-14360 / OMN-14208 / OMN-14482)."""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.validators.tenant_scoped_ingress_schema import (
    AllowlistEntry,
    collect_published_topics,
    load_allowlist,
    main,
    validate_file,
)

_PREFIXED_TOPIC = "tenant-acme.onex.cmd.omnimarket.delegate-skill.v1"
_BARE_TOPIC = "onex.cmd.omnimarket.delegate-skill.v1"
_NO_PRODUCERS: frozenset[str] = frozenset()


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _contract(
    *,
    flag: bool,
    topics: list[str],
    node_id: str = "node_foo",
    publish_topics: list[str] | None = None,
) -> str:
    if topics:
        subscribe = "  subscribe_topics:\n" + "\n".join(f"    - {t}" for t in topics)
    else:
        subscribe = "  subscribe_topics: []"
    publish = ""
    if publish_topics:
        publish = "\n  publish_topics:\n" + "\n".join(
            f"    - {t}" for t in publish_topics
        )
    return (
        f"name: {node_id}\n"
        "node_type: EFFECT_GENERIC\n"
        "event_bus:\n"
        f"  tenant_scoped_ingress: {'true' if flag else 'false'}\n"
        f"{subscribe}{publish}\n"
    )


def _allowlist(
    node_id: str,
    seam_test: str,
    trusted_internal: list[str] | None = None,
) -> dict[str, AllowlistEntry]:
    return {
        node_id: AllowlistEntry(
            seam_test=seam_test,
            trusted_internal_topics=frozenset(trusted_internal or ()),
        )
    }


def test_flag_true_all_prefixed_with_allowlist_passes(tmp_path: Path) -> None:
    f = _write(
        tmp_path, "contract.yaml", _contract(flag=True, topics=[_PREFIXED_TOPIC])
    )
    findings = validate_file(
        f, _allowlist("node_foo", "tests/integration/test_x.py"), _NO_PRODUCERS
    )
    assert findings == []


def test_flag_true_bare_topic_fails(tmp_path: Path) -> None:
    f = _write(tmp_path, "contract.yaml", _contract(flag=True, topics=[_BARE_TOPIC]))
    findings = validate_file(
        f, _allowlist("node_foo", "tests/integration/test_x.py"), _NO_PRODUCERS
    )
    kinds = {fd.kind for fd in findings}
    assert "BARE_OR_MIXED_TOPIC" in kinds
    # bare-only contract also has no prefixed topic → no-op opt-in guard fires.
    assert "NO_PREFIXED_TOPIC" in kinds
    assert any(_BARE_TOPIC in fd.detail for fd in findings)


def test_flag_true_mixed_topics_flags_only_bare(tmp_path: Path) -> None:
    f = _write(
        tmp_path,
        "contract.yaml",
        _contract(flag=True, topics=[_PREFIXED_TOPIC, _BARE_TOPIC]),
    )
    findings = validate_file(
        f, _allowlist("node_foo", "tests/integration/test_x.py"), _NO_PRODUCERS
    )
    bare = [fd for fd in findings if fd.kind == "BARE_OR_MIXED_TOPIC"]
    assert len(bare) == 1
    assert _BARE_TOPIC in bare[0].detail
    # a prefixed topic is present, so the no-op guard must NOT fire.
    assert not any(fd.kind == "NO_PREFIXED_TOPIC" for fd in findings)


def test_flag_true_missing_allowlist_fails(tmp_path: Path) -> None:
    f = _write(
        tmp_path, "contract.yaml", _contract(flag=True, topics=[_PREFIXED_TOPIC])
    )
    findings = validate_file(f, {}, _NO_PRODUCERS)  # empty allowlist
    assert {fd.kind for fd in findings} == {"MISSING_SEAM_TEST"}


def test_flag_true_empty_allowlist_seam_test_fails(tmp_path: Path) -> None:
    f = _write(
        tmp_path, "contract.yaml", _contract(flag=True, topics=[_PREFIXED_TOPIC])
    )
    # node present but seam_test is empty string — still fails closed.
    findings = validate_file(f, _allowlist("node_foo", ""), _NO_PRODUCERS)
    assert {fd.kind for fd in findings} == {"MISSING_SEAM_TEST"}


def test_flag_true_no_subscribe_topics_is_no_op_optin(tmp_path: Path) -> None:
    f = _write(tmp_path, "contract.yaml", _contract(flag=True, topics=[]))
    findings = validate_file(
        f, _allowlist("node_foo", "tests/integration/test_x.py"), _NO_PRODUCERS
    )
    assert {fd.kind for fd in findings} == {"NO_SUBSCRIBE_TOPICS"}


def test_flag_false_is_ignored(tmp_path: Path) -> None:
    f = _write(tmp_path, "contract.yaml", _contract(flag=False, topics=[_BARE_TOPIC]))
    # No opt-in → gate does not apply even with a bare topic and no allowlist.
    assert validate_file(f, {}, _NO_PRODUCERS) == []


def test_flag_absent_is_ignored(tmp_path: Path) -> None:
    body = f"name: node_bar\nevent_bus:\n  subscribe_topics:\n    - {_BARE_TOPIC}\n"
    f = _write(tmp_path, "contract.yaml", body)
    assert validate_file(f, {}, _NO_PRODUCERS) == []


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
    assert validate_file(f, {}, _NO_PRODUCERS) == []


# ---------------------------------------------------------------------------
# OMN-14482 trusted-internal carve-out
# ---------------------------------------------------------------------------


def test_trusted_internal_bare_topic_carveout_passes_when_no_producer(
    tmp_path: Path,
) -> None:
    """A mixed contract (prefixed + a trusted-internal bare topic) is clean.

    This is the B enablement shape: keep the internal command topic, add the
    tenant-prefixed external topic, and allowlist the bare one as
    trusted_internal. With NO in-cluster producer for the bare topic, the
    carve-out is granted and there are zero findings.
    """
    f = _write(
        tmp_path,
        "contract.yaml",
        _contract(flag=True, topics=[_PREFIXED_TOPIC, _BARE_TOPIC]),
    )
    allowlist = _allowlist(
        "node_foo", "tests/integration/test_x.py", trusted_internal=[_BARE_TOPIC]
    )
    findings = validate_file(f, allowlist, _NO_PRODUCERS)
    assert findings == []


def test_trusted_internal_carveout_rejected_when_a_producer_exists(
    tmp_path: Path,
) -> None:
    """RED case: the mislabel cannot dodge the leak check.

    If ANY contract publishes to the bare topic, it is not a pure
    trusted-internal entry point (a client tenant_id could ride in on that
    producer's leg), so the carve-out is REJECTED even though it is allowlisted.
    """
    f = _write(
        tmp_path,
        "contract.yaml",
        _contract(flag=True, topics=[_PREFIXED_TOPIC, _BARE_TOPIC]),
    )
    allowlist = _allowlist(
        "node_foo", "tests/integration/test_x.py", trusted_internal=[_BARE_TOPIC]
    )
    # a producer exists for the bare topic → carve-out must fail closed.
    findings = validate_file(f, allowlist, frozenset({_BARE_TOPIC}))
    assert {fd.kind for fd in findings} == {"TRUSTED_INTERNAL_HAS_PRODUCER"}
    assert any(_BARE_TOPIC in fd.detail for fd in findings)


def test_trusted_internal_only_bare_is_no_op_optin(tmp_path: Path) -> None:
    """A flag-true contract whose ONLY topic is a trusted-internal bare topic
    never fires the stamp — the no-op opt-in guard must fire."""
    f = _write(tmp_path, "contract.yaml", _contract(flag=True, topics=[_BARE_TOPIC]))
    allowlist = _allowlist(
        "node_foo", "tests/integration/test_x.py", trusted_internal=[_BARE_TOPIC]
    )
    findings = validate_file(f, allowlist, _NO_PRODUCERS)
    assert {fd.kind for fd in findings} == {"NO_PREFIXED_TOPIC"}


def test_trusted_internal_for_unsubscribed_topic_is_flagged(tmp_path: Path) -> None:
    f = _write(
        tmp_path, "contract.yaml", _contract(flag=True, topics=[_PREFIXED_TOPIC])
    )
    allowlist = _allowlist(
        "node_foo",
        "tests/integration/test_x.py",
        trusted_internal=["onex.cmd.omnimarket.not-subscribed.v1"],
    )
    findings = validate_file(f, allowlist, _NO_PRODUCERS)
    assert {fd.kind for fd in findings} == {"TRUSTED_INTERNAL_NOT_SUBSCRIBED"}


def test_trusted_internal_naming_a_prefixed_topic_is_flagged(tmp_path: Path) -> None:
    f = _write(
        tmp_path, "contract.yaml", _contract(flag=True, topics=[_PREFIXED_TOPIC])
    )
    allowlist = _allowlist(
        "node_foo",
        "tests/integration/test_x.py",
        trusted_internal=[_PREFIXED_TOPIC],  # already stamped → needs no carve-out
    )
    findings = validate_file(f, allowlist, _NO_PRODUCERS)
    assert {fd.kind for fd in findings} == {"TRUSTED_INTERNAL_IS_PREFIXED"}


# ---------------------------------------------------------------------------
# allowlist + producer-scan loaders
# ---------------------------------------------------------------------------


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
            "    trusted_internal_topics:\n"
            f"      - {_BARE_TOPIC}\n"
            "  - node_id: node_bar\n"
            "    seam_test: tests/integration/test_bar.py\n"
        ),
    )
    mapping = load_allowlist(p)
    assert mapping["node_foo"] == AllowlistEntry(
        seam_test="tests/integration/test_foo.py",
        trusted_internal_topics=frozenset({_BARE_TOPIC}),
    )
    assert mapping["node_bar"] == AllowlistEntry(
        seam_test="tests/integration/test_bar.py",
        trusted_internal_topics=frozenset(),
    )


def test_collect_published_topics_indexes_publish_topics(tmp_path: Path) -> None:
    producer_dir = tmp_path / "node_producer"
    producer_dir.mkdir()
    _write(
        producer_dir,
        "contract.yaml",
        _contract(
            flag=False,
            topics=["onex.cmd.something.v1"],
            node_id="node_producer",
            publish_topics=[_BARE_TOPIC, "onex.evt.other.v1"],
        ),
    )
    published = collect_published_topics(tmp_path)
    assert _BARE_TOPIC in published
    assert "onex.evt.other.v1" in published


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


def test_main_carveout_end_to_end_passes_when_no_producer(tmp_path: Path) -> None:
    """main() with a real allowlist file + producer scan: mixed contract with a
    trusted-internal carve-out and no producer for the bare topic → exit 0."""
    node_dir = tmp_path / "node_foo"
    node_dir.mkdir()
    _write(
        node_dir,
        "contract.yaml",
        _contract(flag=True, topics=[_PREFIXED_TOPIC, _BARE_TOPIC]),
    )
    allowlist = _write(
        tmp_path,
        "allowlist.yaml",
        (
            "allowlist:\n"
            "  - node_id: node_foo\n"
            "    seam_test: tests/integration/test_tenant_stamp_seam_omn14208.py\n"
            "    trusted_internal_topics:\n"
            f"      - {_BARE_TOPIC}\n"
        ),
    )
    rc = main(
        [
            str(node_dir),
            "--allowlist",
            str(allowlist),
            "--producer-scan-root",
            str(node_dir),
        ]
    )
    assert rc == 0


def test_main_carveout_end_to_end_rejected_when_producer_present(
    tmp_path: Path,
) -> None:
    """RED end-to-end: a sibling contract publishes the bare topic, so the
    carve-out is mechanically rejected and main() exits 1."""
    scan_root = tmp_path / "src"
    node_dir = scan_root / "node_foo"
    node_dir.mkdir(parents=True)
    _write(
        node_dir,
        "contract.yaml",
        _contract(flag=True, topics=[_PREFIXED_TOPIC, _BARE_TOPIC]),
    )
    producer_dir = scan_root / "node_producer"
    producer_dir.mkdir()
    _write(
        producer_dir,
        "contract.yaml",
        _contract(
            flag=False,
            topics=["onex.cmd.trigger.v1"],
            node_id="node_producer",
            publish_topics=[_BARE_TOPIC],
        ),
    )
    allowlist = _write(
        tmp_path,
        "allowlist.yaml",
        (
            "allowlist:\n"
            "  - node_id: node_foo\n"
            "    seam_test: tests/integration/test_tenant_stamp_seam_omn14208.py\n"
            "    trusted_internal_topics:\n"
            f"      - {_BARE_TOPIC}\n"
        ),
    )
    rc = main(
        [
            str(node_dir),
            "--allowlist",
            str(allowlist),
            "--producer-scan-root",
            str(scan_root),
        ]
    )
    assert rc == 1
