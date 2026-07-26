# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for the managed-staging canary topic/group catalog (OMN-14727, B7).

Covers:
    * the MSK IAM authorization helper (``onex.*`` / ``omninode.*``);
    * fail-closed rejection of a prefix outside the IAM patterns (the ticket's
      "a prefix outside them fails AUTH, not collision" guard);
    * deterministic catalog generation (prefix applied to every topic + group);
    * zero-collision readback -- both exact-name collision and prefix conflict;
    * the in-repo zero-collision proof against the candidate's real declared
      topic corpus (the live readback against the 1089 cluster topics is
      deferred to Phase 3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.topics.managed_staging_canary_catalog import (
    DEFAULT_CANARY_CATALOG_PATH,
    ModelCanaryNamespace,
    build_canary_catalog_from_candidate,
    extract_candidate_topic_suffixes,
    generate_canary_catalog,
    iam_pattern_authorizes,
    load_canary_namespace,
    verify_zero_collision,
)

# Repo root: tests/unit/topics/<file>.py -> parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]

_IAM_PATTERNS = ("onex.*", "omninode.*")


def _namespace(**overrides: object) -> ModelCanaryNamespace:
    """Build a valid namespace, overriding individual fields for negative tests."""
    base: dict[str, object] = {
        "ticket": "OMN-14727",
        "epoch": "mstg1",
        "topic_prefix": "onex.mstg1.",
        "group_prefix": "onex.mstg1.",
        "iam_topic_patterns": _IAM_PATTERNS,
        "iam_group_patterns": _IAM_PATTERNS,
        "default_partitions": 1,
        "default_replication_factor": 2,
        "candidate_contract_roots": ("src/omnibase_infra/nodes",),
    }
    base.update(overrides)
    return ModelCanaryNamespace.model_validate(base)


# --------------------------------------------------------------------------- #
# IAM authorization helper
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("onex.mstg1.", True),
        ("onex.mstg1.onex.evt.platform.node-registration.v1", True),
        ("omninode.mstg1.", True),
        ("org.omninode.onex.evt.platform.x.v1", False),  # starts with 'org.'
        ("onexfoo.", False),  # 'onex.' pattern requires the literal dot
        ("mstg1.onex.evt.x.v1", False),  # env-first group form fails onex.*
    ],
)
def test_iam_pattern_authorizes(name: str, expected: bool) -> None:
    assert iam_pattern_authorizes(name, _IAM_PATTERNS) is expected


def test_iam_pattern_authorizes_exact_match_without_wildcard() -> None:
    assert iam_pattern_authorizes("onex.exact", ("onex.exact",)) is True
    assert iam_pattern_authorizes("onex.exact.more", ("onex.exact",)) is False


# --------------------------------------------------------------------------- #
# Namespace validation (fail-closed on unauthorized prefix)
# --------------------------------------------------------------------------- #


def test_default_namespace_loads_and_is_authorized() -> None:
    ns = load_canary_namespace()
    assert ns.epoch == "mstg1"
    assert ns.topic_prefix == "onex.mstg1."
    assert ns.group_prefix == "onex.mstg1."
    # Both prefixes must fall inside the IAM patterns.
    assert iam_pattern_authorizes(ns.topic_prefix, ns.iam_topic_patterns)
    assert iam_pattern_authorizes(ns.group_prefix, ns.iam_group_patterns)


def test_default_catalog_file_exists() -> None:
    assert DEFAULT_CANARY_CATALOG_PATH.is_file()


def test_namespace_rejects_topic_prefix_outside_iam_patterns() -> None:
    # 'org.omninode.' is the canonical multi-bus prefix, but it starts with
    # 'org.' -> outside onex.*/omninode.* -> would fail MSK AUTH, not collide.
    with pytest.raises(ValueError, match="fails AUTH"):
        _namespace(topic_prefix="org.omninode.")


def test_namespace_rejects_group_prefix_outside_iam_patterns() -> None:
    with pytest.raises(ValueError, match="fails AUTH"):
        _namespace(group_prefix="staging.canary.")


def test_namespace_rejects_non_dot_terminated_prefix() -> None:
    with pytest.raises(ValueError, match="must end with"):
        _namespace(topic_prefix="onex.mstg1")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def test_generate_prefixes_every_topic_and_group() -> None:
    ns = _namespace()
    suffixes = [
        "onex.evt.platform.alpha.v1",
        "onex.cmd.platform.beta.v1",
        "onex.evt.platform.alpha.v1",  # duplicate collapses
    ]
    bases = ["node_a.consume.v1", "node_b.consume.v1"]
    catalog = generate_canary_catalog(ns, topic_suffixes=suffixes, group_bases=bases)
    # Duplicate suffix collapsed to 2 unique topics.
    assert len(catalog.topics) == 2
    assert len(catalog.groups) == 2
    for name in catalog.topic_names:
        assert name.startswith("onex.mstg1.")
        assert iam_pattern_authorizes(name, ns.iam_topic_patterns)
    for group in catalog.groups:
        assert group.startswith("onex.mstg1.")
        assert iam_pattern_authorizes(group, ns.iam_group_patterns)
    # Sizing carried from the namespace.
    assert all(spec.partitions == 1 for spec in catalog.topics)
    assert all(spec.replication_factor == 2 for spec in catalog.topics)


def test_generated_topic_names_are_deterministic_and_sorted() -> None:
    ns = _namespace()
    suffixes = ["onex.evt.platform.z.v1", "onex.evt.platform.a.v1"]
    catalog = generate_canary_catalog(ns, topic_suffixes=suffixes, group_bases=[])
    assert list(catalog.topic_names) == sorted(catalog.topic_names)


# --------------------------------------------------------------------------- #
# Zero-collision readback
# --------------------------------------------------------------------------- #


def test_verify_zero_collision_clean_when_disjoint() -> None:
    ns = _namespace()
    catalog = generate_canary_catalog(
        ns,
        topic_suffixes=["onex.evt.platform.alpha.v1"],
        group_bases=["node_a.consume.v1"],
    )
    report = verify_zero_collision(
        catalog,
        existing_topics=["onex.evt.platform.alpha.v1", "onex.evt.platform.beta.v1"],
        existing_groups=["dev.svc.node_a.consume.v1"],
    )
    assert report.is_clean
    assert report.colliding_topics == ()
    assert report.prefix_conflicting_topics == ()
    assert report.checked_topic_count == 1
    assert report.existing_topic_count == 2


def test_verify_detects_exact_name_collision() -> None:
    # Non-vacuous: inject an existing topic that exactly equals a canary name.
    ns = _namespace()
    catalog = generate_canary_catalog(
        ns, topic_suffixes=["onex.evt.platform.alpha.v1"], group_bases=[]
    )
    canary_name = catalog.topic_names[0]
    report = verify_zero_collision(
        catalog,
        existing_topics=[canary_name],
        existing_groups=[],
    )
    assert not report.is_clean
    assert report.colliding_topics == (canary_name,)


def test_verify_detects_prefix_conflict_topics_and_groups() -> None:
    ns = _namespace()
    catalog = generate_canary_catalog(
        ns,
        topic_suffixes=["onex.evt.platform.alpha.v1"],
        group_bases=["node_a.consume.v1"],
    )
    # An unrelated existing name that nonetheless lives under the canary prefix.
    report = verify_zero_collision(
        catalog,
        existing_topics=["onex.mstg1.some.other.topic.v1"],
        existing_groups=["onex.mstg1.some.other.group"],
    )
    assert not report.is_clean
    assert report.prefix_conflicting_topics == ("onex.mstg1.some.other.topic.v1",)
    assert report.prefix_conflicting_groups == ("onex.mstg1.some.other.group",)


# --------------------------------------------------------------------------- #
# In-repo zero-collision PROOF against the candidate's real declared corpus.
# The LIVE readback against the 1089 cluster topics is deferred to Phase 3.
# --------------------------------------------------------------------------- #


def test_in_repo_zero_collision_proof_against_declared_corpus() -> None:
    ns = load_canary_namespace()
    # The candidate's contract-owned topic suffixes actually declared in-repo.
    declared_suffixes = extract_candidate_topic_suffixes(
        ns.candidate_contract_roots, base_dir=_REPO_ROOT
    )
    # Non-vacuous: there must be a real corpus to check against.
    assert len(declared_suffixes) > 0

    catalog = build_canary_catalog_from_candidate(ns, base_dir=_REPO_ROOT)
    assert len(catalog.topics) == len(declared_suffixes)
    assert len(catalog.groups) > 0

    # Prove the prefix is what creates disjointness (EXISTS-but-WRONG guard):
    # strip the prefix off every canary name and it lands back in the declared
    # corpus -- i.e. WITHOUT the prefix the readback WOULD flag full collision.
    corpus = set(declared_suffixes)
    for name in catalog.topic_names:
        assert name.startswith(ns.topic_prefix)
        assert name[len(ns.topic_prefix) :] in corpus

    # Readback: the prefixed canary catalog collides with ZERO declared topics.
    report = verify_zero_collision(
        catalog,
        existing_topics=declared_suffixes,
        existing_groups=[],
    )
    assert report.is_clean, (
        f"canary namespace collides with declared corpus: "
        f"colliding={report.colliding_topics} "
        f"prefix_conflicts={report.prefix_conflicting_topics}"
    )
    assert report.checked_topic_count == len(declared_suffixes)
    assert report.existing_topic_count == len(declared_suffixes)
