# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A pull request is labelled ``runtime-train`` or ``no-runtime-train`` (OMN-19568).

Lab-proof plan T4 (``beta/plans/2026-09-25-lab-proof-for-every-pr-plan.md``,
section 4, section 7): "a PR that does not touch the deployed runtime never
enters the rebuild train ... it is classified and labelled so it never enters
the rebuild train". Falsifiers named in the plan: a docs-only PR and a
workflow-only PR are labelled ``no-runtime-train``; a PR touching
``src/omnimarket/nodes/`` is labelled ``runtime-train``.

This module's ``decide_label`` calls no path list of its own -- it is a thin
wrapper over ``scripts/runtime_change_classifier.py``'s ``classify_runtime_paths``
and ``is_runtime_affecting``, the SAME functions the post-merge rebuild
trigger and the release train call. The hermetic canonical-classifier double
in ``tests/fixtures/runtime_path_classifier.py`` stands in for omniclaude's
deploy-gate validator, exactly as the trigger's own test suite uses it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "label_runtime_train.py"
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "runtime_path_classifier.py"


def _load_module(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def label_runtime_train() -> object:
    return _load_module(_SCRIPT, "_label_runtime_train_omn19568")


@pytest.fixture
def canonical_classifier(label_runtime_train: object) -> object:
    fixture = _load_module(_FIXTURE, "_runtime_path_classifier_fixture_omn19568")
    return fixture.find_runtime_paths


# ------------------------------------------------------------------ decide_label


def test_docs_only_pr_is_no_runtime_train(
    label_runtime_train: object, canonical_classifier: object
) -> None:
    changed = ["README.md", "docs/architecture/some-decision.md"]
    label = label_runtime_train.decide_label(changed, [], canonical_classifier)
    assert label == label_runtime_train.LABEL_NO_RUNTIME_TRAIN


def test_workflow_only_pr_is_no_runtime_train(
    label_runtime_train: object, canonical_classifier: object
) -> None:
    changed = [".github/workflows/runtime-rebuild-trigger.yml"]
    label = label_runtime_train.decide_label(changed, [], canonical_classifier)
    assert label == label_runtime_train.LABEL_NO_RUNTIME_TRAIN


def test_nodes_change_is_runtime_train(
    label_runtime_train: object, canonical_classifier: object
) -> None:
    changed = ["src/omnimarket/nodes/some_node/handler.py"]
    label = label_runtime_train.decide_label(
        changed, [], canonical_classifier, source_repo="omnimarket"
    )
    assert label == label_runtime_train.LABEL_RUNTIME_TRAIN


def test_mixed_docs_and_nodes_pr_is_runtime_train(
    label_runtime_train: object, canonical_classifier: object
) -> None:
    """One runtime-affecting path in the diff is enough (union, not majority)."""
    changed = ["README.md", "src/omnimarket/nodes/some_node/handler.py"]
    label = label_runtime_train.decide_label(
        changed, [], canonical_classifier, source_repo="omnimarket"
    )
    assert label == label_runtime_train.LABEL_RUNTIME_TRAIN


def test_runtime_change_override_label_wins_on_a_docs_only_diff(
    label_runtime_train: object, canonical_classifier: object
) -> None:
    """The manual ``runtime_change`` override (OMN-19318) still decides here.

    Same predicate as the trigger: a docs-only diff with the operator's
    override label present is still runtime-train, because a second copy of
    ``is_runtime_affecting`` that ignored the label would disagree with the
    trigger on the same PR.
    """
    changed = ["README.md"]
    label = label_runtime_train.decide_label(
        changed, ["runtime_change"], canonical_classifier
    )
    assert label == label_runtime_train.LABEL_RUNTIME_TRAIN


def test_reuses_the_one_classifier_module_object(
    label_runtime_train: object, canonical_classifier: object
) -> None:
    """Loaded under the shared sys.modules name -- not a second copy of the list."""
    rcc = label_runtime_train._load_runtime_change_classifier()
    assert sys.modules[label_runtime_train._CLASSIFIER_MODULE_NAME] is rcc
    assert rcc.__file__.endswith("scripts/runtime_change_classifier.py")


# ------------------------------------------------------------------ label_actions


def test_label_actions_adds_runtime_train_when_absent(
    label_runtime_train: object,
) -> None:
    to_add, to_remove = label_runtime_train.label_actions(
        label_runtime_train.LABEL_RUNTIME_TRAIN, []
    )
    assert to_add == [label_runtime_train.LABEL_RUNTIME_TRAIN]
    assert to_remove == []


def test_label_actions_swaps_the_pair(label_runtime_train: object) -> None:
    """A PR that flips from runtime-train to no-runtime-train removes the old one."""
    to_add, to_remove = label_runtime_train.label_actions(
        label_runtime_train.LABEL_NO_RUNTIME_TRAIN,
        [label_runtime_train.LABEL_RUNTIME_TRAIN, "bug"],
    )
    assert to_add == [label_runtime_train.LABEL_NO_RUNTIME_TRAIN]
    assert to_remove == [label_runtime_train.LABEL_RUNTIME_TRAIN]


def test_label_actions_is_idempotent(label_runtime_train: object) -> None:
    to_add, to_remove = label_runtime_train.label_actions(
        label_runtime_train.LABEL_RUNTIME_TRAIN,
        [label_runtime_train.LABEL_RUNTIME_TRAIN],
    )
    assert to_add == []
    assert to_remove == []


def test_label_actions_never_touches_an_unrelated_label(
    label_runtime_train: object,
) -> None:
    to_add, to_remove = label_runtime_train.label_actions(
        label_runtime_train.LABEL_RUNTIME_TRAIN, ["bug", "needs-review"]
    )
    assert to_add == [label_runtime_train.LABEL_RUNTIME_TRAIN]
    assert to_remove == []


def test_label_actions_refuses_an_unmanaged_label(label_runtime_train: object) -> None:
    with pytest.raises(ValueError, match="not a managed label"):
        label_runtime_train.label_actions("some-other-label", [])
