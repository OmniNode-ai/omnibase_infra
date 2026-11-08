# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22870 -- the lane model accepts a pattern-declared secret namespace.

WHAT WENT RED. The lane table's dev-lane ``secret_resolver`` gained a
``namespaces`` block beside its ``mappings`` list (OMN-22870, producer side,
2026-11-07): the iLO per-host account names declared by PATTERN, so a host the
setup verb adds later is covered by a rule already written. ``ModelCiBusOverlay``
is ``extra="forbid"``, so the consumer-side parse refused the producer-side
file and the overlay binding gate went red on ``dev`` itself (job
128488372447, 2026-11-07T18:22:24Z) and on every open pull request behind it::

    Extra inputs are not permitted [type=extra_forbidden,
      input_value=[{'namespace': 'ilo_accounts', ...}], input_type=list]

This is the fourth occurrence of the one failure mode OMN-18012, OMN-18060 and
OMN-20275 each recorded: the producer grows a key, the consumer has not learned
it, and the symptom arrives as "no merge rebuilds the runtime".

WHY THESE CASES ARE HERMETIC AND test_ci_bus_overlay_live_binding.py IS NOT.
That file validates whatever the workflow checked out, which is the right guard
and the wrong artefact: it skips without ``CI_BUS_OVERLAY_LIVE_PATH`` and
therefore proves nothing at a merge base. The block below is the live shape
stated IN this repository, so these cases fail at the merge base and pass with
the typed field -- and keep stating the intent after the lane table moves again.

``extra="forbid"`` is not relaxed anywhere here. Admitting the key by typing it
is the fix; admitting it by widening the model would restore exactly the silent
default-routing hole the strictness defends, and
test_omn_18083_lane_model_tracks_the_live_overlay.py pins that it stays.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"

#: The dev lane's secret resolver as the live lane table declares it
#: (config/lab/ci_bus_lanes.yaml in the CI overlay repository, at OMN-22870).
#: A template, a pattern and a folder -- never a secret value.
LIVE_SECRET_RESOLVER: dict[str, Any] = {
    "enable_convention_fallback": False,
    "mappings": [
        {
            "logical_name": "GITHUB_TOKEN",
            "source": {
                "source_type": "infisical",
                "source_path": "/shared/env/GITHUB_TOKEN",
            },
        },
        {
            "logical_name": "ILO_LAB_CA_BUNDLE_PEM",
            "source": {
                "source_type": "infisical",
                "source_path": "/shared/env/ILO_LAB_CA_BUNDLE_PEM",
            },
        },
    ],
    "namespaces": [
        {
            "namespace": "ilo_accounts",
            "ref_pattern": "^ILO_[A-Z0-9_]+_(READ|WRITE)_(USERNAME|PASSWORD)$",
            "source_type": "infisical",
            "source_path_template": "/shared/env/{ref}",
        }
    ],
}

pytestmark = pytest.mark.unit


def _import_trigger_module() -> Any:
    """Import the publisher by file path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "trigger_rebuild_on_merge_omn22870", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _overlay(secret_resolver: dict[str, Any]) -> dict[str, Any]:
    return {
        "default": "inmemory",
        "lanes": {"dev": {"broker": "inmemory", "secret_resolver": secret_resolver}},
    }


def test_the_live_namespaces_block_validates() -> None:
    """AC1. The exact producer-side block that reddened dev now parses.

    At the merge base this raises ValidationError on
    ``lanes.dev.secret_resolver.namespaces`` -- the failure the binding gate
    reported -- so this case is the artefact of the fix, not a restatement.
    """
    mod = _import_trigger_module()
    parsed = mod.ModelCiBusOverlay.model_validate(_overlay(LIVE_SECRET_RESOLVER))

    resolver = parsed.lanes["dev"].secret_resolver
    assert resolver is not None
    assert len(resolver.namespaces) == 1
    declared = resolver.namespaces[0]
    assert declared.namespace == "ilo_accounts"
    assert declared.source_type == "infisical"
    assert declared.source_path_template == "/shared/env/{ref}"
    assert declared.ref_pattern == ("^ILO_[A-Z0-9_]+_(READ|WRITE)_(USERNAME|PASSWORD)$")
    # The namespace is a sibling of the mappings, not a replacement for them.
    assert [m.logical_name for m in resolver.mappings] == [
        "GITHUB_TOKEN",
        "ILO_LAB_CA_BUNDLE_PEM",
    ]


def test_a_resolver_without_namespaces_is_still_accepted() -> None:
    """Every other lane declares no namespaces; the field is optional."""
    mod = _import_trigger_module()
    resolver = dict(LIVE_SECRET_RESOLVER)
    resolver.pop("namespaces")

    parsed = mod.ModelCiBusOverlay.model_validate(_overlay(resolver))
    declared = parsed.lanes["dev"].secret_resolver
    assert declared is not None
    assert declared.namespaces == ()


def test_an_unknown_key_inside_a_namespace_is_refused() -> None:
    """AC2. The nested block is typed, not a free dict.

    A ``dict[str, str]`` would admit ``source_value``, which carries a secret
    BY VALUE -- the one shape the overlay's contract forbids outright. Typing
    the namespace keeps that refusal, so the fix cannot be read as a widening.
    """
    mod = _import_trigger_module()
    namespace = dict(LIVE_SECRET_RESOLVER["namespaces"][0])
    namespace["source_value"] = "hunter2"
    resolver = dict(LIVE_SECRET_RESOLVER) | {"namespaces": [namespace]}

    with pytest.raises(ValidationError, match="source_value"):
        mod.ModelCiBusOverlay.model_validate(_overlay(resolver))


def test_a_misspelled_namespaces_key_is_still_refused() -> None:
    """AC2. The strictness around the new field is intact.

    If the fix had been ``extra="allow"`` or ``extra="ignore"`` on the
    resolver, this typo would be accepted and the resolver would silently
    resolve nothing by pattern at all.
    """
    mod = _import_trigger_module()
    resolver = dict(LIVE_SECRET_RESOLVER)
    resolver["namespacess"] = resolver.pop("namespaces")

    with pytest.raises(ValidationError, match="namespacess"):
        mod.ModelCiBusOverlay.model_validate(_overlay(resolver))


def test_a_namespace_missing_its_template_is_refused() -> None:
    """A rule with no destination resolves nothing; it must not parse."""
    mod = _import_trigger_module()
    namespace = dict(LIVE_SECRET_RESOLVER["namespaces"][0])
    del namespace["source_path_template"]
    resolver = dict(LIVE_SECRET_RESOLVER) | {"namespaces": [namespace]}

    with pytest.raises(ValidationError, match="source_path_template"):
        mod.ModelCiBusOverlay.model_validate(_overlay(resolver))
