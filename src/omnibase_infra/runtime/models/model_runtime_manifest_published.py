# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wire payload for ``onex.evt.omnibase-infra.runtime-manifest-published.v1``.

This is the canonical boot snapshot published once per startup at
``service_kernel`` step 9.8. It is ``ModelRuntimeManifest`` (what the runtime
WIRED) plus one additive field, ``attach_readiness`` (what actually ATTACHED,
and for every contract that did not, which topics failed readiness confirm).

Why a subclass and not a second model
-------------------------------------
``ModelRuntimeManifest`` is ``frozen``/``extra="forbid"`` and lives in
``omnibase_core``, which ``omnibase_infra`` consumes at an immutable pinned rev
(``pyproject.toml [tool.uv.sources]``). Every attach model
(``ModelRuntimeAttachReadiness``, ``ModelContractAttachResult``,
``ModelTopicSetReadiness``) is ``omnibase_infra``-resident and core cannot
import infra (layering: compat -> core -> spi -> infra), so the field cannot be
added to the base without first relocating the attach models into core.

Subclassing keeps ONE model per shape — every base field is inherited, none is
redeclared — and is wire-compatible: the serialized payload is byte-identical
to the previous one plus a single ``attach_readiness`` key. Existing consumers
(``node_runtime_manifest_reducer`` here, ``node_redeploy_orchestrator`` in
omnimarket, which declares the subscription but decodes no typed model) read
the keys they already read. ``contract_hash`` / ``topology_hash`` are inherited
computed fields and are unchanged by this addition, so manifest dedup and drift
detection keep their existing values.

Residual (tracked on OMN-15512): when ``omnibase_core`` next cuts a release
that carries the attach models, fold ``attach_readiness`` onto
``ModelRuntimeManifest`` itself and delete this subclass.

Related Tickets:
    - OMN-11196: Emit the runtime manifest snapshot at boot.
    - OMN-11197: Persist it to the ``runtime_manifests`` projection.
    - OMN-13237: Per-contract provision -> confirm-ready -> attach interleave.
    - OMN-15512: Fold the attach-readiness aggregate onto this payload.
"""

from __future__ import annotations

from pydantic import Field

from omnibase_core.models.runtime_manifest.model_runtime_manifest import (
    ModelRuntimeManifest,
)
from omnibase_infra.event_bus.model_runtime_attach_readiness import (
    ModelRuntimeAttachReadiness,
)


class ModelRuntimeManifestPublished(ModelRuntimeManifest):
    """Runtime manifest snapshot carrying the boot attach-readiness aggregate.

    Attributes:
        attach_readiness: Boot attach-readiness aggregate. ``None`` only when
            the per-contract interleave did not run (auto-wiring disabled or
            failed before subscribe), which is distinct from "ran and every
            contract attached" — that case carries a ``READY`` aggregate with
            an empty ``results``. The published copy is narrowed to the blocker
            set via :meth:`ModelRuntimeAttachReadiness.blockers_only`, so
            ``required_contracts - attached_contracts == len(results)``.
    """

    attach_readiness: ModelRuntimeAttachReadiness | None = Field(default=None)


__all__: list[str] = ["ModelRuntimeManifestPublished"]
