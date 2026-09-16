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

from pydantic import Field, model_validator

from omnibase_core.models.runtime_manifest.model_runtime_manifest import (
    ModelRuntimeManifest,
)
from omnibase_infra.event_bus.model_runtime_attach_readiness import (
    ModelRuntimeAttachReadiness,
)

# ``contract_hash`` and ``topology_hash`` are pydantic ``computed_field``s on the
# base: serialized on the way OUT, rejected on the way IN because the base is
# ``extra="forbid"``. Naming them here rather than widening the model keeps every
# genuinely unknown key a hard failure.
_COMPUTED_FIELD_NAMES = ("contract_hash", "topology_hash")


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

    @model_validator(mode="before")
    @classmethod
    def _accept_own_serialized_form(cls, data: object) -> object:
        """Make the published payload round-trippable through this model.

        OMN-17296: this model is what goes ON the wire, and ``model_dump``
        includes the base's two ``computed_field``s. The base is
        ``extra="forbid"``, so feeding that same dict back to ``model_validate``
        — which is exactly what the auto-wiring dispatch adapter does on the
        consumer side once the contract declares this as its ``event_model`` —
        failed with ``extra_forbidden`` on ``contract_hash`` and
        ``topology_hash``. A model that cannot read its own output is not a wire
        model, and the asymmetry was invisible for as long as nothing on the
        consumer side ever decoded the payload.

        Only the two computed names are dropped, and only after they are checked
        against the values this content actually produces. A mismatch means the
        publisher derived a different hash from the same fields — genuine
        cross-version manifest drift, which is the one thing these hashes exist
        to reveal — so it raises rather than being silently discarded. Every
        other unknown key still fails ``extra_forbidden`` untouched.
        """
        if not isinstance(data, dict):
            return data
        supplied = {name: data[name] for name in _COMPUTED_FIELD_NAMES if name in data}
        if not supplied:
            return data
        remainder = {k: v for k, v in data.items() if k not in supplied}
        recomputed = cls(**remainder)
        for name, value in supplied.items():
            actual = getattr(recomputed, name)
            if value != actual:
                raise ValueError(
                    f"runtime manifest {name} on the wire is {value!r} but this "
                    f"payload's contents derive {actual!r}. The publisher and "
                    "this consumer computed the hash differently, which is "
                    "manifest drift, not a decode problem — do not silence it by "
                    "dropping the field."
                )
        return remainder


__all__: list[str] = ["ModelRuntimeManifestPublished"]
