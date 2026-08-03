# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Single typed source of the config files that MUST exist in the runtime image.

OMN-15676. Three separate incidents have now had the same shape: a config file
is tracked in the repo, every source-level check passes, and the defect exists
only in the *built image* because no ``COPY`` ships the file:

* the grants fixture (OMN-6698 / OMN-12726 class -- deploy scripts omitting
  required build-context paths),
* ``routing_tiers.yaml`` (OMN-15645 -- baked only after the image that pinned
  it had already been built, forcing OMN-15623 to pin a site-packages literal),
* ``runner_fleet.yaml`` (OMN-15676 -- this module's reason for existing; the
  deployed runtime raised ``FileNotFoundError: /app/config/runner_fleet.yaml``
  while the repo looked perfectly correct).

Nothing tested the built artifact, so no source review, static sweep, or unit
suite could see any of the three. This registry is the single typed source that
``scripts/ci/assert_image_config_paths.py`` reads to assert ``test -f`` for
every entry **inside the built image**, wired into the runtime image builds
before the push step so a missing ``COPY`` cannot reach a registry.

Scope -- deliberately narrow, and the exclusions are the load-bearing part:

* IN SCOPE: files under ``/app/config/`` that the runtime resolves at startup.
  ``/app/config/`` carries no volume or bind-mount entry anywhere in the compose
  lane files or the onex-dev manifests, so image-baked content there is never
  shadowed at container start. That property is exactly why the OMN-15645 bake
  chose ``/app/config/`` over ``/app/contracts/``; see the block comment above
  the ``routing_tiers.yaml`` COPY in ``docker/Dockerfile.runtime``.
* OUT OF SCOPE ``/app/data/``: rendered at boot by ``entrypoint-runtime.sh``
  (``render_bifrost_delegation_contract`` / ``render_secret_resolver_config``)
  *after* volumes are mounted, and mount-covered by design in both compose and
  k8s (the OMN-12945 emptyDir shadow lives here). Asserting image-baked content
  under ``/app/data/`` would assert the wrong thing: absence there is correct.
* OUT OF SCOPE ``/app/contracts/``: bind-mounted read-only from the host by the
  compose runtime services, so host content legitimately shadows the baked tree.

Adding an entry here is the whole cost of protecting a new startup-resolved
config path. If a path is resolved from an env-var pin, record the pin in
``resolved_by`` so the binding is greppable from one place.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRequiredImageConfigPath(BaseModel):
    """One config file that must be present in the built runtime image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Absolute in-image path asserted with `test -f` inside the built "
            "image. Must be absolute and must not contain shell metacharacters."
        ),
    )
    resolved_by: str = Field(
        ...,
        min_length=1,
        description=(
            "The startup surface that resolves this path -- a dotted callable, "
            "or an env-var pin name plus where it is bound."
        ),
    )
    why_required: str = Field(
        ...,
        min_length=1,
        description="What breaks at runtime when the file is absent from the image.",
    )
    ticket: str = Field(
        ...,
        min_length=1,
        description="Ticket that added or repaired this requirement.",
    )


REQUIRED_IMAGE_CONFIG_PATHS: tuple[ModelRequiredImageConfigPath, ...] = (
    ModelRequiredImageConfigPath(
        image_path="/app/config/runner_fleet.yaml",
        resolved_by=(
            "omnibase_infra.observability.runner_health.model_runner_fleet_config."
            "default_runner_fleet_config_path (RUNNER_FLEET_CONFIG_PATH override, "
            "else repo-root-relative -- which under the image's PYTHONPATH=/app/src "
            "resolves to /app/config/runner_fleet.yaml)"
        ),
        why_required=(
            "HandlerRunnerFleetSnapshot.__init__ calls load_runner_fleet_config(), "
            "which raises FileNotFoundError rather than falling back to embedded "
            "lab values. The handler is instantiated during auto-wiring, so under "
            "ONEX_WIRING_STRICT_MODE=1 (bound on onex-dev) the absence is a boot "
            "failure, not a degraded mode."
        ),
        ticket="OMN-15676",
    ),
    ModelRequiredImageConfigPath(
        image_path="/app/config/delegation/routing_tiers.yaml",
        resolved_by=(
            "DELEGATION_ROUTING_TIERS_PATH -- pinned in docker-compose.infra.yml "
            "x-runtime-env and in the onex-dev ConfigMap plus the three runtime "
            "Deployments (omninode_infra k8s/onex-dev/runtime/)"
        ),
        why_required=(
            "The delegation routing reducer fails closed when the tiers file the "
            "pin names does not exist. Baked from the installed omnimarket package "
            "to a stable path so the pin never embeds a python3.X site-packages "
            "directory that a base-image Python bump silently invalidates."
        ),
        ticket="OMN-15645",
    ),
)


def required_image_config_paths() -> tuple[str, ...]:
    """Return just the in-image paths, in declaration order."""
    return tuple(entry.image_path for entry in REQUIRED_IMAGE_CONFIG_PATHS)


__all__ = [
    "REQUIRED_IMAGE_CONFIG_PATHS",
    "ModelRequiredImageConfigPath",
    "required_image_config_paths",
]
