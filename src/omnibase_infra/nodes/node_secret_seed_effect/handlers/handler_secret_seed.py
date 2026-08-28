# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handler for headless Infisical secret seeding (OMN-16897).

What this is
------------
One canonical def-B handler — ``handle(request) -> result`` — that seeds
secrets into a named Infisical instance with no UI and no interactive
login. It exists because the underlying capability was already complete and
simply unreachable as a node: ``InfisicalSecretStore.set_secret`` has been
a working async upsert since OMN-10557, and ``AdapterInfisical`` has carried
``create_secret``/``update_secret`` since OMN-2286. Nothing here
reimplements either. This handler is the contract-shaped front door to the
write path that already existed.

The one rule everything else follows from
-----------------------------------------
**A secret value must never leave this function.**

Not onto the bus, not into the event log, not into the receipt, not into a
log line, not into an exception message. Concretely:

* The request model has no value field and refuses one
  (``model_secret_seed_request``). Values are named by FILE PATH.
* Values are read here, at execution time, wrapped in ``SecretStr`` the
  instant they are parsed, and handed to ``set_secret`` unwrapped exactly
  once — the same "value crosses exactly once" discipline
  ``omnimarket/projection/credential_publisher.py`` uses for customer BYOK.
* Every message derived from an exception goes through
  :func:`_safe_error`, which runs ``sanitize_error_message`` AND then
  explicitly replaces any parsed value substring with ``***``. The
  sanitiser alone is not enough: it pattern-matches credential-shaped text,
  and an API key echoed back inside a third-party SDK error is not
  guaranteed to match a pattern. Redacting the values we are literally
  holding is the only version of this that is sound.
* Verification is a NAME listing. This node never calls ``get_secret`` and
  therefore cannot read an existing value even by accident — which is also
  why a source parse error reports a line NUMBER and never line content.

Fail-fast, not fallback
-----------------------
Missing machine-identity material is ``AUTH_UNAVAILABLE`` with the missing
VARIABLE NAMES, never a fallback to some ambient identity (CLAUDE.md Rule
8). Writing is a positive opt-in (``execute=True``) and ``dry_run`` is
derived from its absence, so the mode you get by forgetting the flag is the
one that writes nothing.

Related:
    - OMN-16897: this ticket
    - OMN-10557: ``InfisicalSecretStore`` — the write path reused here
    - OMN-2286: ``AdapterInfisical`` lives in ``_internal/``; bootstrap and
      admin write paths are its sanctioned importers, which is what this is
    - OMN-16451 / ``docs/plans/2026-08-24-config-control-plane-plan.md``:
      the read side keeps secrets as refs and resolves values at read time.
      This node is that discipline's write-side complement — a config value
      still holds a ref, and the value it points at gets here by file, not
      by payload.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol

from pydantic import SecretStr

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_secret_seed_effect.models.enum_secret_seed_verdict import (
    EnumSecretSeedVerdict,
)
from omnibase_infra.nodes.node_secret_seed_effect.models.model_secret_seed_request import (
    STDIN_SENTINEL,
    ModelSecretSeedRequest,
)
from omnibase_infra.nodes.node_secret_seed_effect.models.model_secret_seed_result import (
    ModelSecretSeedResult,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

__all__ = [
    "REQUIRED_AUTH_ENV_VARS",
    "HandlerSecretSeed",
    "parse_source",
    "render_receipt",
]

#: Machine-identity (Universal Auth) material. Read from the environment
#: because this is a BOOTSTRAP admin path — the same sanctioned exception
#: ``AdapterInfisical``'s own module docstring carves out, and the same
#: variables ``omnimarket.projection.credential_publisher`` reads. Config
#: doctrine puts env at the bootstrap edge only; a tool whose job is to
#: populate the secret store cannot resolve its own credentials from the
#: secret store it is populating. ``scripts/check-env-reads.sh`` names this
#: exact pair in its own BOOTSTRAP_ALLOWLIST for that reason ("a keyring
#: cannot unlock itself").
#:
#: OMN-14951 gap 2: self-declared secret-ish env-var names read by this
#: boundary file (see ``check_secret_name_declarations``). Named
#: ``required_secrets`` so the scanner's DECLARATION_MARKER matches.
required_secrets: tuple[str, ...] = (
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
)
REQUIRED_AUTH_ENV_VARS: tuple[str, ...] = required_secrets

_REDACTED = "***"

# Below this length a "value" is too short to redact usefully and blanking
# every occurrence would mangle unrelated text (a 1-char value would redact
# every instance of that character in a message). A secret this short is not
# a secret; the run still writes it, we just do not attempt substring
# redaction on it.
_MIN_REDACTABLE_VALUE_LENGTH = 4


class ProtocolSeedSecretStore(Protocol):
    """The slice of ``ProtocolSecretStore`` this handler is allowed to use.

    Deliberately narrower than the full SPI protocol: ``get_secret`` is
    ABSENT. A handler that cannot call it cannot read an existing secret
    value even by mistake, which makes "this node never reads a value" a
    property of the type rather than a promise in a docstring.
    ``InfisicalSecretStore`` satisfies this structurally.
    """

    async def set_secret(self, key: str, value: str) -> bool: ...

    async def list_keys(self, prefix: str | None = None) -> list[str]: ...

    async def close(self, timeout_seconds: float = 30.0) -> None: ...


TypeStoreFactory = Callable[[ModelSecretSeedRequest], ProtocolSeedSecretStore]
TypeSourceReader = Callable[[str], str]
TypeAuthProbe = Callable[[], list[str]]


def _default_auth_probe() -> list[str]:
    """Return the names of the missing auth variables (names only, ever).

    Written as two LITERAL reads rather than a loop over
    :data:`required_secrets`. A loop reads through a variable name, which
    ``check-env-reads.sh``'s name extractor deliberately cannot resolve —
    it fails such a read closed rather than silently grandfathering it. The
    literal form lets the gate see exactly which two variables this
    boundary consumes, which is the entire point of that check.
    """
    # ONEX_EXCLUDE: bootstrap credential resolution for the tool that seeds
    # the secret store — it cannot read its own identity out of the store it
    # is about to write to. Both names are in check-env-reads.sh's own
    # BOOTSTRAP_ALLOWLIST for this reason.
    client_id = os.environ.get("INFISICAL_CLIENT_ID", "")  # ONEX_EXCLUDE
    client_secret = os.environ.get("INFISICAL_CLIENT_SECRET", "")  # ONEX_EXCLUDE
    missing: list[str] = []
    if not client_id.strip():
        missing.append("INFISICAL_CLIENT_ID")
    if not client_secret.strip():
        missing.append("INFISICAL_CLIENT_SECRET")
    return missing


def _default_source_reader(source_path: str) -> str:
    """Read the raw source text from a file, or from stdin on ``-``."""
    if source_path == STDIN_SENTINEL:
        return sys.stdin.read()
    return Path(source_path).expanduser().read_text(encoding="utf-8")


def _default_store_factory(request: ModelSecretSeedRequest) -> ProtocolSeedSecretStore:
    """Build the real Infisical-backed store for this request's target.

    Imported inside the function so the SDK-bearing adapter is not a hard
    import cost for a dry run or a unit test that injects its own store.
    """
    from omnibase_infra.adapters._internal.adapter_infisical import AdapterInfisical
    from omnibase_infra.adapters.models.model_infisical_config import (
        ModelInfisicalAdapterConfig,
    )
    from omnibase_infra.secret_stores.infisical_secret_store import (
        InfisicalSecretStore,
    )

    config = ModelInfisicalAdapterConfig(
        host=request.infisical_host,
        # ONEX_EXCLUDE: see REQUIRED_AUTH_ENV_VARS — bootstrap-edge identity.
        client_id=SecretStr(os.environ["INFISICAL_CLIENT_ID"]),  # ONEX_EXCLUDE
        client_secret=SecretStr(os.environ["INFISICAL_CLIENT_SECRET"]),  # ONEX_EXCLUDE
        project_id=request.project_id,
        environment_slug=request.environment_slug,
        secret_path=request.secret_path,
    )
    adapter = AdapterInfisical(config)
    adapter.initialize()
    return InfisicalSecretStore(
        adapter,
        project_id=str(request.project_id),
        environment_slug=request.environment_slug,
        secret_path=request.secret_path,
    )


def _redact(text: str, values: Iterable[SecretStr]) -> str:
    """Blank every held secret value out of ``text``.

    This runs on top of ``sanitize_error_message``, not instead of it. The
    sanitiser recognises credential-SHAPED text; it cannot recognise an
    arbitrary API key that a third-party SDK helpfully echoed back inside
    its own error string. We are holding those values, so we can remove them
    by identity, which is the only reliable version of this.
    """
    redacted = text
    for secret in values:
        raw = secret.get_secret_value()
        if len(raw) >= _MIN_REDACTABLE_VALUE_LENGTH and raw in redacted:
            redacted = redacted.replace(raw, _REDACTED)
    return redacted


def _safe_error(exc: Exception, values: Iterable[SecretStr]) -> str:
    """Sanitise an exception, then redact held values out of the remainder."""
    return _redact(sanitize_error_message(exc), values)


def parse_source(raw: str) -> dict[str, SecretStr]:
    """Parse dotenv-style ``NAME=VALUE`` text into name -> ``SecretStr``.

    Values are wrapped in ``SecretStr`` at the moment of parsing, so any
    accidental ``repr()``/``str()``/f-string of the parsed mapping renders
    ``SecretStr('**********')`` rather than the key material.

    Raises:
        ValueError: on a malformed line. The message carries the line
            NUMBER and nothing else — a malformed line in a secrets file is
            itself likely to contain a secret, so echoing it back would
            defeat the entire point of this module.
    """
    parsed: dict[str, SecretStr] = {}
    for lineno, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        if "=" not in stripped:
            raise ValueError(
                f"line {lineno}: expected NAME=VALUE (line content withheld — "
                "it may itself be a secret)"
            )
        name, _, value = stripped.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"line {lineno}: empty secret name")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if name in parsed:
            raise ValueError(f"line {lineno}: duplicate secret name {name!r}")
        parsed[name] = SecretStr(value)
    return parsed


class HandlerSecretSeed:
    """Seed secrets into a named Infisical instance, headlessly.

    Every collaborator is injectable so the unit suite drives the real
    handler with no network and no live secret store.
    """

    def __init__(
        self,
        store_factory: TypeStoreFactory | None = None,
        source_reader: TypeSourceReader | None = None,
        auth_probe: TypeAuthProbe | None = None,
    ) -> None:
        self._store_factory: TypeStoreFactory = store_factory or _default_store_factory
        self._source_reader: TypeSourceReader = source_reader or _default_source_reader
        self._auth_probe: TypeAuthProbe = auth_probe or _default_auth_probe

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: ModelSecretSeedRequest) -> ModelSecretSeedResult:
        """Run one seed. Values live only inside this call."""
        # 1. Auth first, before the source is even read. If we cannot write,
        #    there is no reason to hold key material in memory at all.
        missing = self._auth_probe()
        if missing:
            return self._failure(
                request,
                EnumSecretSeedVerdict.AUTH_UNAVAILABLE,
                (
                    "machine-identity auth material is not set: "
                    f"{sorted(missing)}. Mint a machine identity in the "
                    "target Infisical instance and export its Universal Auth "
                    "credentials before seeding. No fallback identity is "
                    "attempted and nothing was written."
                ),
            )

        # 2. Read + parse. From here on, `values` is held and every outbound
        #    string is redacted against it.
        try:
            raw = self._source_reader(request.source_path)
        except OSError as exc:
            return self._failure(
                request,
                EnumSecretSeedVerdict.SOURCE_UNREADABLE,
                f"could not read source {request.source_path!r}: "
                f"{sanitize_error_message(exc)}",
            )
        try:
            parsed = parse_source(raw)
        except ValueError as exc:
            # parse_source is written to never put line CONTENT in its
            # message, so this is safe to surface directly.
            return self._failure(
                request,
                EnumSecretSeedVerdict.SOURCE_UNREADABLE,
                f"could not parse source {request.source_path!r}: {exc}",
            )
        finally:
            del raw

        selected, missing_from_source = self._select(parsed, request.keys)
        values = list(selected.values())

        if not selected:
            return self._failure(
                request,
                EnumSecretSeedVerdict.NO_KEYS,
                (
                    "nothing to seed: the source yielded no matching names"
                    + (
                        f" (requested but absent from source: {missing_from_source})"
                        if missing_from_source
                        else ""
                    )
                    + ". A seed run that seeds nothing is reported as failing, "
                    "not as green."
                ),
                missing_from_source=missing_from_source,
            )

        store = None
        try:
            try:
                store = self._store_factory(request)
            except Exception as exc:  # noqa: BLE001 - construction failure is a verdict
                return self._failure(
                    request,
                    EnumSecretSeedVerdict.AUTH_UNAVAILABLE,
                    "could not authenticate to the target Infisical instance "
                    f"{request.infisical_host}: {_safe_error(exc, values)}",
                    missing_from_source=missing_from_source,
                )

            # 3. Name listing — the ONLY read this node performs. It tells us
            #    create-vs-update without ever touching a stored value.
            try:
                existing = set(await store.list_keys())
            except Exception as exc:  # noqa: BLE001 - a verdict, not a crash
                return self._failure(
                    request,
                    EnumSecretSeedVerdict.STORE_UNREACHABLE,
                    "could not list existing secret names at "
                    f"{request.secret_path!r}: {_safe_error(exc, values)}",
                    missing_from_source=missing_from_source,
                )

            names = sorted(selected)
            would_create = [n for n in names if n not in existing]
            would_update = [n for n in names if n in existing]

            if request.dry_run:
                return ModelSecretSeedResult(
                    **self._base(request),
                    verdict=EnumSecretSeedVerdict.DRY_RUN,
                    success=True,
                    detail=(
                        f"dry run: {len(would_create)} name(s) would be created, "
                        f"{len(would_update)} would be updated at "
                        f"{request.secret_path} on {request.infisical_host}. "
                        "Zero writes were issued. Re-run with dry_run=false to "
                        "apply."
                    ),
                    created_names=would_create,
                    updated_names=would_update,
                    missing_from_source_names=missing_from_source,
                )

            # 4. The write. Idempotent upsert via the existing store path.
            created, updated, failed, errors = await self._write(
                store, selected, existing, values
            )

            written = created + updated
            verified, unverified = await self._verify(store, written, request, values)

            verdict, detail = self._decide(
                request=request,
                created=created,
                updated=updated,
                failed=failed,
                unverified=unverified,
            )
            return ModelSecretSeedResult(
                **self._base(request),
                verdict=verdict,
                success=verdict is EnumSecretSeedVerdict.SEEDED,
                detail=detail,
                created_names=created,
                updated_names=updated,
                failed_names=failed,
                verified_names=verified,
                unverified_names=unverified,
                missing_from_source_names=missing_from_source,
                errors=errors,
            )
        finally:
            if store is not None:
                try:
                    await store.close()
                except Exception as exc:  # noqa: BLE001 - teardown noise
                    logger.warning(
                        "secret store close failed: %s", _safe_error(exc, values)
                    )
            # Drop the last references we hold to the parsed material.
            selected.clear()
            parsed.clear()
            values.clear()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _select(
        parsed: Mapping[str, SecretStr], keys: list[str]
    ) -> tuple[dict[str, SecretStr], list[str]]:
        """Apply the optional allowlist; report requested-but-absent names."""
        if not keys:
            return dict(parsed), []
        selected = {name: parsed[name] for name in keys if name in parsed}
        missing = sorted(name for name in keys if name not in parsed)
        return selected, missing

    async def _write(
        self,
        store: ProtocolSeedSecretStore,
        selected: Mapping[str, SecretStr],
        existing: set[str],
        values: list[SecretStr],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Upsert every selected name, collecting a complete report.

        Deliberately does NOT stop at the first failure. A seed that aborts
        halfway leaves an operator guessing which names landed; a complete
        report of what succeeded and what did not is the thing they can act
        on.
        """
        created: list[str] = []
        updated: list[str] = []
        failed: list[str] = []
        errors: list[str] = []
        for name in sorted(selected):
            try:
                ok = await store.set_secret(name, selected[name].get_secret_value())
            except Exception as exc:  # noqa: BLE001 - per-name failure is data
                failed.append(name)
                errors.append(f"{name}: {_safe_error(exc, values)}")
                continue
            if not ok:
                failed.append(name)
                errors.append(f"{name}: store reported the write was not applied")
            elif name in existing:
                updated.append(name)
            else:
                created.append(name)
        return created, updated, failed, errors

    async def _verify(
        self,
        store: ProtocolSeedSecretStore,
        written: list[str],
        request: ModelSecretSeedRequest,
        values: list[SecretStr],
    ) -> tuple[list[str], list[str]]:
        """Confirm each written NAME is present. Never compares a value."""
        if not request.verify_readback or not written:
            return [], []
        try:
            after = set(await store.list_keys())
        except Exception as exc:  # noqa: BLE001 - unverifiable is not verified
            logger.warning(
                "post-write name readback failed: %s", _safe_error(exc, values)
            )
            return [], sorted(written)
        verified = sorted(n for n in written if n in after)
        unverified = sorted(n for n in written if n not in after)
        return verified, unverified

    @staticmethod
    def _decide(
        *,
        request: ModelSecretSeedRequest,
        created: list[str],
        updated: list[str],
        failed: list[str],
        unverified: list[str],
    ) -> tuple[EnumSecretSeedVerdict, str]:
        """Rank the outcomes. Most specific failure wins."""
        if failed:
            return (
                EnumSecretSeedVerdict.WRITE_FAILED,
                (
                    f"{len(failed)} of {len(failed) + len(created) + len(updated)} "
                    f"name(s) failed to write at {request.secret_path} on "
                    f"{request.infisical_host}. See errors for per-name detail; "
                    "successful writes are listed and were not rolled back."
                ),
            )
        if unverified:
            return (
                EnumSecretSeedVerdict.VERIFY_FAILED,
                (
                    f"{len(unverified)} name(s) were written but did not appear "
                    "in the post-write name listing. Failing closed — an "
                    "unconfirmed write is not a confirmed one."
                ),
            )
        return (
            EnumSecretSeedVerdict.SEEDED,
            (
                f"seeded {len(created)} new and {len(updated)} updated name(s) "
                f"at {request.secret_path} on {request.infisical_host}, "
                "each confirmed present by name readback."
            ),
        )

    @staticmethod
    def _base(request: ModelSecretSeedRequest) -> dict[str, object]:
        return {
            "correlation_id": request.correlation_id,
            "dry_run": request.dry_run,
            "infisical_host": request.infisical_host,
            "project_id": request.project_id,
            "environment_slug": request.environment_slug,
            "secret_path": request.secret_path,
            "source_path": request.source_path,
        }

    def _failure(
        self,
        request: ModelSecretSeedRequest,
        verdict: EnumSecretSeedVerdict,
        detail: str,
        missing_from_source: list[str] | None = None,
    ) -> ModelSecretSeedResult:
        """Build a failing receipt.

        ``detail`` is value-free by construction at every call site: it is
        either a fixed string, or the output of :func:`_safe_error`, which
        has already sanitised and redacted it. This is asserted by
        ``test_no_secret_value_reaches_any_failure_receipt``.
        """
        return ModelSecretSeedResult(
            **self._base(request),
            verdict=verdict,
            success=False,
            detail=detail,
            missing_from_source_names=missing_from_source or [],
        )


def render_receipt(result: ModelSecretSeedResult) -> str:
    """Render the run receipt as pretty JSON for a job summary or a ticket.

    Safe by construction: ``ModelSecretSeedResult`` has no value-carrying
    field, and the contract-conformance test asserts it stays that way.
    """
    return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
