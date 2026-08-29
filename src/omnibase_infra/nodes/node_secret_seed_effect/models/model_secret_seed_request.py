# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Request model for one headless secret-seeding run (OMN-16897).

The load-bearing property of this model is what it *cannot* carry.

A seeding request names a local **source file**, a **target address**, and a
**mode**. It never carries a secret value. That is not a stylistic
preference: this model is the input to a node, node inputs are serialised
onto the event bus and into the event log, and a value placed here would be
durably persisted in both. So the value-shaped field simply does not exist,
``extra="forbid"`` refuses one that is invented at the call site, and
:meth:`ModelSecretSeedRequest._reject_inline_values` refuses the specific
names a caller is most likely to reach for — with an error that says why,
rather than pydantic's generic "extra inputs are not permitted".

The handler reads values from the named file at execution time, inside the
effect surface, and hands them straight to the store. They exist in memory
for the duration of one ``handle()`` call and are in no event, no receipt,
no log line, and no field of the result model.
"""

from __future__ import annotations

from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Field names a caller might plausibly reach for when trying to pass a value
# inline. ``extra="forbid"`` already rejects every one of them; this list
# exists so the rejection explains the design instead of reading as a typo.
_VALUE_CARRYING_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "value",
        "values",
        "secret",
        "secrets",
        "secret_value",
        "secret_values",
        "key_value",
        "key_values",
        "api_key",
        "payload",
        "data",
        "env",
        "entries",
    }
)

#: Sentinel meaning "read the source from stdin" — the shape that keeps a
#: secret off disk entirely (``... | onex skill seed_secrets --source-path -``).
STDIN_SENTINEL = "-"


class ModelSecretSeedRequest(BaseModel):
    """Addressing and mode for one seed run. Carries no secret values.

    Every target field is REQUIRED with no default. There are three live,
    physically separate Infisical instances in this estate and a wrong
    default on any of ``infisical_host``, ``project_id``,
    ``environment_slug`` or ``secret_path`` seeds a real key into the wrong
    place — a failure that looks exactly like success. CLAUDE.md Rule 8:
    fail fast on missing config, never silently pick one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Seed run correlation ID.")
    source_path: str = Field(
        ...,
        description=(
            "Path to a local dotenv-style source file (``NAME=VALUE`` per "
            f"line), or {STDIN_SENTINEL!r} to read it from stdin. This field "
            "names WHERE the values are; it never carries them. The file is "
            "read once, at execution time, inside the handler."
        ),
    )
    infisical_host: str = Field(
        ...,
        description=(
            "Absolute base URL of the target Infisical instance. REQUIRED "
            "with no default: this estate runs three separate instances (dev "
            "lane, stability lane, in-cluster) and a guessed host would write "
            "a real key into an instance nobody meant to touch. The concrete "
            "hosts and ports are in docs/runbooks/headless-secret-seeding.md "
            "-- deliberately not inlined here, because an address literal in "
            "source is the thing that turns into a default."
        ),
    )
    project_id: UUID = Field(
        ...,
        description="Target Infisical project UUID (not the project name).",
    )
    environment_slug: str = Field(
        ...,
        min_length=1,
        description="Target environment slug, e.g. 'dev' / 'prod'.",
    )
    secret_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Target secret folder, e.g. '/shared/llm/'. Required, not "
            "defaulted to '/': the folder IS the namespace, and seeding a "
            "platform key into the wrong one is silent."
        ),
    )
    keys: list[str] = Field(
        default_factory=list,
        description=(
            "Optional allowlist of source NAMES to seed. Empty means every "
            "name in the source. A name listed here but absent from the "
            "source is a failing NO_KEYS run, never a silent skip."
        ),
    )
    execute: bool = Field(
        default=False,
        description=(
            "Opt IN to writing. Defaults to False, so a request that omits "
            "it plans and writes nothing — the safe mode is the one you get "
            "by forgetting the flag. Expressed as a positive opt-in rather "
            "than a 'dry_run' input because ``onex skill`` boolean args are "
            "PRESENCE flags: a --dry-run flag defaulting to true could never "
            "be turned off from the CLI, which is a worse trap than no flag "
            "at all. Read :attr:`dry_run` for the derived mode."
        ),
    )
    verify_readback: bool = Field(
        default=True,
        description=(
            "After writing, confirm each seeded NAME appears in the store's "
            "name listing. Name-only — this node never reads a secret value, "
            "so it can never compare one."
        ),
    )

    @property
    def dry_run(self) -> bool:
        """True when this run must issue zero writes.

        Derived, not an input field, so "plan only" is the state you reach
        by doing nothing and writing requires an explicit ``execute=True``.
        """
        return not self.execute

    @model_validator(mode="before")
    @classmethod
    def _reject_inline_values(cls, data: object) -> object:
        """Refuse a value-shaped field with an error that explains itself.

        Typed ``object`` rather than ``Any``: a ``mode="before"`` validator
        receives whatever the caller passed, and ``object`` says exactly that
        while keeping the repo's no-``Any``-in-signatures gate satisfied. The
        ``isinstance`` narrowing below is the only thing that reads it.
        """
        if not isinstance(data, dict):
            return data
        offenders = sorted(
            str(key) for key in data if str(key).lower() in _VALUE_CARRYING_FIELD_NAMES
        )
        if offenders:
            raise ValueError(
                "secret values must never be passed inline on a node request "
                f"(rejected field(s): {offenders}). A node request is "
                "serialised onto the event bus and into the event log, so a "
                "value placed here would be durably persisted. Pass "
                "'source_path' naming a local file (or '-' for stdin) "
                "instead; the handler reads the values at execution time and "
                "they never leave it."
            )
        return data

    @field_validator("source_path")
    @classmethod
    def _validate_source_path(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("source_path must not be empty")
        return stripped

    @field_validator("infisical_host")
    @classmethod
    def _validate_infisical_host(cls, value: str) -> str:
        stripped = value.strip().rstrip("/")
        if not stripped:
            raise ValueError("infisical_host must not be empty")
        parsed = urlsplit(stripped)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(
                f"infisical_host must be an absolute HTTPS URL, got: {value!r}"
            )
        if parsed.username or parsed.password:
            raise ValueError("infisical_host must not include credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("infisical_host must not include query or fragment data")
        if parsed.path not in ("", "/"):
            raise ValueError(
                f"infisical_host must be a base URL without a path, got: {value!r}"
            )
        return stripped

    @field_validator("secret_path")
    @classmethod
    def _validate_secret_path(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped.startswith("/"):
            raise ValueError(
                f"secret_path must be absolute (start with '/'), got: {value!r}"
            )
        return stripped

    @field_validator("keys")
    @classmethod
    def _validate_keys(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("keys must not contain duplicates")
        return cleaned


__all__ = ["STDIN_SENTINEL", "ModelSecretSeedRequest"]
