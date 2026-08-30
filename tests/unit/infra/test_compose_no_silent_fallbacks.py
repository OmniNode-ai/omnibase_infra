# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that no env var uses the empty-string-means-disabled pattern."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

COMPOSE_FILE = (
    Path(__file__).resolve().parents[3] / "docker" / "docker-compose.infra.yml"
)

# ROLE_* passwords in postgres service are intentionally empty-means-skip:
# the init script creates roles only when the corresponding password is set.
# These are NOT feature flags or opt-in toggles.
ALLOWED_EMPTY_DEFAULTS = {
    "ROLE_OMNIBASE_PASSWORD",
    "ROLE_OMNICLAUDE_PASSWORD",
    "ROLE_OMNIDASH_PASSWORD",
    "ROLE_OMNIINTELLIGENCE_PASSWORD",
    "ROLE_OMNIMEMORY_PASSWORD",
    "ROLE_OMNINODE_PASSWORD",
    # OMN-16843: same empty-means-skip contract as the ROLE_* passwords above.
    # This is the postgres FIRST-STARTUP provisioning seam
    # (000_create_multiple_databases.sh LOGIN_ONLY_ROLE_MAP); an unprovisioned
    # volume must skip the role, not wedge the whole lane at compose render.
    # The fail-closed `:?` form lives on the CONSUMER of the credential,
    # OMNINODE_INTERNAL_DB_URL in x-runtime-env.
    "OMNINODE_RUNTIME_PASSWORD",
    # OMN-15425: the TENANT-domain half of the identical split, on the identical
    # contract as OMNINODE_RUNTIME_PASSWORD directly above — same postgres
    # first-startup seam (000_create_multiple_databases.sh LOGIN_ONLY_ROLE_MAP),
    # same empty-means-skip semantics on an unprovisioned volume. The
    # fail-closed `:?` form lives on the CONSUMER, ONEX_TENANT_DB_URL in
    # x-runtime-env, so an unset credential still wedges compose render for any
    # lane that actually runs the tenant projections.
    "TENANT_PROJECTION_WRITER_PASSWORD",
    # OMN-4316: OMNIMEMORY_* vars are intentionally opt-in (empty = disabled)
    "OMNIMEMORY_ENABLED",
    "OMNIMEMORY_MEMGRAPH_HOST",
    "OMNIMEMORY_DB_URL",
    # Qdrant vector store: opt-in (empty = Qdrant not configured)
    "QDRANT_HOST",
    "QDRANT_API_KEY",
    # Infisical: opt-in config store; empty = fall back to env vars
    "INFISICAL_ADDR",
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
    "INFISICAL_PROJECT_ID",
    "INFISICAL_REQUIRED",
    # Local Docker runtime prefetch remains opt-in through prefixed host vars.
    "ONEX_RUNTIME_INFISICAL_ADDR",
    "ONEX_RUNTIME_INFISICAL_CLIENT_ID",
    "ONEX_RUNTIME_INFISICAL_CLIENT_SECRET",
    "ONEX_RUNTIME_INFISICAL_PROJECT_ID",
    "ONEX_RUNTIME_INFISICAL_ENVIRONMENT",
    # Keycloak: opt-in auth; empty = keycloak not configured
    "KEYCLOAK_ADMIN_CLIENT_SECRET",
    # OMN-15529: OnexBot-OCC-Writer App identity. Opt-in per host — the App key
    # is minted onto a specific runtime host (.201) by the operator, so every
    # other lane must render and run without it. Empty is NOT "silently
    # disabled" here, it is the current, explicit PAT behaviour: both consumers
    # read the mode as `os.environ.get(..., "pat").strip().lower() or "pat"`, and
    # `resolve_api_key(..., required=False)` treats an empty credential as absent
    # and raises GitHubAppCredentialMissingError in app mode instead of falling
    # back to the shared PAT. A `:?` form would wedge every unprovisioned lane.
    "ONEXBOT_OCC_APP_ID",
    "ONEXBOT_OCC_PRIVATE_KEY",
    "OMNI_OCC_GITHUB_AUTH_MODE",
    # OMN-16778: Slack alert delivery credentials, on the identical contract to
    # the OnexBot App identity above and for the identical reason. The bot token
    # and channel id live in the operator's host `.env` on the lanes that alert
    # (.201: /data/omninode/omnibase_infra/.env, mode 0600) and are absent
    # everywhere else, so a `:?` form would wedge compose render on every lane
    # that does not alert. Empty here is NOT "silently disabled": the consumer
    # is `node_slack_publish_effect`, whose handler resolves SLACK_BOT_TOKEN
    # through `resolve_api_key_async(required=True)` and RAISES on an empty
    # value, and `node_consumer_flow_stall_alert_effect`, which records a
    # decided-but-unpublished alert by name on its terminal event. Both fail
    # loudly at the effect boundary; neither substitutes a default.
    "SLACK_BOT_TOKEN",
    "SLACK_CHANNEL_ID",
}


def strip_yaml_comments(content: str) -> str:
    """Blank out ``#`` comments, leaving quoted ``#`` characters intact.

    The ban below is about what compose actually EXPANDS, which is only the
    uncommented YAML. Scanning raw text made the check fire on prose: the
    OMN-15645 comment in ``docker-compose.infra.yml`` explains the rule by
    quoting the banned form (``no ${VAR:-} ambient-override footgun``), and the
    scanner read that illustration as a violation on a variable literally named
    ``VAR`` — a false positive that put ``dev`` red with no real defect
    (OMN-15628 remediation). Narrowing the matcher is the fix; the ban itself is
    unchanged and still fails on a genuine ``${VAR:-}`` in live YAML.

    Line structure is preserved so reported positions stay meaningful.
    """
    cleaned: list[str] = []
    for line in content.splitlines():
        quote: str | None = None
        cut: int | None = None
        for index, char in enumerate(line):
            if quote is not None:
                if char == quote:
                    quote = None
            elif char in ("'", '"'):
                quote = char
            elif char == "#" and (index == 0 or line[index - 1].isspace()):
                cut = index
                break
        cleaned.append(line if cut is None else line[:cut])
    return "\n".join(cleaned)


@pytest.mark.unit
def test_no_empty_default_fallbacks_in_runtime_env() -> None:
    """No env var should use the empty-string-means-disabled pattern.

    Pattern banned: ${VAR:-}  (empty default = silently disabled)
    Pattern allowed: ${VAR:?message}  (required, fails loud)
    Pattern allowed: ${VAR:-value}  (operational default with real value)
    Pattern allowed: "literal"  (hardcoded)

    Exception: ROLE_*_PASSWORD in postgres service (empty = skip role creation).

    Comments are stripped first — see :func:`strip_yaml_comments`. Only YAML
    compose actually expands can violate this; prose that quotes the banned form
    to explain it cannot.
    """
    with open(COMPOSE_FILE) as f:
        content = strip_yaml_comments(f.read())

    # Find all ${VAR:-} patterns (empty default)
    empty_defaults = re.findall(r"\$\{([A-Z_]+):-\}", content)

    # Filter out allowed exceptions
    violations = [v for v in empty_defaults if v not in ALLOWED_EMPTY_DEFAULTS]

    assert len(violations) == 0, (
        f"Found {len(violations)} empty-default fallbacks that should be "
        f"converted to required or removed: {violations}"
    )


@pytest.mark.unit
def test_comment_stripping_does_not_defang_the_ban() -> None:
    """Narrowing the matcher to non-comment YAML must not weaken it.

    Three properties, because "ignore comments" is exactly the kind of fix that
    quietly turns a gate into a no-op:

    1. A real ``${VAR:-}`` in live YAML is still found (the ban has teeth).
    2. The same form inside a comment is not (the false positive is gone).
    3. A ``#`` inside a quoted scalar does not truncate the value (the stripper
       does not eat real configuration).
    """
    live_yaml = 'services:\n  app:\n    environment:\n      A: "${REAL_ONE:-}"\n'
    assert re.findall(r"\$\{([A-Z_]+):-\}", strip_yaml_comments(live_yaml)) == [
        "REAL_ONE"
    ]

    commented = "  # no ${VAR:-} ambient-override footgun\n  B: literal\n"
    assert re.findall(r"\$\{([A-Z_]+):-\}", strip_yaml_comments(commented)) == []

    quoted_hash = '      PROMPT: "value#notacomment"\n'
    assert "value#notacomment" in strip_yaml_comments(quoted_hash)
