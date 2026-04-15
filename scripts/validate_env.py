#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Environment variable validation for the OmniNode platform.

Checks for missing critical vars, warns on known collisions and deprecations,
and reports undocumented vars found in the environment.

Usage:
    python scripts/validate_env.py                  # validate current env
    python scripts/validate_env.py --env-file .env  # validate a .env file
    python scripts/validate_env.py --strict         # exit 1 on any warning
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry of known variables
# ---------------------------------------------------------------------------

CRITICAL_VARS: dict[str, str] = {
    # Any of these missing when their feature is enabled should be a hard error
    "KAFKA_BOOTSTRAP_SERVERS": "Kafka broker addresses (required when USE_EVENT_ROUTING=true or KAFKA_ENABLE_INTELLIGENCE=true)",
    "POSTGRES_HOST": "PostgreSQL hostname (required when ENABLE_POSTGRES=true and no *_DB_URL set)",
    "POSTGRES_PORT": "PostgreSQL port (required when ENABLE_POSTGRES=true and no *_DB_URL set)",
    "POSTGRES_USER": "PostgreSQL username (required when ENABLE_POSTGRES=true and no *_DB_URL set)",
    "POSTGRES_PASSWORD": "PostgreSQL password (required when ENABLE_POSTGRES=true and no *_DB_URL set)",
}

CONDITIONAL_RULES: list[dict[str, str | list[str]]] = [
    {
        "gate": "ENABLE_POSTGRES",
        "gate_value": "true",
        "unless_any": [
            "OMNICLAUDE_DB_URL",
            "OMNIDASH_ANALYTICS_DB_URL",
            "OMNIINTELLIGENCE_DB_URL",
            "OMNIMEMORY_DB_URL",
            "OMNIBASE_INFRA_DB_URL",
        ],
        "requires": [
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        ],
        "message": "ENABLE_POSTGRES=true but no full DB URL or individual POSTGRES_* fields set",
    },
    {
        "gate": "USE_EVENT_ROUTING",
        "gate_value": "true",
        "unless_any": [],
        "requires": ["KAFKA_BOOTSTRAP_SERVERS"],
        "message": "USE_EVENT_ROUTING=true but KAFKA_BOOTSTRAP_SERVERS not set",
    },
    {
        "gate": "KAFKA_ENABLE_INTELLIGENCE",
        "gate_value": "true",
        "unless_any": [],
        "requires": ["KAFKA_BOOTSTRAP_SERVERS"],
        "message": "KAFKA_ENABLE_INTELLIGENCE=true but KAFKA_BOOTSTRAP_SERVERS not set",
    },
    {
        "gate": "ENABLE_QDRANT",
        "gate_value": "true",
        "unless_any": ["QDRANT_URL"],
        "requires": ["QDRANT_HOST"],
        "message": "ENABLE_QDRANT=true but neither QDRANT_URL nor QDRANT_HOST set",
    },
]

COLLISIONS: list[dict[str, str]] = [
    {
        "var": "POSTGRES_DATABASE",
        "description": (
            "POSTGRES_DATABASE collides across services: omnidash expects 'omnidash_analytics', "
            "omniclaude expects its own DB name. Use service-specific *_DB_URL variables instead."
        ),
    },
]

DEPRECATED_VARS: dict[str, str] = {
    "ARCHON_INTELLIGENCE_URL": "Use INTELLIGENCE_SERVICE_URL instead",
    "DATABASE_URL": "Use OMNIDASH_ANALYTICS_DB_URL instead",
    "DUAL_PUBLISH_LEGACY_TOPICS": "Legacy topic dual-publish (OMN-2368) — remove after migration",
}

# All known variable names (union of inventory). Used to detect undocumented vars.
KNOWN_VARS: set[str] = {
    # Kafka
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_BROKER_ALLOWLIST",
    "KAFKA_ENVIRONMENT",
    "KAFKA_GROUP_ID",
    "KAFKA_CONSUMER_GROUP_ID",
    "KAFKA_CLIENT_ID",
    "KAFKA_HEALTH_CHECK_CLIENT_ID",
    "KAFKA_REQUEST_TIMEOUT_MS",
    "KAFKA_CONNECTION_TIMEOUT_MS",
    "KAFKA_MAX_RETRIES",
    "KAFKA_RETRY_BASE_DELAY_MS",
    "KAFKA_RETRY_MAX_DELAY_MS",
    "KAFKA_ENABLE_INTELLIGENCE",
    "ENABLE_REAL_TIME_EVENTS",
    "ENABLE_KAFKA_LOGGING",
    "KAFKA_PATTERN_DISCOVERY_TIMEOUT_MS",
    "KAFKA_CODE_ANALYSIS_TIMEOUT_MS",
    "KAFKA_QUALITY_ASSESSMENT_TIMEOUT_MS",
    "DUAL_PUBLISH_LEGACY_TOPICS",
    "OMNIINTELLIGENCE_ALLOW_DEFAULT_KAFKA",
    # Database
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DATABASE",
    "OMNICLAUDE_POSTGRES_DATABASE",
    "OMNICLAUDE_DB_URL",
    "OMNIDASH_ANALYTICS_DB_URL",
    "OMNIINTELLIGENCE_DB_URL",
    "OMNIMEMORY_DB_URL",
    "OMNIBASE_INFRA_DB_URL",
    "OMNIWEB_DB_URL",
    "DATABASE_URL",
    "ENABLE_POSTGRES",
    "POSTGRES_MIN_POOL_SIZE",
    "POSTGRES_MAX_POOL_SIZE",
    "POSTGRES_COMMAND_TIMEOUT",
    "DB_POOL_MAX_CONNECTIONS",
    "DB_POOL_MIN_CONNECTIONS",
    "DB_POOL_IDLE_TIMEOUT_MS",
    "TEST_DATABASE_URL",
    # Qdrant
    "QDRANT_URL",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_HTTP_PORT",
    "QDRANT_GRPC_PORT",
    "QDRANT_VERSION",
    "QDRANT_API_KEY",
    "QDRANT_HTTPS",
    "QDRANT_CPU_LIMIT",
    "QDRANT_MEMORY_LIMIT",
    "ENABLE_QDRANT",
    # Valkey
    "VALKEY_URL",
    "VALKEY_PASSWORD",
    "VALKEY_PORT",
    "VALKEY_HOST",
    "VALKEY_VERSION",
    "ENABLE_INTELLIGENCE_CACHE",
    "CACHE_TTL_PATTERNS",
    "CACHE_TTL_INFRASTRUCTURE",
    "CACHE_TTL_SCHEMAS",
    # LLM
    "LLM_CODER_URL",
    "LLM_EMBEDDING_URL",
    "LLM_EMBEDDING_URL_ALT",
    "LLM_FUNCTION_URL",
    "LLM_QWEN_72B_URL",
    "LLM_QWEN_72B_URL_ALT",
    "LLM_QWEN_14B_URL",
    "LLM_VISION_URL",
    "LLM_DEEPSEEK_R1_URL",
    "LLM_GLM_URL",
    "LLM_GLM_API_KEY",
    "LLM_GLM_MODEL_NAME",
    # Auth
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "Z_AI_API_KEY",
    "Z_AI_API_URL",
    "GITHUB_TOKEN",
    "GH_PAT",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "LINEAR_API_KEY",
    "LINEAR_INSIGHTS_OUTPUT_DIR",
    "KEYCLOAK_ISSUER",
    "KEYCLOAK_CLIENT_ID",
    "KEYCLOAK_CLIENT_SECRET",
    "ONEX_SERVICE_CLIENT_ID",
    "ONEX_SERVICE_CLIENT_SECRET",
    # Infisical
    "INFISICAL_ENCRYPTION_KEY",
    "INFISICAL_AUTH_SECRET",
    "INFISICAL_ADDR",
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
    "INFISICAL_PROJECT_ID",
    # Services
    "INTELLIGENCE_SERVICE_URL",
    "OMNICLAUDE_CONTEXT_API_URL",
    "OMNICLAUDE_CONTEXT_API_ENABLED",
    "OMNICLAUDE_CONTEXT_API_TIMEOUT_MS",
    "ARCHON_INTELLIGENCE_URL",
    "MAIN_SERVER_URL",
    "SEMANTIC_SEARCH_URL",
    "AGENT_LEARNING_RETRIEVAL_URL",
    "ONEX_API_BASE_URL",
    "HEALTH_CHECK_PORT",
    "INTELLIGENCE_API_EXTERNAL_PORT",
    # Feature flags
    "USE_EVENT_ROUTING",
    "USE_LLM_ROUTING",
    "LLM_ROUTING_TIMEOUT_S",
    "ENABLE_SHADOW_VALIDATION",
    "ENABLE_PATTERN_ENFORCEMENT",
    "ENABLE_PATTERN_QUALITY_FILTER",
    "MIN_PATTERN_QUALITY",
    "ENABLE_DISABLED_PATTERN_FILTER",
    "ENABLE_LOCAL_INFERENCE_PIPELINE",
    "ENABLE_PHASE_1_VALIDATION",
    "ENABLE_PHASE_2_STRUCTURAL",
    "ENABLE_PHASE_3_SEMANTIC",
    "ENABLE_PHASE_4_AI_QUORUM",
    "ENFORCEMENT_MODE",
    "PERFORMANCE_BUDGET_SECONDS",
    "OMNICLAUDE_CONSTRAINT_INJECTION_ENABLED",
    "OMNICLAUDE_CONSTRAINT_MAX_ITEMS",
    "OMNICLAUDE_COHORT_CONTROL_PERCENTAGE",
    "OMNICLAUDE_COHORT_SALT",
    "DEMO_MODE",
    "VITE_USE_MOCK_DATA",
    "ENABLE_EVENT_INTELLIGENCE",
    "ENABLE_EVENT_PRELOAD",
    "OMNIDASH_READ_MODEL_USE_CATALOG",
    "OMNIDASH_AUTH_ENABLED",
    "OMNIDASH_ENABLE_EXTERNAL_GATEWAY",
    "ARCHON_ENABLE_EXTERNAL_GATEWAY",
    "FEATURE_SKIP_PROVISION",
    "FEATURE_WAITLIST_MODE",
    # Runtime
    "NODE_ENV",
    "ONEX_ENVIRONMENT",
    "ONEX_LOG_LEVEL",
    "LOG_LEVEL",
    "ONEX_STATE_DIR",
    "ONEX_TASK_ID",
    "ONEX_RUN_ID",
    "OMNICLAUDE_PATH",
    "CLAUDE_PLUGIN_ROOT",
    "PROJECT_ROOT",
    "PORT",
    "OMNIDASH_PORT",
    "HOST",
    # WebSocket
    "WS_PATH",
    "WS_HEARTBEAT_INTERVAL_MS",
    "WS_MAX_MISSED_PINGS",
    "WS_RECONNECT_DELAY_MS",
    "WS_RECONNECT_MAX_DELAY_MS",
    "WS_RECONNECT_MAX_ATTEMPTS",
    # Performance
    "ENABLE_RESPONSE_CACHE",
    "API_CACHE_TTL_SECONDS",
    "API_RATE_LIMIT_PER_MINUTE",
    "API_RATE_LIMIT_BURST",
    "METRICS_COLLECTION_INTERVAL_MS",
    "METRICS_RETENTION_MS",
    # Events
    "EVENT_DATA_RETENTION_MS",
    "EVENT_PRUNE_INTERVAL_MS",
    "EVENT_MAX_ACTIONS",
    "EVENT_MAX_DECISIONS",
    "EVENT_MAX_TRANSFORMATIONS",
    "EVENT_MAX_PERFORMANCE_METRICS",
    "EVENT_CONSUMER_VERBOSE_LOGGING",
    "PRELOAD_WINDOW_MINUTES",
    # Logging
    "REQUEST_LOG_LEVEL",
    "REQUEST_LOG_MAX_BODY_LENGTH",
    # Client
    "VITE_INTELLIGENCE_SERVICE_URL",
    "VITE_API_BASE_URL",
    "VITE_POSTHOG_API_KEY",
    "VITE_POSTHOG_API_HOST",
    "NEXT_PUBLIC_GA_MEASUREMENT_ID",
    "NEXT_PUBLIC_APP_URL",
    "UMAMI_WEBSITE_ID",
    # Docker
    "MEMGRAPH_VERSION",
    # K8s
    "KUBECONFIG",
    "KUBE_CONTEXT",
    "AWS_PROFILE",
    "AWS_REGION",
    # Test
    "PYTEST_ADDOPTS",
    "TEST_VERBOSE",
    "TEST_TIMEOUT",
    "TEST_SKIP_SLOW",
    "TEST_RETRIES",
    "DEBUG",
    "LOG_HTTP",
    "SAVE_RESPONSES",
    "RESPONSE_DIR",
    # Hook/plugin
    "LOG_FILE",
    "PLUGIN_PYTHON_BIN",
    "OMNICLAUDE_NO_HANDLERS",
    "OMNICLAUDE_EMIT_SOCKET",
    "OMNICLAUDE_EMIT_TIMEOUT",
}

# Standard vars to ignore when reporting undocumented
SYSTEM_VAR_PREFIXES: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "TERM",
    "LANG",
    "LC_",
    "DISPLAY",
    "SSH_",
    "EDITOR",
    "VISUAL",
    "PAGER",
    "LESS",
    "COLORTERM",
    "XDG_",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LOGNAME",
    "HOSTNAME",
    "SHLVL",
    "PWD",
    "OLDPWD",
    "MANPATH",
    "INFOPATH",
    "MAIL",
    "LS_COLORS",
    "CLICOLOR",
    "LSCOLORS",
    "ZDOTDIR",
    "ZSH",
    "NVM_",
    "PYENV_",
    "CONDA_",
    "VIRTUAL_ENV",
    "PIPENV_",
    "POETRY_",
    "CARGO_",
    "RUSTUP_",
    "GOPATH",
    "GOROOT",
    "JAVA_HOME",
    "ANDROID_",
    "DOCKER_",
    "COMPOSE_",
    "npm_",
    "NODE_",
    "DENO_",
    "BUN_",
    "RUBY_",
    "GEM_",
    "BUNDLE_",
    "HOMEBREW_",
    "HISTFILE",
    "HISTSIZE",
    "SAVEHIST",
    "HIST",
    "CURSOR_",
    "VSCODE_",
    "ELECTRON_",
    "CHROME_",
    "WEBKIT_",
    "Apple_",
    "SECURITYSESSION",
    "__CF_",
    "__CFBundle",
    "COMMAND_MODE",
    "MallocNano",
    "ORIGINAL_XDG_",
    "LaunchInstance",
    "ITERM_",
    "FIG_",
    "Q_",
    "STARSHIP_",
    "ATUIN_",
    "_",
    "COMP_",
    "FPATH",
    "FUNCNEST",
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_env_file(path: str) -> dict[str, str]:
    """Parse a .env file into a dict (no shell expansion)."""
    env: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        env[key] = value
    return env


def _is_system_var(name: str) -> bool:
    return any(name.startswith(p) for p in SYSTEM_VAR_PREFIXES)


def _truthy(val: str | None) -> bool:
    return val is not None and val.lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def validate(env: dict[str, str]) -> ValidationResult:
    result = ValidationResult()

    # 1. Conditional requirements
    for rule in CONDITIONAL_RULES:
        gate = str(rule["gate"])
        gate_value = str(rule["gate_value"])
        gate_actual = env.get(gate, "")

        if gate_actual.lower() != gate_value.lower():
            continue

        # Check "unless_any" — if any of these are set, skip
        unless_any = rule.get("unless_any", [])
        assert isinstance(unless_any, list)
        if any(env.get(u) for u in unless_any):
            continue

        requires = rule.get("requires", [])
        assert isinstance(requires, list)
        missing = [r for r in requires if not env.get(r)]
        if missing:
            result.errors.append(f"{rule['message']} — missing: {', '.join(missing)}")

    # 2. Collision warnings
    for collision in COLLISIONS:
        if env.get(collision["var"]):
            result.warnings.append(f"COLLISION: {collision['description']}")

    # 3. Deprecated var warnings
    for var, msg in DEPRECATED_VARS.items():
        if env.get(var):
            result.warnings.append(f"DEPRECATED: {var} is set. {msg}")

    # 4. ONEX_STATE_DIR existence check
    state_dir = env.get("ONEX_STATE_DIR")
    if state_dir and not Path(state_dir).is_dir():
        result.warnings.append(f"ONEX_STATE_DIR={state_dir} does not exist on disk")

    # 5. Undocumented vars
    undocumented = []
    for key in sorted(env):
        if key in KNOWN_VARS:
            continue
        if _is_system_var(key):
            continue
        undocumented.append(key)

    if undocumented:
        result.info.append(
            f"Undocumented env vars ({len(undocumented)}): {', '.join(undocumented)}"
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate OmniNode platform environment variables"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to a .env file to validate (default: current process env)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on any warning (not just errors)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and warnings, suppress info",
    )
    args = parser.parse_args()

    if args.env_file:
        env = _load_env_file(args.env_file)
        print(f"Validating env file: {args.env_file} ({len(env)} vars)")
    else:
        env = dict(os.environ)
        print(f"Validating current environment ({len(env)} vars)")

    result = validate(env)

    # Print results
    for err in result.errors:
        print(f"  ERROR: {err}")
    for warn in result.warnings:
        print(f"  WARN:  {warn}")
    if not args.quiet:
        for info in result.info:
            print(f"  INFO:  {info}")

    # Summary
    print()
    print(
        f"Result: {len(result.errors)} error(s), "
        f"{len(result.warnings)} warning(s), "
        f"{len(result.info)} info"
    )

    if result.errors:
        return 1
    if args.strict and result.warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
