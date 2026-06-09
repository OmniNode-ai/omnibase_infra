#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Render contract-owned runtime policy as environment variables."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
    RuntimeProfileName,
)
from omnibase_infra.runtime.models.model_runtime_process_policy import (
    RuntimeProcessName,
)
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)

_DEFAULT_CONTRACT = (
    _REPO_ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
)
_SHELL_SAFE_DOTENV_VALUE = re.compile(r"^[A-Za-z0-9_./,:@+-]+$")

_PROFILE_ENV_PREFIX: dict[RuntimeProfileName, str] = {
    "dev": "DEV",
    "stability-test": "STABILITY_TEST",
    "prod": "PROD",
}
_PROCESS_ENV_PREFIX: dict[RuntimeProcessName, str] = {
    "main": "MAIN",
    "effects": "EFFECTS",
    "worker": "WORKER",
}


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _bifrost_text(value: bool) -> str:
    return "1" if value else "0"


def load_contract(path: Path = _DEFAULT_CONTRACT) -> ModelRuntimePolicyContract:
    """Load the runtime policy contract."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelRuntimePolicyContract.model_validate(raw)


def render_env(contract: ModelRuntimePolicyContract) -> dict[str, str]:
    """Render all runtime profile policy values as env vars."""
    env: dict[str, str] = {
        "AUXILIARY_SERVICES_OMNIMEMORY_ENABLED": _bool_text(
            contract.auxiliary_services_omnimemory_enabled
        ),
        "ONEX_ACTIVE_RUNTIME_PACKAGES": ",".join(contract.active_runtime_packages),
        "OMNIMEMORY_MEMGRAPH_PORT": str(contract.omnimemory_memgraph_port),
    }

    dev_main = contract.profiles["dev"].processes["main"]
    env.update(
        {
            "BIFROST_VERIFY_ENDPOINTS": _bifrost_text(
                dev_main.bifrost_verify_endpoints
            ),
            "OMNIMEMORY_ENABLED": _bool_text(dev_main.omnimemory_enabled),
            "OMNIMEMORY_MEMGRAPH_HOST": dev_main.omnimemory_memgraph_host,
        }
    )

    for profile_name, profile in contract.profiles.items():
        profile_prefix = _PROFILE_ENV_PREFIX[profile_name]
        env[f"{profile_prefix}_COMPOSE_PROJECT"] = profile.compose_project
        env[f"{profile_prefix}_RUNTIME_MAIN_PORT"] = str(profile.main_port)
        env[f"{profile_prefix}_RUNTIME_EFFECTS_PORT"] = str(profile.effects_port)
        env[f"{profile_prefix}_TOPIC_PROVISIONER_MAX_PARTITIONS"] = str(
            profile.topic_provisioner_max_partitions
        )
        secret_resolver_config_json = ""
        if profile.secret_resolver_mappings:
            secret_resolver_config_json = json.dumps(
                ModelSecretResolverConfig(
                    mappings=list(profile.secret_resolver_mappings),
                    enable_convention_fallback=False,
                ).model_dump(mode="json", exclude_defaults=True),
                separators=(",", ":"),
                sort_keys=True,
            )

        for process_name, process in profile.processes.items():
            process_prefix = _PROCESS_ENV_PREFIX[process_name]
            prefix = f"{profile_prefix}_RUNTIME_{process_prefix}"
            env[f"{prefix}_CAPABILITIES"] = ",".join(process.capabilities)
            env[f"{prefix}_BIFROST_VERIFY_ENDPOINTS"] = _bifrost_text(
                process.bifrost_verify_endpoints
            )
            env[f"{prefix}_OMNIMEMORY_ENABLED"] = _bool_text(process.omnimemory_enabled)
            env[f"{prefix}_OMNIMEMORY_MEMGRAPH_HOST"] = process.omnimemory_memgraph_host
            env[f"{prefix}_PUBLISH_INTROSPECTION"] = _bool_text(
                process.publish_introspection
            )
            env[f"{prefix}_SECRET_RESOLVER_CONFIG_PATH"] = (
                profile.secret_resolver_config_path
            )
            env[f"{prefix}_SECRET_RESOLVER_CONFIG_JSON"] = secret_resolver_config_json

    return dict(sorted(env.items()))


def format_dotenv(env: dict[str, str]) -> str:
    """Format env vars as a dotenv file that is safe to shell-source."""
    lines = [
        "# Generated from contracts/services/runtime_policy.contract.yaml.",
        "# Do not edit policy values here; edit the contract and re-render.",
    ]
    lines.extend(
        f"{key}={value if not value or _SHELL_SAFE_DOTENV_VALUE.fullmatch(value) else shlex.quote(value)}"
        for key, value in env.items()
    )
    return "\n".join(lines) + "\n"


def format_shell(env: dict[str, str]) -> str:
    """Format env vars as shell exports."""
    return (
        "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=_DEFAULT_CONTRACT,
        help="Path to contracts/services/runtime_policy.contract.yaml.",
    )
    parser.add_argument(
        "--format",
        choices=("dotenv", "shell"),
        default="dotenv",
        help="Output format.",
    )
    parser.add_argument(
        "--check-env-file",
        type=Path,
        default=None,
        help="Compare rendered dotenv output with this env file.",
    )
    args = parser.parse_args(argv)

    rendered = (
        format_shell(render_env(load_contract(args.contract)))
        if args.format == "shell"
        else format_dotenv(render_env(load_contract(args.contract)))
    )

    if args.check_env_file is not None:
        expected = args.check_env_file.read_text(encoding="utf-8")
        if rendered != expected:
            print(
                f"{args.check_env_file} is not in sync with {args.contract}",
                file=sys.stderr,
            )
            return 1
        print(f"{args.check_env_file} matches {args.contract}")
        return 0

    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
