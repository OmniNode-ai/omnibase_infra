# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No workflow in this repository references a provider key secret (OMN-20974).

The team is its own customer: a team provider key reaches a model through the
tenant route, resolved by reference at the effect boundary, never from a
GitHub Actions secret. This ratchet scans ``.github/workflows`` for secret
references whose name is a provider's key. It fails closed: a provider list
cannot name every provider or every generic alias (LLM_API_KEY, MODEL_TOKEN),
so every credential secret a workflow reads must be declared in
``_NON_PROVIDER_CREDENTIALS``, and no declared name may be a provider's key.
The scan also refuses a secret read whose name is hidden (toJSON(secrets),
a computed index).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"

_PROVIDERS = (
    "AI21",
    "ANTHROPIC",
    "AZURE",
    "BEDROCK",
    "CEREBRAS",
    "CLAUDE",
    "COHERE",
    "DEEPSEEK",
    "FIREWORKS",
    "GEMINI",
    "GLM",
    "GOOGLE",
    "GROQ",
    "HF",
    "HUGGINGFACE",
    "KIMI",
    "MISTRAL",
    "MOONSHOT",
    "NVIDIA",
    "OLLAMA",
    "OPENAI",
    "OPENROUTER",
    "PERPLEXITY",
    "QWEN",
    "REPLICATE",
    "SAMBANOVA",
    "TOGETHER",
    "VERTEX",
    "VOYAGE",
    "XAI",
    "ZAI",
)
# Every credential secret a workflow may read. A new one is declared here, where
# review sees it; a provider key is refused even when declared.
_NON_PROVIDER_CREDENTIALS = frozenset(
    {
        "CHAIN_CANARY_GATEWAY_API_KEY",  # tenant gateway key: the sanctioned route
        "GITHUB_TOKEN",
        "INTEGRATION_POSTGRES_PASSWORD",
        "KAFKA_SASL_PASSWORD",
        "LINEAR_API_KEY",
        "LINEAR_CLOSER_CLIENT_SECRET",
        "LOCAL_LLM_SHARED_SECRET",  # lab model route shared secret, not a provider key
        "OMN18012_BROKER_SASL_PASSWORD",
        "OMNIBASE_CORE_TOKEN",
        "OMNI_GITHUB_TOKEN",
        "ONEXBOT-OCC-PRIVATE-KEY",
        "ONEXBOT_APP_PRIVATE_KEY",
        "ONEXBOT_OCC_PRIVATE_KEY",
        "PYPI_TOKEN",
        "RUNNER_FLEET_STATUS_TOKEN",
        "RUNTIME_PG_PASSWORD",
        "SLACK_BOT_TOKEN",
    }
)
# GitHub expression context names are case-insensitive, so every pattern is too.
_SECRET_REF = re.compile(
    r"secrets\s*\.\s*([A-Za-z0-9_-]+)|secrets\s*\[\s*'([^']+)'\s*\]"
    r'|secrets\s*\[\s*"([^"]+)"\s*\]',
    re.IGNORECASE,
)
_OPAQUE_SECRET_READ = re.compile(
    r"\btoJSON\s*\(\s*secrets\s*\)|\bsecrets\s*\[\s*[^\s'\"\]][^\]]*\]",
    re.IGNORECASE,
)
_CREDENTIAL = r"(?:KEY|TOKEN|SECRET|CREDENTIAL|AUTH|PASSWORD)"
# A provider token delimited by underscores, or run straight into a credential
# word (OPENAIKEY).
_PROVIDER_KEY_NAME = re.compile(
    rf"(?:^|_)(?:{'|'.join(_PROVIDERS)})(?:_|$|(?={_CREDENTIAL}))", re.IGNORECASE
)
_CREDENTIAL_WORD = re.compile(_CREDENTIAL, re.IGNORECASE)


def _secret_names(text: str) -> list[str]:
    return [
        match.group(1) or match.group(2) or match.group(3)
        for match in _SECRET_REF.finditer(text)
    ]


def _is_provider_key(name: str) -> bool:
    return bool(_PROVIDER_KEY_NAME.search(name) and _CREDENTIAL_WORD.search(name))


def provider_key_secrets(text: str) -> list[str]:
    """Names of the provider key secrets a workflow's text references."""
    return [name for name in _secret_names(text) if _is_provider_key(name)]


def undeclared_credential_secrets(text: str) -> list[str]:
    """Credential secrets a workflow reads that are not declared non-provider."""
    return [
        name
        for name in _secret_names(text)
        if _CREDENTIAL_WORD.search(name)
        and not _is_provider_key(name)
        and name.upper() not in _NON_PROVIDER_CREDENTIALS
    ]


def opaque_secret_reads(text: str) -> list[str]:
    """Secret reads whose names cannot be checked for a provider key."""
    return [match.group(0) for match in _OPAQUE_SECRET_READ.finditer(text)]


def _scanned_files() -> list[Path]:
    return sorted(
        {
            *_WORKFLOWS.glob("*.yml"),
            *_WORKFLOWS.glob("*.yaml"),
            *_WORKFLOWS.parent.rglob("action.yml"),
            *_WORKFLOWS.parent.rglob("action.yaml"),
        }
    )


def test_the_detector_flags_a_planted_provider_key_reference() -> None:
    """Positive control: the scan would catch the reference this ticket removed."""
    planted = (
        "env:\n"
        "  RUNG_EVAL_CLOUD_OPENROUTER_API_KEY: "
        "${{ secrets.RUNG_EVAL_CLOUD_OPENROUTER_API_KEY }}\n"
        "  OTHER: ${{ secrets['ANTHROPIC_API_KEY'] }}\n"
        '  THIRD: ${{ secrets["CLAUDE_API_KEY"] }}\n'
    )
    assert provider_key_secrets(planted) == [
        "RUNG_EVAL_CLOUD_OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",
    ]


def test_the_detector_ignores_the_case_of_the_secrets_context() -> None:
    planted = (
        "${{ Secrets.ANTHROPIC_API_KEY }}\n"
        "${{ SECRETS['OPENAI_API_KEY'] }}\n"
        "${{ TOJSON(Secrets) }}\n"
        "${{ SECRETS[matrix.name] }}\n"
    )
    assert provider_key_secrets(planted) == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    assert opaque_secret_reads(planted) == ["TOJSON(Secrets)", "SECRETS[matrix.name]"]


def test_the_detector_flags_other_credential_words_and_providers() -> None:
    planted = "".join(
        f"${{{{ secrets.{name} }}}}\n"
        for name in (
            "ANTHROPIC_CREDENTIALS",
            "OPENAI_AUTH",
            "OPENAIKEY",
            "GOOGLE_API_KEY",
            "AZURE_OPENAI_KEY",
            "VERTEX_SA_SECRET",
            "SAMBANOVA_API_KEY",
            "AI21_TOKEN",
            "OLLAMA_PASSWORD",
        )
    )
    assert len(provider_key_secrets(planted)) == 9
    # A credential-free provider-named value is not a key.
    assert provider_key_secrets("${{ secrets.OPENAI_BASE_URL }}") == []


def test_the_detector_flags_a_hidden_secret_name() -> None:
    planted = (
        "${{ toJSON(secrets) }}\n${{ secrets[format('{0}_API_KEY', matrix.p)] }}\n"
    )
    assert opaque_secret_reads(planted) == [
        "toJSON(secrets)",
        "secrets[format('{0}_API_KEY', matrix.p)]",
    ]
    assert opaque_secret_reads("    secrets: inherit\n") == []
    assert opaque_secret_reads("${{ secrets['LINEAR_API_KEY'] }}") == []


def test_the_detector_leaves_a_non_provider_secret_alone() -> None:
    clean = (
        "LINEAR_API_KEY: ${{ secrets.LINEAR_API_KEY }}\nT: ${{ secrets.GITHUB_TOKEN }}"
    )
    assert provider_key_secrets(clean) == []
    assert undeclared_credential_secrets(clean) == []


def test_an_unlisted_provider_or_generic_alias_fails_closed() -> None:
    """A credential secret the provider list does not name is still refused."""
    planted = "".join(
        f"${{{{ secrets.{name} }}}}\n"
        for name in ("LLM_API_KEY", "GATEWAY_API_KEY", "MODEL_TOKEN", "ACME_AI_SECRET")
    )
    assert provider_key_secrets(planted) == []
    assert undeclared_credential_secrets(planted) == [
        "LLM_API_KEY",
        "GATEWAY_API_KEY",
        "MODEL_TOKEN",
        "ACME_AI_SECRET",
    ]
    # A declared name is accepted in any case, as GitHub resolves it.
    assert undeclared_credential_secrets("${{ secrets.linear_api_key }}") == []


def test_no_declared_credential_is_a_provider_key() -> None:
    assert [name for name in _NON_PROVIDER_CREDENTIALS if _is_provider_key(name)] == []


def test_workflows_are_scanned() -> None:
    """Zero-hit control: the scan reads real files, so an empty result means clean."""
    scanned = {path.relative_to(_ROOT) for path in _scanned_files()}
    assert Path(".github/workflows/hostile-reviewer.yml") in scanned
    assert Path(".github/actions/setup-python-uv/action.yml") in scanned


def test_no_provider_key_in_workflows() -> None:
    scanned = _scanned_files()
    # An empty scan would make the zero-offender assertion below vacuous.
    assert _WORKFLOWS / "hostile-reviewer.yml" in scanned, (
        f"the scan found no hostile-reviewer.yml under {_WORKFLOWS}: {scanned}"
    )
    offenders = {
        str(path.relative_to(_ROOT)): names
        for path in scanned
        if (
            names := provider_key_secrets(text := path.read_text(encoding="utf-8"))
            + undeclared_credential_secrets(text)
            + opaque_secret_reads(text)
        )
    }
    assert offenders == {}, (
        "a workflow references a provider key secret; a team key is read by "
        f"reference through the tenant route, not from Actions: {offenders}"
    )
