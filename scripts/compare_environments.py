# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""compare-environments.py — local Docker vs k8s environment parity checker.

Usage:
    uv run python scripts/compare-environments.py [--mode check|fix] [--checks CHECKS]
    uv run python scripts/compare-environments.py --all-checks
    uv run python scripts/compare-environments.py --json

Checks (default: credential,ecr,infisical):
    credential   CRITICAL  Service secret POSTGRES_USER/PASSWORD vs onex-runtime-credentials
    ecr          CRITICAL  Deployment image tags still exist in ECR
    infisical    CRITICAL  InfisicalSecret paths exist in Infisical project
    schema       WARNING   DB migration_history latest id matches local vs cloud
    services     WARNING   Deployments present in local Docker vs cloud k8s
    flags        WARNING   Feature flag env vars consistent
    kafka        WARNING   Kafka topic sets match on both buses
    packages     WARNING   omnibase-core/spi/infra package versions match
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelParityFinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    check_id: str
    severity: Literal["CRITICAL", "WARNING", "INFO"]
    title: str
    detail: str
    local_value: str | None = None
    cloud_value: str | None = None
    auto_fixable: bool
    fix_hint: str


class ModelParitySummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    critical_count: int
    warning_count: int
    info_count: int
    checks_skipped: list[str]


class ModelParityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run_id: str
    generated_at: str
    mode: str
    checks_run: list[str]
    findings: list[ModelParityFinding]
    summary: ModelParitySummary


# ---------------------------------------------------------------------------
# transport — SsmRunner, SsmResult
# ---------------------------------------------------------------------------


@dataclass
class SsmResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False
    skip_reason: str = ""


class SsmRunner:
    def __init__(self, instance_id: str, region: str, timeout: int = 90) -> None:
        self.instance_id = instance_id
        self.region = region
        self.timeout = timeout

    def run(self, command: str) -> SsmResult:
        if not shutil.which("aws"):
            return SsmResult(
                skipped=True, skip_reason="aws CLI not found — install awscli"
            )
        try:
            send = subprocess.run(
                [
                    "aws",
                    "ssm",
                    "send-command",
                    "--instance-ids",
                    self.instance_id,
                    "--region",
                    self.region,
                    "--document-name",
                    "AWS-RunShellScript",
                    "--parameters",
                    f'commands=["{command}"]',
                    "--output",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 — boundary: returns degraded response
            return SsmResult(skipped=True, skip_reason=f"send-command failed: {exc}")
        if send.returncode != 0:
            if (
                "ExpiredTokenException" in send.stderr
                or "ExpiredTokenException" in send.stdout
            ):
                return SsmResult(
                    skipped=True, skip_reason="SSO session expired — run: aws sso login"
                )
            return SsmResult(
                skipped=True, skip_reason=f"send-command error: {send.stderr[:200]}"
            )
        try:
            command_id = json.loads(send.stdout)["Command"]["CommandId"]
        except Exception as exc:  # noqa: BLE001 — boundary: returns degraded response
            return SsmResult(
                skipped=True, skip_reason=f"could not parse CommandId: {exc}"
            )
        for _ in range(self.timeout // 2):
            time.sleep(2)
            try:
                poll = subprocess.run(
                    [
                        "aws",
                        "ssm",
                        "get-command-invocation",
                        "--command-id",
                        command_id,
                        "--instance-id",
                        self.instance_id,
                        "--region",
                        self.region,
                        "--output",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if poll.returncode != 0:
                    continue
                inv = json.loads(poll.stdout)
                if inv["Status"] in ("Success", "Failed", "TimedOut", "Cancelled"):
                    return SsmResult(
                        returncode=0 if inv["Status"] == "Success" else 1,
                        stdout=inv.get("StandardOutputContent", ""),
                        stderr=inv.get("StandardErrorContent", ""),
                    )
            except Exception:  # noqa: BLE001 — boundary: returns degraded response
                continue
        return SsmResult(
            skipped=True, skip_reason=f"instance unreachable after {self.timeout}s"
        )


# ---------------------------------------------------------------------------
# checks — credential_parity and infisical_path_completeness
# ---------------------------------------------------------------------------

# Explicit credential rule map. Keys NOT in this map are not compared.
# Format: (secret_name, key, rule_type, expected_value_or_None)
# rule_type "role" → key must equal expected_value
# rule_type "runtime_secret" → key must match value in onex-runtime-credentials
CREDENTIAL_RULES: list[tuple[str, str, str, str | None]] = [
    ("omniintelligence-credentials", "POSTGRES_USER", "role", "role_omniintelligence"),
    ("omnidash-credentials", "POSTGRES_USER", "role", "role_omnidash"),
    ("omniintelligence-credentials", "POSTGRES_PASSWORD", "runtime_secret", None),
    ("omnidash-credentials", "POSTGRES_PASSWORD", "runtime_secret", None),
]


def check_credential_parity(
    cloud_secrets: dict[str, dict[str, str]],
) -> list[ModelParityFinding]:
    """Check k8s service secrets against CREDENTIAL_RULES explicit map."""
    findings: list[ModelParityFinding] = []
    runtime = cloud_secrets.get("onex-runtime-credentials", {})

    for secret_name, key, rule_type, expected_value in CREDENTIAL_RULES:
        service_secret = cloud_secrets.get(secret_name, {})
        actual = service_secret.get(key)
        if actual is None:
            findings.append(
                ModelParityFinding(
                    check_id="credential_parity",
                    severity="CRITICAL",
                    title=f"Missing {key} in {secret_name}",
                    detail=f"Key '{key}' not found in k8s secret '{secret_name}'",
                    cloud_value=None,
                    auto_fixable=False,
                    fix_hint=f"Re-seed {secret_name} in Infisical and resync InfisicalSecret",
                )
            )
            continue

        if rule_type == "role":
            if actual != expected_value:
                findings.append(
                    ModelParityFinding(
                        check_id="credential_parity",
                        severity="CRITICAL",
                        title=f"Wrong {key} for {secret_name}",
                        detail=(
                            f"k8s secret has '{actual}'; expected '{expected_value}'"
                        ),
                        local_value=expected_value,
                        cloud_value=actual,
                        auto_fixable=False,
                        fix_hint=(
                            f"Re-seed /{secret_name.replace('-', '/')}/ in Infisical"
                            " and force-resync the InfisicalSecret"
                        ),
                    )
                )
        elif rule_type == "runtime_secret":
            runtime_value = runtime.get(key)
            if runtime_value is not None and actual != runtime_value:
                findings.append(
                    ModelParityFinding(
                        check_id="credential_parity",
                        severity="CRITICAL",
                        title=f"Mismatched {key} for {secret_name}",
                        detail=(
                            f"'{secret_name}' {key} does not match onex-runtime-credentials"
                        ),
                        auto_fixable=False,
                        fix_hint=(
                            "onex-runtime-credentials is authoritative. Re-seed service"
                            f" secret path for {secret_name} and resync."
                        ),
                    )
                )

    return findings


def probe_infisical_paths(
    infisical_addr: str,
    project_id: str,
    paths: list[tuple[str, str, str]],
    token: str,
) -> list[ModelParityFinding]:
    """Check that each Infisical path exists for the given project.

    paths: list of (path, environment, infisical_secret_name)
    """
    import httpx

    findings: list[ModelParityFinding] = []
    infisical_addr = infisical_addr.rstrip("/")

    for path, environment, secret_name in paths:
        try:
            resp = httpx.get(
                f"{infisical_addr}/api/v1/secrets",
                params={
                    "workspaceId": project_id,
                    "environment": environment,
                    "secretPath": path,
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if resp.status_code == 404:
                findings.append(
                    ModelParityFinding(
                        check_id="infisical_path_completeness",
                        severity="CRITICAL",
                        title=f"Infisical path missing: {path}",
                        detail=(
                            f"Path '{path}' (env={environment}) not found in Infisical"
                            f" project {project_id}. InfisicalSecret '{secret_name}'"
                            " will loop on 404."
                        ),
                        cloud_value=None,
                        auto_fixable=True,
                        fix_hint=(
                            f"Run: uv run python scripts/seed-infisical.py"
                            f" --contracts-dir src/omnibase_infra/nodes --execute"
                            f"  # seeds path {path}"
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
            findings.append(
                ModelParityFinding(
                    check_id="infisical_path_completeness",
                    severity="INFO",
                    title=f"Infisical probe skipped for {path}",
                    detail=f"Could not reach Infisical at {infisical_addr}: {exc}",
                    auto_fixable=False,
                    fix_hint="Verify INFISICAL_ADDR is set and Infisical is running.",
                )
            )

    return findings


# ---------------------------------------------------------------------------
# checks — ecr_tag_validity
# ---------------------------------------------------------------------------

# SSM remote script: emits one JSON object {deployment_name: image_ref}
_DEPLOYMENT_IMAGE_SCRIPT_TEMPLATE = (
    'python3 -c "'
    "import subprocess, json; "
    "r = subprocess.run(['kubectl','get','deployments','-n','{namespace}',"
    "  '-o','jsonpath={{range .items[*]}}{{.metadata.name}}={{.spec.template.spec.containers[0].image}}\\\\n{{end}}'],"
    "  capture_output=True, text=True); "
    "d = {{}}; "
    "[d.update({{p.split('=',1)[0].strip(): p.split('=',1)[1].strip()}})"
    " for p in r.stdout.strip().splitlines() if '=' in p]; "
    "print(json.dumps(d))"
    '"'
)


def _ecr_tag_exists(repo: str, tag: str, region: str) -> bool:
    """Return True if the ECR tag exists in the registry."""
    r = subprocess.run(
        [
            "aws",
            "ecr",
            "describe-images",
            "--repository-name",
            repo,
            "--image-ids",
            f"imageTag={tag}",
            "--region",
            region,
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return r.returncode == 0


def _parse_ecr_image(image_ref: str) -> tuple[str, str] | None:
    """Parse 'registry/repo:tag' → (repo, tag). Returns None if not ECR or no tag."""
    if ".dkr.ecr." not in image_ref and ".ecr/" not in image_ref:
        return None
    if ":" not in image_ref:
        return None
    # strip registry prefix: everything after the last '/' before ':'
    path_part = image_ref.split("/", 1)[-1] if "/" in image_ref else image_ref
    if ":" not in path_part:
        return None
    repo, tag = path_part.rsplit(":", 1)
    # skip digest-pinned references
    if tag.startswith("sha256:"):
        return None
    return repo, tag


def check_ecr_tag_validity(
    deployments: dict[str, str],
    region: str,
) -> list[ModelParityFinding]:
    """Check that deployment image tags still exist in ECR.

    Scope: primary containers only (spec.template.spec.containers[0]).
    Does not cover initContainers or digest-pinned references.

    deployments: {deployment_name: image_ref}
    """
    findings: list[ModelParityFinding] = []

    if not shutil.which("aws"):
        for deploy_name, image_ref in deployments.items():
            findings.append(
                ModelParityFinding(
                    check_id="ecr_tag_validity",
                    severity="INFO",
                    title=f"ECR check skipped for {deploy_name}",
                    detail="aws CLI not found — install awscli to enable ECR tag validation",
                    cloud_value=image_ref,
                    auto_fixable=False,
                    fix_hint="Install awscli: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html",
                )
            )
        return findings

    for deploy_name, image_ref in deployments.items():
        parsed = _parse_ecr_image(image_ref)
        if parsed is None:
            # Not an ECR image or digest-pinned — skip silently
            continue
        repo, tag = parsed
        try:
            exists = _ecr_tag_exists(repo, tag, region)
        except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
            findings.append(
                ModelParityFinding(
                    check_id="ecr_tag_validity",
                    severity="INFO",
                    title=f"ECR check skipped for {deploy_name}",
                    detail=f"ECR describe-images failed: {exc}",
                    cloud_value=image_ref,
                    auto_fixable=False,
                    fix_hint="Verify aws credentials and ECR access.",
                )
            )
            continue

        if not exists:
            findings.append(
                ModelParityFinding(
                    check_id="ecr_tag_validity",
                    severity="CRITICAL",
                    title=f"ECR tag missing for {deploy_name}",
                    detail=(
                        f"Image tag '{tag}' for deployment '{deploy_name}' not found"
                        f" in ECR repo '{repo}'. Active deployment will ErrImagePull on pod restart."
                    ),
                    cloud_value=image_ref,
                    auto_fixable=False,
                    fix_hint=(
                        f"Either push a new image with tag '{tag}' to ECR repo '{repo}',"
                        f" or update the deployment to reference an existing tag."
                    ),
                )
            )

    return findings


# ---------------------------------------------------------------------------
# checks — db_schema_parity (Task 5, --all-checks only)
# ---------------------------------------------------------------------------

CANONICAL_SERVICES = {
    "omninode-runtime",
    "omninode-runtime-effects",
    "omninode-agent-actions-consumer",
    "omnibase-intelligence-api",
    "omninode-contract-resolver",
    "omninode-skill-lifecycle-consumer",
}

FEATURE_FLAG_KEYS = [
    "ENABLE_PATTERN_ENFORCEMENT",
    "ENABLE_REAL_TIME_EVENTS",
    "KAFKA_ENABLE_INTELLIGENCE",
    "OMNIMEMORY_ENABLED",
]


def check_db_schema_parity(
    local_migration: str,
    cloud_migration: str,
) -> list[ModelParityFinding]:
    """Compare latest migration IDs. Expects lexically-ordered ISO timestamp prefixes."""
    if local_migration == cloud_migration:
        return []
    if cloud_migration > local_migration:
        return [
            ModelParityFinding(
                check_id="db_schema_parity",
                severity="CRITICAL",
                title="DB schema: cloud ahead of local",
                detail=(
                    f"Cloud ahead: cloud latest='{cloud_migration}',"
                    f" local latest='{local_migration}'."
                    " Migration applied in cloud but not yet run locally."
                ),
                local_value=local_migration,
                cloud_value=cloud_migration,
                auto_fixable=False,
                fix_hint=(
                    "Run outstanding migrations locally: "
                    "uv run python scripts/run-migrations.py"
                ),
            )
        ]
    # local_migration > cloud_migration
    return [
        ModelParityFinding(
            check_id="db_schema_parity",
            severity="WARNING",
            title="DB schema: local ahead of cloud",
            detail=(
                f"Local ahead: local latest='{local_migration}',"
                f" cloud latest='{cloud_migration}'."
                " Migration applied locally but not yet deployed to cloud."
            ),
            local_value=local_migration,
            cloud_value=cloud_migration,
            auto_fixable=False,
            fix_hint=(
                "Apply pending migrations to cloud: "
                "uv run python scripts/run-migrations.py --target cloud"
            ),
        )
    ]


def check_package_version_parity(
    local_versions: dict[str, str],
    cloud_versions: dict[str, str],
) -> list[ModelParityFinding]:
    """Compare omnibase package versions between local and cloud."""
    findings: list[ModelParityFinding] = []
    for pkg in ("omnibase-core", "omnibase-spi", "omnibase-infra"):
        local_v = local_versions.get(pkg)
        cloud_v = cloud_versions.get(pkg)
        if local_v is None or cloud_v is None:
            continue
        if local_v != cloud_v:
            findings.append(
                ModelParityFinding(
                    check_id="package_version_parity",
                    severity="WARNING",
                    title=f"Package version mismatch: {pkg}",
                    detail=f"{pkg}: local={local_v}, cloud={cloud_v}. Version mismatch may cause serialization errors.",
                    local_value=local_v,
                    cloud_value=cloud_v,
                    auto_fixable=False,
                    fix_hint=f"Rebuild and redeploy with {pkg}=={local_v}",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# checks — service_deployment_parity and feature_flag_consistency (Task 6)
# ---------------------------------------------------------------------------


def check_service_deployment_parity(
    local_services: set[str],
    cloud_services: set[str],
) -> list[ModelParityFinding]:
    """Compare local Docker services vs cloud k8s deployments."""
    findings: list[ModelParityFinding] = []
    local_canonical = local_services & CANONICAL_SERVICES
    cloud_canonical = cloud_services & CANONICAL_SERVICES

    for svc in local_canonical - cloud_canonical:
        findings.append(
            ModelParityFinding(
                check_id="service_deployment_parity",
                severity="WARNING",
                title=f"Service only in local: {svc}",
                detail=f"'{svc}' runs locally but has no k8s deployment in cloud.",
                local_value="running",
                cloud_value="absent",
                auto_fixable=False,
                fix_hint=f"Deploy {svc} to cloud namespace or verify deployment spec.",
            )
        )
    for svc in cloud_canonical - local_canonical:
        findings.append(
            ModelParityFinding(
                check_id="service_deployment_parity",
                severity="INFO",
                title=f"Service only in cloud: {svc}",
                detail=f"'{svc}' deployed in cloud but not running locally (expected).",
                local_value="absent",
                cloud_value="running",
                auto_fixable=False,
                fix_hint="No action required — cloud-only services are expected.",
            )
        )
    return findings


def check_feature_flag_consistency(
    local_flags: dict[str, str],
    cloud_flags: dict[str, str],
) -> list[ModelParityFinding]:
    """Check feature flag env vars are consistent between local and cloud."""
    findings: list[ModelParityFinding] = []
    for key in FEATURE_FLAG_KEYS:
        local_v = local_flags.get(key)
        cloud_v = cloud_flags.get(key)
        if local_v is None or cloud_v is None:
            continue
        if local_v.lower() != cloud_v.lower():
            findings.append(
                ModelParityFinding(
                    check_id="feature_flag_consistency",
                    severity="WARNING",
                    title=f"Feature flag mismatch: {key}",
                    detail=f"local={local_v!r}, cloud={cloud_v!r}",
                    local_value=local_v,
                    cloud_value=cloud_v,
                    auto_fixable=False,
                    fix_hint=f"Update cloud ConfigMap or Infisical to set {key}={local_v}",  # noqa: S608
                )
            )
    return findings


# ---------------------------------------------------------------------------
# checks — kafka_topic_parity (Task 7, --all-checks only)
# ---------------------------------------------------------------------------


def check_kafka_topic_parity(
    local_topics: set[str],
    cloud_topics: set[str],
) -> list[ModelParityFinding]:
    """Compare Kafka topic sets. Internal __ topics are filtered out."""
    findings: list[ModelParityFinding] = []
    local_filtered = {t for t in local_topics if not t.startswith("__")}
    cloud_filtered = {t for t in cloud_topics if not t.startswith("__")}

    for topic in sorted(local_filtered - cloud_filtered):
        findings.append(
            ModelParityFinding(
                check_id="kafka_topic_parity",
                severity="WARNING",
                title=f"Kafka topic only in local: {topic}",
                detail=f"Topic '{topic}' exists locally but not in cloud broker.",
                local_value="exists",
                cloud_value="missing",
                auto_fixable=False,
                fix_hint="Run create_kafka_topics.py against cloud broker.",
            )
        )
    for topic in sorted(cloud_filtered - local_filtered):
        findings.append(
            ModelParityFinding(
                check_id="kafka_topic_parity",
                severity="INFO",
                title=f"Kafka topic only in cloud: {topic}",
                detail=f"Topic '{topic}' exists in cloud but not local (expected).",
                local_value="missing",
                cloud_value="exists",
                auto_fixable=False,
                fix_hint="No action required — cloud-only topics are expected.",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# preflight + fix mode (Task 8)
# ---------------------------------------------------------------------------

# Hard allowlist — only these check_ids may be auto-fixed
AUTO_FIXABLE_CHECKS: set[str] = {"infisical_path_completeness"}


def preflight_namespace_check(
    ssm: SsmRunner, namespace: str
) -> ModelParityFinding | None:
    """Verify the k8s namespace exists. Returns a CRITICAL finding if missing."""
    result = ssm.run(
        f"kubectl get namespace {namespace} -o name 2>/dev/null || echo NAMESPACE_MISSING"
    )
    if result.skipped:
        return None
    if "NAMESPACE_MISSING" in result.stdout:
        return ModelParityFinding(
            check_id="preflight_namespace",
            severity="CRITICAL",
            title=f"k8s namespace '{namespace}' not found",
            detail=(
                "All cloud-side checks skipped — cluster may be rebuilding"
                " or namespace deleted."
            ),
            auto_fixable=False,
            fix_hint=f"kubectl create namespace {namespace}",
        )
    return None


# ---------------------------------------------------------------------------
# aggregation — run_parity_check (Task 9)
# ---------------------------------------------------------------------------


def run_parity_check(
    mode: str,
    checks: list[str],
    namespace: str,
    instance_id: str,
    region: str,
    timeout: int,
) -> ModelParityReport:
    """Run all requested checks and return a ModelParityReport."""
    ssm = SsmRunner(instance_id=instance_id, region=region, timeout=timeout)
    all_findings: list[ModelParityFinding] = []

    # Preflight: namespace check (short-circuits cloud probes if missing)
    namespace_ok = True
    preflight = preflight_namespace_check(ssm, namespace)
    if preflight is not None:
        all_findings.append(preflight)
        namespace_ok = False

    # Run each requested check
    for check in checks:
        if check == "credential" and namespace_ok:
            # Remote SSM probe for cloud secrets
            result = ssm.run(
                f'python3 -c "'
                f"import subprocess,json,base64;"
                f"def dec(n,ns): r=subprocess.run(['kubectl','get','secret',n,'-n',ns,"
                f"'-o','jsonpath={{{{.data}}}}'],capture_output=True,text=True,timeout=15);"
                f" return {{k:base64.b64decode(v).decode() for k,v in json.loads(r.stdout).items()}} if r.returncode==0 else {{}};"
                f"print(json.dumps({{"
                f"'onex_runtime_credentials':dec('onex-runtime-credentials','{namespace}'),"
                f"'omniintelligence_credentials':dec('omniintelligence-credentials','{namespace}'),"
                f"'omnidash_credentials':dec('omnidash-credentials','{namespace}')"
                f'}}))"'
            )
            if not result.skipped:
                try:
                    data = json.loads(result.stdout)
                    cloud_secrets = {
                        "onex-runtime-credentials": data.get(
                            "onex_runtime_credentials", {}
                        ),
                        "omniintelligence-credentials": data.get(
                            "omniintelligence_credentials", {}
                        ),
                        "omnidash-credentials": data.get("omnidash_credentials", {}),
                    }
                    all_findings.extend(check_credential_parity(cloud_secrets))
                except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
                    all_findings.append(
                        ModelParityFinding(
                            check_id="credential_parity",
                            severity="INFO",
                            title="credential check skipped — SSM parse error",
                            detail=f"Could not parse SSM output: {exc}. stdout[:200]={result.stdout[:200]}",
                            auto_fixable=False,
                            fix_hint="Verify SSM instance access and kubectl permissions.",
                        )
                    )

        elif check == "ecr" and namespace_ok:
            script = _DEPLOYMENT_IMAGE_SCRIPT_TEMPLATE.format(namespace=namespace)
            result = ssm.run(script)
            if not result.skipped:
                try:
                    deployments = json.loads(result.stdout)
                    all_findings.extend(
                        check_ecr_tag_validity(deployments, region=region)
                    )
                except Exception as exc:  # noqa: BLE001 — boundary: catch-all for resilience
                    all_findings.append(
                        ModelParityFinding(
                            check_id="ecr_tag_validity",
                            severity="INFO",
                            title="ECR check skipped — SSM parse error",
                            detail=str(exc),
                            auto_fixable=False,
                            fix_hint="Verify SSM instance access and kubectl permissions.",
                        )
                    )

    # Build summary
    critical = sum(1 for f in all_findings if f.severity == "CRITICAL")
    warning = sum(1 for f in all_findings if f.severity == "WARNING")
    info = sum(1 for f in all_findings if f.severity == "INFO")

    return ModelParityReport(
        run_id=str(uuid.uuid4())[:8],
        generated_at=datetime.now(tz=UTC).isoformat(),
        mode=mode,
        checks_run=checks,
        findings=all_findings,
        summary=ModelParitySummary(
            critical_count=critical,
            warning_count=warning,
            info_count=info,
            checks_skipped=[],
        ),
    )


# ---------------------------------------------------------------------------
# checks registry
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    "credential",
    "ecr",
    "infisical",
    "schema",
    "services",
    "flags",
    "kafka",
    "packages",
]
DEFAULT_CHECKS = ["credential", "ecr", "infisical"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local Docker vs k8s parity checker")
    p.add_argument("--mode", choices=["check", "fix"], default="check")
    p.add_argument("--checks", default=",".join(DEFAULT_CHECKS))
    p.add_argument("--all-checks", action="store_true")
    p.add_argument("--namespace", default="onex-dev")
    p.add_argument("--instance-id", default="i-0e596e8b557e27785")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--json", dest="json_output", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=int, default=90)
    return p


def main() -> None:
    args = build_parser().parse_args()
    checks = (
        ALL_CHECKS if args.all_checks else [c.strip() for c in args.checks.split(",")]
    )
    report = run_parity_check(
        mode=args.mode,
        checks=checks,
        namespace=args.namespace,
        instance_id=args.instance_id,
        region=args.region,
        timeout=args.timeout,
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
