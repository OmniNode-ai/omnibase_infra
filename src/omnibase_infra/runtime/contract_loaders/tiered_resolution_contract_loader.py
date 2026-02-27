# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Contract YAML loader for tiered resolution and trust domain configuration.

Loads the ``tiered_resolution`` section from dependency blocks and the
``trust_domains`` top-level section from ONEX contract YAML files. Bridges
declarative YAML configuration into runtime objects for the
``ServiceTieredResolver``.

This is Phase 7 Part 2 of the Authenticated Dependency Resolution epic
(OMN-2897). It connects the contract-declared resolution constraints to
the runtime tiered resolver and multi-bus topic resolver (Phase 5).

All new YAML sections are **optional** -- existing contracts without these
sections load unchanged with no errors.

Contract Structure::

    # Per-dependency tiered resolution constraints
    dependencies:
      - alias: "db"
        capability: "database.relational"
        tiered_resolution:
          min_tier: "local_exact"
          max_tier: "org_trusted"
          require_proofs: ["node_identity", "capability_attested"]
          classification: "internal"

    # Top-level trust domain declarations
    trust_domains:
      - domain_id: "local.default"
        tier: "local_exact"
      - domain_id: "org.omninode"
        tier: "org_trusted"
        trust_root_ref: "secrets://keys/org-omninode-trust-root"

Usage::

    from pathlib import Path
    from omnibase_infra.runtime.contract_loaders.tiered_resolution_contract_loader import (
        load_tiered_resolution_configs,
        load_trust_domain_configs,
        load_tiered_resolution_from_contract,
    )

    contract_path = Path("nodes/my_node/contract.yaml")

    # Load all tiered resolution configs from dependencies
    configs = load_tiered_resolution_configs(contract_path)

    # Load trust domain declarations
    domains = load_trust_domain_configs(contract_path)

    # Load both in one call
    result = load_tiered_resolution_from_contract(contract_path)

See Also:
    - ModelTieredResolutionConfigLocal: Per-dependency resolution constraints
    - ModelTrustDomainConfigLocal: Trust domain declarations
    - ModelBusDescriptor: Phase 5 bus descriptor for topic routing
    - TopicResolver: Phase 5 multi-bus topic resolution

Related:
    - OMN-2896: Contract YAML Integration (Phase 7 of OMN-2897 epic)
    - OMN-2894: Multi-Bus Topic Resolution (Phase 5)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.services.resolution.model_tiered_resolution_config_local import (
    VALID_CLASSIFICATIONS,
    VALID_PROOF_TYPES,
    VALID_RESOLUTION_TIERS,
    ModelTieredResolutionConfigLocal,
)
from omnibase_infra.services.resolution.model_trust_domain_config_local import (
    ModelTrustDomainConfigLocal,
)
from omnibase_infra.topics.model_bus_descriptor import ModelBusDescriptor

logger = logging.getLogger(__name__)

# Maximum allowed file size for contract.yaml files (10MB).
# Security control to prevent memory exhaustion via large YAML files.
MAX_CONTRACT_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB

# Error codes for this loader (TIERED_LOADER_0xx series).
ERROR_CODE_CONTRACT_NOT_FOUND = "TIERED_LOADER_010"
ERROR_CODE_YAML_PARSE_ERROR = "TIERED_LOADER_011"
ERROR_CODE_CONTRACT_NOT_DICT = "TIERED_LOADER_012"
ERROR_CODE_FILE_SIZE_EXCEEDED = "TIERED_LOADER_050"
ERROR_CODE_INVALID_TIER = "TIERED_LOADER_020"
ERROR_CODE_INVALID_PROOF_TYPE = "TIERED_LOADER_021"
ERROR_CODE_INVALID_CLASSIFICATION = "TIERED_LOADER_022"
ERROR_CODE_INVALID_TRUST_DOMAIN = "TIERED_LOADER_023"
ERROR_CODE_TIER_RANGE_INVALID = "TIERED_LOADER_024"

# Ordered tiers for range validation.
_TIER_ORDER: tuple[str, ...] = (
    "local_exact",
    "local_compatible",
    "org_trusted",
    "federated_trusted",
    "quarantine",
)
_TIER_INDEX: dict[str, int] = {tier: idx for idx, tier in enumerate(_TIER_ORDER)}


class ModelTieredResolutionContractResult:
    """Result of loading tiered resolution and trust domain config from a contract.

    Attributes:
        tiered_configs: Mapping from dependency alias to its tiered resolution
            configuration. Only includes dependencies that declare a
            ``tiered_resolution`` section.
        trust_domains: List of trust domain declarations from the
            ``trust_domains`` top-level section.
        bus_descriptors: List of bus descriptors bridged from trust domain
            declarations. Each trust domain generates a bus descriptor with
            the domain_id as trust_domain, an auto-generated bus_id, and
            sensible defaults.
    """

    __slots__ = ("bus_descriptors", "tiered_configs", "trust_domains")

    def __init__(
        self,
        tiered_configs: dict[str, ModelTieredResolutionConfigLocal],
        trust_domains: list[ModelTrustDomainConfigLocal],
        bus_descriptors: list[ModelBusDescriptor],
    ) -> None:
        self.tiered_configs = tiered_configs
        self.trust_domains = trust_domains
        self.bus_descriptors = bus_descriptors


def _check_file_size(contract_path: Path, operation: str) -> None:
    """Check that contract file does not exceed maximum allowed size.

    Args:
        contract_path: Path to the contract.yaml file.
        operation: Name of the operation for error context.

    Raises:
        ProtocolConfigurationError: If file exceeds MAX_CONTRACT_FILE_SIZE_BYTES.
    """
    try:
        file_size = contract_path.stat().st_size
    except FileNotFoundError:
        return

    if file_size > MAX_CONTRACT_FILE_SIZE_BYTES:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation=operation,
            target_name=str(contract_path),
        )
        logger.error(
            "Contract file exceeds maximum size: %d bytes > %d bytes at %s",
            file_size,
            MAX_CONTRACT_FILE_SIZE_BYTES,
            contract_path,
        )
        raise ProtocolConfigurationError(
            f"Contract file exceeds maximum size: {file_size} bytes > "
            f"{MAX_CONTRACT_FILE_SIZE_BYTES} bytes. "
            f"Error code: FILE_SIZE_EXCEEDED ({ERROR_CODE_FILE_SIZE_EXCEEDED})",
            context=ctx,
        )


# ONEX_EXCLUDE: any_type - yaml.safe_load returns heterogeneous dict from contract YAML
def _load_contract_yaml(
    contract_path: Path,
    operation: str,
) -> dict[str, Any]:
    """Load and validate a contract.yaml file.

    Args:
        contract_path: Path to the contract.yaml file.
        operation: Name of the calling operation for error context.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ProtocolConfigurationError: If file does not exist, contains
            invalid YAML, exceeds file size limit, or root is not a dict.
    """
    _check_file_size(contract_path, operation)

    ctx = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.FILESYSTEM,
        operation=operation,
        target_name=str(contract_path),
    )

    if not contract_path.exists():
        raise ProtocolConfigurationError(
            f"Contract file not found: {contract_path}. "
            f"Error code: CONTRACT_NOT_FOUND ({ERROR_CODE_CONTRACT_NOT_FOUND})",
            context=ctx,
        )

    try:
        with contract_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        error_type = type(e).__name__
        logger.exception(
            "Invalid YAML syntax in contract.yaml at %s: %s",
            contract_path,
            error_type,
        )
        raise ProtocolConfigurationError(
            f"Invalid YAML syntax in contract.yaml at {contract_path}: {error_type}. "
            f"Error code: YAML_PARSE_ERROR ({ERROR_CODE_YAML_PARSE_ERROR})",
            context=ctx,
        ) from e

    if data is None:
        # Empty file -- valid but nothing to extract.
        return {}

    if not isinstance(data, dict):
        raise ProtocolConfigurationError(
            f"Contract YAML root is not a dict in {contract_path}: "
            f"got {type(data).__name__}. "
            f"Error code: CONTRACT_NOT_DICT ({ERROR_CODE_CONTRACT_NOT_DICT})",
            context=ctx,
        )

    return data


def _validate_tier(
    tier: str,
    field_name: str,
    contract_path: Path,
) -> None:
    """Validate that a tier name is a known resolution tier.

    Args:
        tier: The tier name to validate.
        field_name: The field name for error messages.
        contract_path: Path for error context.

    Raises:
        ProtocolConfigurationError: If the tier is not valid.
    """
    if tier not in VALID_RESOLUTION_TIERS:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="validate_tier",
            target_name=str(contract_path),
        )
        raise ProtocolConfigurationError(
            f"Invalid tier '{tier}' in {field_name} at {contract_path}. "
            f"Valid tiers: {sorted(VALID_RESOLUTION_TIERS)}. "
            f"Error code: INVALID_TIER ({ERROR_CODE_INVALID_TIER})",
            context=ctx,
        )


def _validate_tier_range(
    min_tier: str | None,
    max_tier: str | None,
    contract_path: Path,
    dep_alias: str,
) -> None:
    """Validate that min_tier <= max_tier in the tier ordering.

    Args:
        min_tier: The minimum tier (may be None).
        max_tier: The maximum tier (may be None).
        contract_path: Path for error context.
        dep_alias: Dependency alias for error messages.

    Raises:
        ProtocolConfigurationError: If min_tier > max_tier.
    """
    if min_tier is not None and max_tier is not None:
        min_idx = _TIER_INDEX.get(min_tier)
        max_idx = _TIER_INDEX.get(max_tier)
        if min_idx is not None and max_idx is not None and min_idx > max_idx:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="validate_tier_range",
                target_name=str(contract_path),
            )
            raise ProtocolConfigurationError(
                f"Invalid tier range for dependency '{dep_alias}' at "
                f"{contract_path}: min_tier '{min_tier}' is higher than "
                f"max_tier '{max_tier}'. "
                f"Error code: TIER_RANGE_INVALID ({ERROR_CODE_TIER_RANGE_INVALID})",
                context=ctx,
            )


def _parse_tiered_resolution_config(
    raw: dict[str, object],
    dep_alias: str,
    contract_path: Path,
) -> ModelTieredResolutionConfigLocal:
    """Parse a tiered_resolution dict into ModelTieredResolutionConfigLocal.

    Args:
        raw: Raw dict from the ``tiered_resolution`` YAML block.
        dep_alias: Alias of the parent dependency (for error messages).
        contract_path: Path for error context.

    Returns:
        Validated ``ModelTieredResolutionConfigLocal`` instance.

    Raises:
        ProtocolConfigurationError: If any field value is invalid.
    """
    ctx_field = f"dependencies[alias={dep_alias}].tiered_resolution"

    # Validate min_tier
    min_tier = raw.get("min_tier")
    if min_tier is not None:
        min_tier = str(min_tier)
        _validate_tier(min_tier, f"{ctx_field}.min_tier", contract_path)

    # Validate max_tier
    max_tier = raw.get("max_tier")
    if max_tier is not None:
        max_tier = str(max_tier)
        _validate_tier(max_tier, f"{ctx_field}.max_tier", contract_path)

    # Validate tier range
    _validate_tier_range(min_tier, max_tier, contract_path, dep_alias)

    # Validate require_proofs
    raw_proofs = raw.get("require_proofs", [])
    proofs: list[str] = []
    if isinstance(raw_proofs, list):
        for proof in raw_proofs:
            proof_str = str(proof)
            if proof_str not in VALID_PROOF_TYPES:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.FILESYSTEM,
                    operation="validate_proof_type",
                    target_name=str(contract_path),
                )
                raise ProtocolConfigurationError(
                    f"Invalid proof type '{proof_str}' in "
                    f"{ctx_field}.require_proofs at {contract_path}. "
                    f"Valid proof types: {sorted(VALID_PROOF_TYPES)}. "
                    f"Error code: INVALID_PROOF_TYPE ({ERROR_CODE_INVALID_PROOF_TYPE})",
                    context=ctx,
                )
            proofs.append(proof_str)

    # Validate classification
    classification = raw.get("classification")
    if classification is not None:
        classification = str(classification)
        if classification not in VALID_CLASSIFICATIONS:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="validate_classification",
                target_name=str(contract_path),
            )
            raise ProtocolConfigurationError(
                f"Invalid classification '{classification}' in "
                f"{ctx_field}.classification at {contract_path}. "
                f"Valid classifications: {sorted(VALID_CLASSIFICATIONS)}. "
                f"Error code: INVALID_CLASSIFICATION ({ERROR_CODE_INVALID_CLASSIFICATION})",
                context=ctx,
            )

    return ModelTieredResolutionConfigLocal(
        min_tier=min_tier,
        max_tier=max_tier,
        require_proofs=tuple(proofs),
        classification=classification,
    )


def _parse_trust_domain_config(
    raw: dict[str, object],
    index: int,
    contract_path: Path,
) -> ModelTrustDomainConfigLocal:
    """Parse a trust domain dict into ModelTrustDomainConfigLocal.

    Args:
        raw: Raw dict from the ``trust_domains`` YAML list entry.
        index: Index in the trust_domains list (for error messages).
        contract_path: Path for error context.

    Returns:
        Validated ``ModelTrustDomainConfigLocal`` instance.

    Raises:
        ProtocolConfigurationError: If required fields are missing or
            field values are invalid.
    """
    domain_id = raw.get("domain_id")
    if not domain_id:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="validate_trust_domain",
            target_name=str(contract_path),
        )
        raise ProtocolConfigurationError(
            f"Missing 'domain_id' in trust_domains[{index}] at {contract_path}. "
            f"Error code: INVALID_TRUST_DOMAIN ({ERROR_CODE_INVALID_TRUST_DOMAIN})",
            context=ctx,
        )

    tier = raw.get("tier")
    if not tier:
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.FILESYSTEM,
            operation="validate_trust_domain",
            target_name=str(contract_path),
        )
        raise ProtocolConfigurationError(
            f"Missing 'tier' in trust_domains[{index}] at {contract_path}. "
            f"Error code: INVALID_TRUST_DOMAIN ({ERROR_CODE_INVALID_TRUST_DOMAIN})",
            context=ctx,
        )

    tier_str = str(tier)
    _validate_tier(tier_str, f"trust_domains[{index}].tier", contract_path)

    trust_root_ref = raw.get("trust_root_ref")
    if trust_root_ref is not None:
        trust_root_ref = str(trust_root_ref)

    return ModelTrustDomainConfigLocal(
        domain_id=str(domain_id),
        tier=tier_str,
        trust_root_ref=trust_root_ref,
    )


def load_tiered_resolution_configs(
    contract_path: Path,
) -> dict[str, ModelTieredResolutionConfigLocal]:
    """Load tiered_resolution config blocks from contract dependencies.

    Scans the ``dependencies`` section of a contract YAML file and extracts
    any ``tiered_resolution`` sub-blocks, keyed by the dependency ``alias``.
    Dependencies without an ``alias`` use their index as the key.

    Args:
        contract_path: Path to the contract.yaml file.

    Returns:
        Dictionary mapping dependency alias to its tiered resolution config.
        Empty dict if no dependencies declare tiered_resolution.

    Raises:
        ProtocolConfigurationError: If the contract file is invalid or
            contains invalid tiered_resolution values.
    """
    data = _load_contract_yaml(contract_path, "load_tiered_resolution_configs")

    dependencies = data.get("dependencies")
    if not dependencies or not isinstance(dependencies, list):
        return {}

    configs: dict[str, ModelTieredResolutionConfigLocal] = {}

    for idx, dep in enumerate(dependencies):
        if not isinstance(dep, dict):
            continue

        tiered_raw = dep.get("tiered_resolution")
        if tiered_raw is None or not isinstance(tiered_raw, dict):
            continue

        alias = str(dep.get("alias", idx))
        config = _parse_tiered_resolution_config(tiered_raw, alias, contract_path)
        configs[alias] = config

        logger.debug(
            "Loaded tiered_resolution config for dependency '%s' from %s",
            alias,
            contract_path,
        )

    if configs:
        logger.info(
            "Loaded %d tiered_resolution configs from %s",
            len(configs),
            contract_path,
        )

    return configs


def load_trust_domain_configs(
    contract_path: Path,
) -> list[ModelTrustDomainConfigLocal]:
    """Load trust domain declarations from contract YAML.

    Reads the ``trust_domains`` top-level section from a contract YAML file
    and returns a list of validated trust domain configurations.

    Args:
        contract_path: Path to the contract.yaml file.

    Returns:
        List of trust domain configurations. Empty list if the contract
        does not declare any trust domains.

    Raises:
        ProtocolConfigurationError: If the contract file is invalid or
            contains invalid trust domain declarations.
    """
    data = _load_contract_yaml(contract_path, "load_trust_domain_configs")

    trust_domains_raw = data.get("trust_domains")
    if not trust_domains_raw or not isinstance(trust_domains_raw, list):
        return []

    domains: list[ModelTrustDomainConfigLocal] = []

    for idx, entry in enumerate(trust_domains_raw):
        if not isinstance(entry, dict):
            logger.warning(
                "Skipping non-dict entry in trust_domains[%d] at %s",
                idx,
                contract_path,
            )
            continue

        domain = _parse_trust_domain_config(entry, idx, contract_path)
        domains.append(domain)

    if domains:
        logger.info(
            "Loaded %d trust domain configs from %s",
            len(domains),
            contract_path,
        )

    return domains


def bridge_trust_domains_to_bus_descriptors(
    trust_domains: list[ModelTrustDomainConfigLocal],
) -> list[ModelBusDescriptor]:
    """Bridge trust domain declarations to bus descriptors for TopicResolver.

    Converts ``ModelTrustDomainConfigLocal`` instances into
    ``ModelBusDescriptor`` instances suitable for use with the
    ``TopicResolver`` (Phase 5). Each trust domain generates a bus
    descriptor with:
    - ``bus_id``: ``"bus.<domain_id>"``
    - ``trust_domain``: the domain_id
    - ``transport_type``: KAFKA (default)
    - ``namespace_prefix``: ``"<domain_id>."`` for non-local domains,
      empty string for ``local.*`` domains.

    Args:
        trust_domains: List of trust domain configurations.

    Returns:
        List of bus descriptors. One per trust domain.
    """
    descriptors: list[ModelBusDescriptor] = []

    for domain in trust_domains:
        # Local domains get no namespace prefix; others get domain_id as prefix.
        is_local = domain.domain_id.startswith("local.")
        namespace_prefix = "" if is_local else f"{domain.domain_id}."

        descriptor = ModelBusDescriptor(
            bus_id=f"bus.{domain.domain_id}",
            trust_domain=domain.domain_id,
            transport_type=EnumInfraTransportType.KAFKA,
            namespace_prefix=namespace_prefix,
        )
        descriptors.append(descriptor)

    return descriptors


def load_tiered_resolution_from_contract(
    contract_path: Path,
) -> ModelTieredResolutionContractResult:
    """Load all tiered resolution and trust domain config from a contract.

    This is the primary entry point for Phase 7 contract YAML integration.
    It loads both the per-dependency ``tiered_resolution`` configurations
    and the top-level ``trust_domains`` declarations, then bridges the
    trust domains into bus descriptors for the TopicResolver.

    Args:
        contract_path: Path to the contract.yaml file.

    Returns:
        ``ModelTieredResolutionContractResult`` containing:
        - tiered_configs: per-dependency resolution constraints
        - trust_domains: trust domain declarations
        - bus_descriptors: bridged bus descriptors for TopicResolver

    Raises:
        ProtocolConfigurationError: If the contract file is invalid or
            contains invalid configuration values.
    """
    tiered_configs = load_tiered_resolution_configs(contract_path)
    trust_domains = load_trust_domain_configs(contract_path)
    bus_descriptors = bridge_trust_domains_to_bus_descriptors(trust_domains)

    logger.info(
        "Loaded tiered resolution contract config from %s: "
        "%d dependency configs, %d trust domains, %d bus descriptors",
        contract_path,
        len(tiered_configs),
        len(trust_domains),
        len(bus_descriptors),
    )

    return ModelTieredResolutionContractResult(
        tiered_configs=tiered_configs,
        trust_domains=trust_domains,
        bus_descriptors=bus_descriptors,
    )


__all__: list[str] = [
    "ERROR_CODE_CONTRACT_NOT_DICT",
    "ERROR_CODE_CONTRACT_NOT_FOUND",
    "ERROR_CODE_FILE_SIZE_EXCEEDED",
    "ERROR_CODE_INVALID_CLASSIFICATION",
    "ERROR_CODE_INVALID_PROOF_TYPE",
    "ERROR_CODE_INVALID_TIER",
    "ERROR_CODE_INVALID_TRUST_DOMAIN",
    "ERROR_CODE_TIER_RANGE_INVALID",
    "ERROR_CODE_YAML_PARSE_ERROR",
    "MAX_CONTRACT_FILE_SIZE_BYTES",
    "ModelTieredResolutionContractResult",
    "bridge_trust_domains_to_bus_descriptors",
    "load_tiered_resolution_configs",
    "load_tiered_resolution_from_contract",
    "load_trust_domain_configs",
]
