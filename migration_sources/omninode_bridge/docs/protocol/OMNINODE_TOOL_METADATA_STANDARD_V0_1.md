# OmniNode Metadata Stamper — Extensible Baseline (v0.1)

**See the practical implementation and usage guide:** [../test_and_script_metadata_standard.md](../test_and_script_metadata_standard.md)

This document defines the **metadata header format** for OmniNode tools, agents, and components. This version incorporates critical refinements for validator behavior, runtime policy enforcement, dynamic registry integration, and cross-platform test compatibility.

## 📦 Purpose

This metadata spec defines the universal contract for:
- Validating and parsing tools used in the OmniNode ecosystem
- Enabling registry-based discovery and execution governance
- Supporting federation, trust scores, lifecycle tracking, and reproducible environments
- Providing cross-language compatibility and agent composability

## 🧱 Structure

The metadata block must appear at the **top** of the main file and be enclosed in language-specific comment delimiters. For Python:

    # === OmniNode:Tool_Metadata ===
    ...
    # === /OmniNode:Tool_Metadata ===

Each line must follow a `key: value` format. Multiline fields use `|`. All values must be statically parseable (no dynamic references or functions).

## ✅ Example

    # === OmniNode:Tool_Metadata ===
    # metadata_version: 0.1
    # name: github_sniper_agent
    # title: GitHub Sniper Agent
    # version: 0.1.2
    # status: beta
    # autoupdate: true
    # protocols_supported: [O.N.E. v0.1]
    # namespace: omninode.agents.discovery
    # category: agent.discovery.monitoring
    # type: agent
    # role: discovery
    # description: |
    #   Monitors GitHub for new frameworks or tools and evaluates their utility
    #   for integration into the OmniNode ecosystem.
    # tags: [agent, github, discovery, auto-eval]
    # author: Jonah Gray
    # authors: [Jonah Gray, Alice Lin]
    # contact: jonah@omninode.dev
    # license: MIT
    # license_url: https://opensource.org/licenses/MIT
    # docs_url: https://docs.omninode.dev/github-sniper
    # source_url: https://github.com/omninode/github-sniper-agent
    # registry_url: https://registry.omninode.dev/github-sniper-agent
    # entrypoint: main.py
    # message_bus: redis_streams
    # dependencies: [{"name": "docker", "version": ">=20.10"}, {"name": "gitpython", "version": "^3.1"}]
    # environment: [python>=3.11, ubuntu-22.04]
    # runtime_constraints: {sandboxed: true, privileged: false, requires_network: true, requires_gpu: false}
    # test_suite: true
    # test_status: passing
    # coverage: 91.2
    # ci_url: https://ci.omninode.dev/github-sniper/status
    # classification: {maturity: beta, trust_score: 87}
    # related_tools: [docker_scanner_agent, git_mirror_proxy]
    # deprecated_since: null
    # replacement: null
    # sunset_date: null
    # encryption: {encrypted_fields: [], encryption_alg: none}
    # telemetry: {logging: true, metrics: true}
    # signature_alg: ed25519
    # signature_format: hex
    # signature: 4a5b1c57... (truncated)
    # signed_by: omninode:jonah
    # extends: base_agent_profile
    # dynamic_fields: [trust_score, last_registry_sync, registry_classification]
    # policy:
    #   network_access: true
    #   allowed_endpoints:
    #     - github.com
    #     - pypi.org
    #   resource_limits:
    #     memory: "512mb"
    #     threads: 2
    #   compliance_requirements:
    #     - gdpr
    #     - iso27001
    # platform_matrix:
    #   - os: linux
    #     arch: amd64
    #     python: "3.11"
    #   - os: macos
    #     arch: arm64
    #     python: "3.12"
    # last_updated: 2025-04-23
    # === /OmniNode:Tool_Metadata ===

## 🔑 Required Keys (v0.1 Final)

    required_keys:
      - metadata_version
      - name
      - namespace
      - version
      - entrypoint
      - protocols_supported

## 🚨 Addendum Refinements

- **Mandatory `metadata_version`** for compatibility routing
- **`dynamic_fields`** declaration for registry-owned keys
- **Standardized `policy` schema** for runtime safety and governance
- **`platform_matrix`** for CI/test/deploy alignment
- Replaces scattered permissions with structured, extensible controls

## 🔍 Parsing and Validation

Metadata blocks should be parsed as YAML after removing line-level comment syntax. Every file must pass validation using the latest version of the spec it declares via `metadata_version`.

Example:

    extract_block(file_text):
        lines = file_text.split("\n")
        inside = False
        block = []
        for line in lines:
            if line.strip() == "# === OmniNode:Tool_Metadata ===":
                inside = True
                continue
            if line.strip() == "# === /OmniNode:Tool_Metadata ===":
                break
            if inside:
                block.append(line.lstrip("# ").rstrip())
        return parse_yaml("\n".join(block))

## 🛠 Linting & Tooling

- Planned CLI tool: `omninode-meta-lint`
- Validates metadata blocks for compliance
- Enforces required keys and type constraints
- Can be run as a pre-commit hook or CI validator
- Will support schema autogeneration and key autocompletion

## 🌐 Language Support

| Language | Start Delimiter                        | End Delimiter                          |
|----------|----------------------------------------|----------------------------------------|
| Python   | `# === OmniNode:Tool_Metadata ===`     | `# === /OmniNode:Tool_Metadata ===`    |
| JS/TS    | `/* === OmniNode:Tool_Metadata ===`    | `=== /OmniNode:Tool_Metadata === */`   |
| Rust     | `// === OmniNode:Tool_Metadata ===`    | `// === /OmniNode:Tool_Metadata ===`   |
| Go       | `// === OmniNode:Tool_Metadata ===`    | `// === /OmniNode:Tool_Metadata ===`   |

## 🚀 Future-Proofing and Federation

This metadata format is designed to:
- Support **real-time validation and registration**
- Enable **agent trust scoring and classification**
- Allow **decentralized discovery** via signed manifests
- Plug into **PRISM, CAIA, and registry federation layers**

Future additions will include:
- Secure metadata diffing
- Manifest bundle signing
- Public trust graph integration
- Live registry sync hooks

## 🚀 Possibilities and Future Directions

**1. Digital Signatures & Provenance**
- Add an (optional) `signature` or `signed_by` field, supporting cryptographic signatures to verify authorship and integrity.
- Enables trust, provenance, and tamper detection, especially for marketplace or federated agent distribution.

**2. Machine/Agent Readable Tags**
- Standardize a `type` or `role` field (e.g., `type: template`, `type: agent`, `type: validator`) for automated filtering, validation, and discovery.
- Simplifies exclusion of templates, examples, or test files from runtime or validation workflows.

**3. Protocol Versioning Granularity**
- Allow multiple protocol versions or compatibility ranges (e.g., `protocols_supported: [O.N.E. v0.1, O.N.E. v0.2]`).
- Smooths migration as the protocol evolves.
- Add a `metadata_version` field to the block itself for future-proofing and migration tooling.

**4. Security & Trust**
- Standardize signature algorithm and encoding (e.g., `signature_alg: ed25519`, `signature_format: hex`).
- Enables automated signature verification and cross-tool trust.

**5. Policy & Permissions**
- Add a `permissions` or `policy` field for runtime access controls (e.g., allowed APIs, network access, storage).
- Enables fine-grained policy enforcement at runtime.
- Expand `policy` to support structured runtime policies, e.g., allowed endpoints, resource quotas, compliance requirements.

**6. Localization & Documentation**
- Add `locales` or `translations` field for multi-language descriptions, docs, and UI.
- Supports internationalization and broader adoption.
- Allow `description_locales` or similar for multi-language documentation.

**7. Lifecycle & Deprecation**
- Add `deprecated_since`, `replacement`, or `sunset_date` fields for lifecycle management.
- Makes it easier to automate sunset and migration flows.

**8. Testing & Validation**
- Add `test_status`, `coverage`, or `ci_url` fields for test/validation status.
- Surfaces code quality and integration health in registries.

**9. Agent/Tool Relationships**
- Add `related_tools`, `parent`, or `children` fields for composability and dependency graphs.
- Enables richer dependency and relationship mapping.

**10. Manifest/Bundle Support**
- Allow a single metadata block to describe a bundle of tools/agents (e.g., `bundle: true`, `components: [...]`).
- Useful for multi-agent systems or packaged workflows.

**11. Runtime/Platform Compatibility**
- Add `platforms_supported` (e.g., `linux/amd64`, `macos/arm64`) for clear compatibility.
- Aids in deployment targeting and registry filtering.

**12. Observability & Telemetry**
- Add `telemetry`, `logging`, or `metrics` fields for observability hooks.
- Enables automated integration with monitoring and analytics tools.
- Expand `telemetry` to include log/metric endpoints, e.g., `telemetry: {logging: true, metrics: true, endpoint: ...}`.

**13. Metadata Validation & Linting**
- Develop a CLI tool or pre-commit hook that validates metadata presence, structure, and required keys.
- Ensures adoption and correctness across the codebase.

**14. Metadata Inheritance or Extension**
- Allow metadata blocks to reference or inherit from shared profiles (e.g., `extends: base_agent_profile`).
- Reduces duplication for families of agents/tools.

**15. Metadata Block Format for Other Languages**
- Define equivalent block delimiters for other languages (e.g., `""" ... """` for Python docstrings, `/* ... */` for JS).
- Enables multi-language agent/tool support.

**16. OmniNode Registry Integration**
- Specify a `registry_url` or `discovery_endpoint` for agent auto-registration or discovery.
- Streamlines onboarding into federated or enterprise registries.
- Add a `federation` or `trust_zones` field to specify which registries or trust zones the tool is authorized to sync with.

**17. Bundles & Components**
- Add a `bundle: true` or `components: [...]` field for describing multi-tool/agent bundles, supporting monorepo or packaged agent systems.

**18. Dynamic Metadata Sync**
- Optionally allow for a `sync_fields` or `dynamic_fields` section, indicating which metadata fields may be updated by registries (e.g., trust score, last_registry_sync).

**19. Backward Compatibility & Migration**
- Define a clear migration path for v0.1/v0.2 blocks, possibly with a `migrated_from` field for traceability.
