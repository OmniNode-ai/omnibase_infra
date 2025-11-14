"""ONEX Node Generator - Main orchestration for node scaffolding generation.

This module provides high-level interface for generating ONEX nodes from templates.
"""

from pathlib import Path
from typing import Dict, List, Optional

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

from .utils.file_writer import FileWriter
from .utils.name_converter import NameConverter
from .utils.template_processor import TemplateProcessor


class NodeGenerator:
    """Generate ONEX node scaffolding from templates.

    This is the main entry point for the generation pipeline.

    Usage:
        generator = NodeGenerator(output_dir="/path/to/output")

        generator.generate_node(
            node_type="effect",
            repository_name="omnibase_infra",
            domain="infrastructure",
            microservice_name="postgres_adapter",
            business_description="PostgreSQL database adapter",
            external_system="PostgreSQL",
        )
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        templates_dir: Optional[Path] = None,
    ):
        """Initialize node generator.

        Args:
            output_dir: Base directory for generated files.
                       Defaults to current working directory.
            templates_dir: Path to templates directory.
                          Defaults to package templates directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.template_processor = TemplateProcessor(templates_dir)
        self.file_writer = FileWriter(self.output_dir)

    def generate_node(
        self,
        node_type: str,
        repository_name: str,
        domain: str,
        microservice_name: str,
        business_description: str = "",
        external_system: str = "",
        dry_run: bool = False,
    ) -> Dict[str, Path]:
        """Generate complete node scaffolding.

        Args:
            node_type: Node type (effect, compute, reducer, orchestrator)
            repository_name: Repository name (e.g., "omnibase_infra")
            domain: Domain name (e.g., "infrastructure")
            microservice_name: Microservice name (e.g., "postgres_adapter")
            business_description: Description of business functionality
            external_system: External system being integrated
            dry_run: If True, log what would be generated without writing

        Returns:
            Dictionary mapping description to Path objects

        Raises:
            OnexError: If generation fails
        """
        # Validate node type
        valid_types = ["effect", "compute", "reducer", "orchestrator"]
        if node_type.lower() not in valid_types:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Invalid node type: {node_type}",
                details={"valid_types": valid_types},
            )

        # Generate file contents from template
        generated_files = self.template_processor.generate_node_from_template(
            node_type=node_type.lower(),
            repository_name=repository_name,
            domain=domain,
            microservice_name=microservice_name,
            business_description=business_description,
            external_system=external_system,
        )

        # Write files to filesystem
        written_paths = self.file_writer.write_files(
            file_contents=generated_files,
            dry_run=dry_run,
        )

        # Create directory structure with __init__.py files
        domain_snake = NameConverter.to_snake_case(domain)
        microservice_snake = NameConverter.to_snake_case(microservice_name)

        node_dir = (
            self.output_dir
            / "src"
            / repository_name
            / "nodes"
            / f"node_{domain_snake}_{microservice_snake}_{node_type}"
            / "v1_0_0"
        )

        directories_to_init = [
            node_dir,
            node_dir / "models",
            node_dir / "enums",
            node_dir / "contracts",
        ]

        init_paths = self.file_writer.create_init_files(
            directories=directories_to_init,
            dry_run=dry_run,
        )

        # Return summary
        return {
            "node_directory": node_dir,
            "generated_files": written_paths,
            "init_files": init_paths,
        }

    def generate_from_contract(
        self,
        contract_path: Path,
        dry_run: bool = False,
    ) -> Dict[str, Path]:
        """Generate node from existing contract.yaml file.

        Args:
            contract_path: Path to contract.yaml
            dry_run: If True, log without writing

        Returns:
            Dictionary of generated paths

        Raises:
            OnexError: If generation from contract fails
        """
        import yaml

        try:
            contract_data = yaml.safe_load(contract_path.read_text())
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.PROCESSING_ERROR,
                message=f"Failed to load contract: {contract_path}",
                details={"error": str(e)},
            ) from e

        # Extract metadata from contract
        node_type = contract_data.get("node_type", "effect")
        node_name = contract_data.get("node_name", "")
        description = contract_data.get("description", "")

        # Parse node name to extract domain and microservice
        # Expected format: node_{domain}_{microservice}_{type}
        parts = node_name.split("_")
        if len(parts) < 3:
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Invalid node name format: {node_name}",
                details={"expected_format": "node_{domain}_{microservice}_{type}"},
            )

        domain = parts[1] if len(parts) > 1 else "unknown"
        microservice = "_".join(parts[2:-1]) if len(parts) > 3 else parts[2]

        # Determine repository name from contract path
        repository_name = "omnibase_infra"  # Default

        # Generate node
        return self.generate_node(
            node_type=node_type,
            repository_name=repository_name,
            domain=domain,
            microservice_name=microservice,
            business_description=description,
            dry_run=dry_run,
        )

    def list_available_templates(self) -> List[str]:
        """List all available node templates.

        Returns:
            List of template names
        """
        template_files = list(self.template_processor.templates_dir.glob("*_NODE_TEMPLATE.md"))

        return [
            template.stem.replace("_NODE_TEMPLATE", "").lower()
            for template in template_files
        ]
