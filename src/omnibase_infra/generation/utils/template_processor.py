"""Template processing for ONEX node generation.

Reads template files and substitutes placeholders to generate node scaffolding.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

from .name_converter import NameConverter


class TemplateProcessor:
    """Process ONEX node templates with placeholder substitution."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize template processor.

        Args:
            templates_dir: Path to templates directory.
                          Defaults to {package}/generation/templates/
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent / "templates"

        self.templates_dir = Path(templates_dir)

        if not self.templates_dir.exists():
            raise OnexError(
                code=CoreErrorCode.INITIALIZATION_FAILED,
                message=f"Templates directory not found: {self.templates_dir}",
            )

    def load_template(self, node_type: str) -> str:
        """Load template content for specified node type.

        Args:
            node_type: Node type (effect, compute, reducer, orchestrator)

        Returns:
            Template content as string

        Raises:
            OnexError: If template file not found
        """
        node_type_upper = node_type.upper()
        template_path = self.templates_dir / f"{node_type_upper}_NODE_TEMPLATE.md"

        if not template_path.exists():
            raise OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message=f"Template not found for node type: {node_type}",
                details={"template_path": str(template_path)},
            )

        try:
            return template_path.read_text(encoding="utf-8")
        except Exception as e:
            raise OnexError(
                code=CoreErrorCode.PROCESSING_ERROR,
                message=f"Failed to read template: {template_path}",
                details={"error": str(e)},
            ) from e

    def substitute_placeholders(
        self,
        template_content: str,
        replacements: Dict[str, str],
    ) -> str:
        """Substitute placeholders in template content.

        Args:
            template_content: Template string with placeholders
            replacements: Dictionary of placeholder -> value mappings

        Returns:
            Template with placeholders replaced
        """
        result = template_content

        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def extract_code_blocks(self, template_content: str) -> Dict[str, List[str]]:
        """Extract code blocks from markdown template.

        Args:
            template_content: Markdown template with code blocks

        Returns:
            Dictionary mapping file types to code blocks
        """
        code_blocks = {}

        # Match markdown code blocks with language specifiers
        pattern = r"```(\w+)\n(.*?)\n```"
        matches = re.finditer(pattern, template_content, re.DOTALL)

        for match in matches:
            language = match.group(1)
            code = match.group(2)

            if language not in code_blocks:
                code_blocks[language] = []

            code_blocks[language].append(code)

        return code_blocks

    def generate_node_from_template(
        self,
        node_type: str,
        repository_name: str,
        domain: str,
        microservice_name: str,
        business_description: str = "",
        external_system: str = "",
    ) -> Dict[str, str]:
        """Generate node files from template.

        Args:
            node_type: Node type (effect, compute, reducer, orchestrator)
            repository_name: Repository name
            domain: Domain name
            microservice_name: Microservice name
            business_description: Business functionality description
            external_system: External system being integrated

        Returns:
            Dictionary mapping file paths to generated content

        Raises:
            OnexError: If template processing fails
        """
        # Load template
        template_content = self.load_template(node_type)

        # Generate replacements
        replacements = NameConverter.generate_placeholder_replacements(
            repository_name=repository_name,
            domain=domain,
            microservice_name=microservice_name,
            business_description=business_description,
            external_system=external_system,
        )

        # Substitute placeholders
        processed_content = self.substitute_placeholders(template_content, replacements)

        # Extract code blocks
        code_blocks = self.extract_code_blocks(processed_content)

        # Generate file mappings
        generated_files = {}

        # Python code blocks
        if "python" in code_blocks:
            domain_snake = NameConverter.to_snake_case(domain)
            microservice_snake = NameConverter.to_snake_case(microservice_name)

            base_path = f"src/{repository_name}/nodes/node_{domain_snake}_{microservice_snake}_{node_type}/v1_0_0"

            # First code block is typically the main node.py
            if len(code_blocks["python"]) > 0:
                generated_files[f"{base_path}/node.py"] = code_blocks["python"][0]

            # Subsequent blocks for models, configs, etc.
            for i, code in enumerate(code_blocks["python"][1:], start=1):
                generated_files[f"{base_path}/generated_block_{i}.py"] = code

        # YAML code blocks (contracts, configs)
        if "yaml" in code_blocks:
            for i, code in enumerate(code_blocks["yaml"]):
                generated_files[f"contracts/generated_contract_{i}.yaml"] = code

        return generated_files
