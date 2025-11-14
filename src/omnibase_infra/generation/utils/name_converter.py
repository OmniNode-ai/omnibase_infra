"""Name conversion utilities for ONEX node generation.

Converts between different naming conventions:
- snake_case: my_service_name
- PascalCase: MyServiceName
- kebab-case: my-service-name
- SCREAMING_SNAKE_CASE: MY_SERVICE_NAME
"""

import re
from typing import Dict


class NameConverter:
    """Convert names between different conventions for code generation."""

    @staticmethod
    def to_snake_case(name: str) -> str:
        """Convert name to snake_case.

        Examples:
            MyServiceName -> my_service_name
            my-service-name -> my_service_name
            MY_SERVICE_NAME -> my_service_name
        """
        # Handle PascalCase and camelCase
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        # Handle kebab-case
        name = name.replace('-', '_')
        # Lowercase
        return name.lower()

    @staticmethod
    def to_pascal_case(name: str) -> str:
        """Convert name to PascalCase.

        Examples:
            my_service_name -> MyServiceName
            my-service-name -> MyServiceName
            MY_SERVICE_NAME -> MyServiceName
        """
        # Replace separators with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        # Title case and remove spaces
        return ''.join(word.capitalize() for word in name.split())

    @staticmethod
    def to_kebab_case(name: str) -> str:
        """Convert name to kebab-case.

        Examples:
            MyServiceName -> my-service-name
            my_service_name -> my-service-name
            MY_SERVICE_NAME -> my-service-name
        """
        # Convert to snake_case first
        snake = NameConverter.to_snake_case(name)
        # Replace underscores with hyphens
        return snake.replace('_', '-')

    @staticmethod
    def to_screaming_snake_case(name: str) -> str:
        """Convert name to SCREAMING_SNAKE_CASE.

        Examples:
            MyServiceName -> MY_SERVICE_NAME
            my-service-name -> MY_SERVICE_NAME
            my_service_name -> MY_SERVICE_NAME
        """
        return NameConverter.to_snake_case(name).upper()

    @staticmethod
    def generate_placeholder_replacements(
        repository_name: str,
        domain: str,
        microservice_name: str,
        business_description: str = "",
        external_system: str = "",
    ) -> Dict[str, str]:
        """Generate all placeholder replacements for templates.

        Args:
            repository_name: Repository name (e.g., "omnibase_infra")
            domain: Domain name (e.g., "infrastructure", "ai")
            microservice_name: Microservice name (e.g., "postgres_adapter")
            business_description: Description of business functionality
            external_system: External system being integrated

        Returns:
            Dictionary of placeholder replacements
        """
        return {
            # Repository
            "{REPOSITORY_NAME}": repository_name,

            # Domain
            "{DOMAIN}": NameConverter.to_snake_case(domain),
            "{DOMAIN_PASCAL}": NameConverter.to_pascal_case(domain),
            "{DomainCamelCase}": NameConverter.to_pascal_case(domain),

            # Microservice
            "{MICROSERVICE_NAME}": NameConverter.to_snake_case(microservice_name),
            "{MICROSERVICE_NAME_PASCAL}": NameConverter.to_pascal_case(microservice_name),
            "{MicroserviceCamelCase}": NameConverter.to_pascal_case(microservice_name),

            # Business context
            "{BUSINESS_DESCRIPTION}": business_description,
            "{EXTERNAL_SYSTEM}": external_system,

            # Performance targets (can be customized)
            "{PERFORMANCE_TARGET}": "100",
        }
