#!/usr/bin/env python3
"""
Mock LLM responses and generated code samples.

Provides realistic mock responses for testing without actual LLM API calls.
"""


# Mock LLM response for simple business logic
MOCK_LLM_RESPONSE_SIMPLE = """# Execute CRUD operation
logger.info(f"Executing {operation} operation for user {user_id}")

# Perform the operation
result_data = {"status": "success", "operation": operation}

# Return result
return ModelContainer(
    data=result_data,
    metadata={"execution_time": 0.05}
)"""


# Mock LLM response for moderate complexity logic
MOCK_LLM_RESPONSE_MODERATE = """# Transform data according to rules
logger.info(f"Transforming {len(data)} items")

# Apply transformation rules
transformed = []
for item in data:
    # Apply rule-based transformation
    transformed_item = self._apply_rules(item, transformation_rules)
    transformed.append(transformed_item)

# Validate transformed data
validation_results = self._validate_transformed_data(transformed)

# Return result
return ModelContainer(
    data={
        "transformed_data": transformed,
        "validation_results": validation_results,
        "items_processed": len(transformed)
    },
    metadata={"execution_time": 0.2}
)"""


# Mock LLM response for complex orchestration
MOCK_LLM_RESPONSE_COMPLEX = """# Orchestrate multi-step payment processing
logger.info(f"Orchestrating payment for transaction {payment_data.get('transaction_id')}")

# Step 1: Validate transaction
validation_result = await self._validate_transaction(payment_data)
if not validation_result["valid"]:
    raise ModelOnexError(
        error_code=EnumCoreErrorCode.VALIDATION_ERROR,
        message="Transaction validation failed"
    )

# Step 2: Assess risk
risk_score = await self._assess_risk(payment_data, customer_profile)
logger.info(f"Risk score calculated: {risk_score}")

# Step 3: Route payment based on risk
routing_decision = self._determine_routing(risk_score, merchant_config)

# Step 4: Process payment
payment_result = await self._process_payment(
    payment_data,
    routing_decision
)

# Return orchestration result
return ModelContainer(
    data={
        "transaction_id": payment_result["transaction_id"],
        "status": payment_result["status"],
        "routing_decisions": [routing_decision],
        "risk_score": risk_score
    },
    metadata={
        "execution_time": 0.8,
        "steps_executed": 4
    }
)"""


# Sample valid generated code (no stubs)
SAMPLE_VALID_CODE = '''#!/usr/bin/env python3
"""
Generated Effect Node.

Author: CodeGenerationService
Version: v1_0_0
"""

import logging
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

logger = logging.getLogger(__name__)


class NodeTestServiceEffect:
    """Test service effect node."""

    def __init__(self):
        """Initialize the effect node."""
        logger.info("NodeTestServiceEffect initialized")

    async def execute_effect(self, contract: ModelContractEffect) -> ModelContainer:
        """
        Execute the effect operation.

        Args:
            contract: Effect contract with input data

        Returns:
            ModelContainer with result data
        """
        logger.info(f"Executing effect for {contract.node_id}")

        # Process the input data
        result_data = {"status": "success", "processed": True}

        # Return result
        return ModelContainer(
            data=result_data,
            metadata={"execution_time": 0.1}
        )
'''


# Sample code with stubs (for injection testing)
SAMPLE_CODE_WITH_STUBS = '''#!/usr/bin/env python3
"""
Generated Effect Node with Stubs.

Author: CodeGenerationService
Version: v1_0_0
"""

import logging
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer

logger = logging.getLogger(__name__)


class NodeTestServiceEffect:
    """Test service effect node."""

    def __init__(self):
        """Initialize the effect node."""
        logger.info("NodeTestServiceEffect initialized")

    async def execute_effect(self, contract: ModelContractEffect) -> ModelContainer:
        """
        Execute the effect operation.

        This method requires implementation.
        """
        # IMPLEMENTATION REQUIRED
        pass

    async def validate_input(self, data: dict) -> bool:
        """
        Validate input data.

        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement validation logic
        pass
'''


# Sample code with syntax errors
SAMPLE_CODE_WITH_SYNTAX_ERROR = '''#!/usr/bin/env python3
"""
Generated Effect Node with Syntax Error.
"""

import logging

logger = logging.getLogger(__name__)


class NodeTestServiceEffect:
    """Test service effect node."""

    def __init__(self)
        """Missing colon causes syntax error."""
        logger.info("Initialized")
'''


# Sample code with security issues
SAMPLE_CODE_WITH_SECURITY_ISSUES = '''#!/usr/bin/env python3
"""
Generated Effect Node with Security Issues.
"""

import logging
import pickle  # Dangerous import

logger = logging.getLogger(__name__)

# Hardcoded secret (security issue)
API_KEY = "sk_test_12345abcdefg"  # pragma: allowlist secret


class NodeTestServiceEffect:
    """Test service effect node."""

    def __init__(self):
        """Initialize with hardcoded credentials."""
        self.db_password = "admin123"  # Security issue
        logger.info("Initialized")

    async def execute_effect(self, contract):
        """Execute with dangerous pattern."""
        # SQL injection vulnerability
        query = f"SELECT * FROM users WHERE id = {contract.input_data['user_id']}"

        # Dangerous eval usage
        result = eval(contract.input_data['code'])  # Security issue

        return result
'''
