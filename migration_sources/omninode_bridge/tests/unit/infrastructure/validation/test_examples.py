#!/usr/bin/env python3
"""Unit tests for validation examples.

Tests cover:
- ModelExample1 using JsonbField helper
- ModelExample2 using manual Field with validator
- ModelExample3 using JsonbValidatedModel base class
- JSONB validation functionality
"""

from uuid import uuid4

import pytest

from omninode_bridge.infrastructure.validation.examples import (
    ModelExample1,
    ModelExample2,
    ModelExample3,
)


class TestValidationExamples:
    """Test suite for validation examples."""

    def test_model_example1_instantiation(self):
        """Test ModelExample1 instantiation with JsonbField helper."""
        model = ModelExample1(
            namespace="test.namespace",
            status="active",
            metadata={"key": "value"},
            config_data={"setting": "value"},
        )

        assert model.namespace == "test.namespace"
        assert model.status == "active"
        assert model.metadata == {"key": "value"}
        assert model.config_data == {"setting": "value"}
        assert model.id is not None  # UUID should be generated

    def test_model_example1_default_values(self):
        """Test ModelExample1 with default values."""
        model = ModelExample1(namespace="test.namespace", status="active")

        assert model.metadata == {}  # default_factory=dict
        assert model.config_data == {}  # default_factory=dict
        assert model.id is not None  # default_factory=uuid4

    def test_model_example1_jsonb_fields(self):
        """Test that JSONB fields are properly annotated."""
        model = ModelExample1(namespace="test.namespace", status="active")

        # Check that the model has the JSONB annotations
        # This is tested through the model's field definitions
        assert hasattr(model, "metadata")
        assert hasattr(model, "config_data")

        # Test that the fields accept dict values
        model.metadata = {"test": "data"}
        model.config_data = {"config": "value"}

        assert model.metadata == {"test": "data"}
        assert model.config_data == {"config": "value"}

    def test_model_example2_instantiation(self):
        """Test ModelExample2 instantiation with manual Field and validator."""
        model = ModelExample2(
            namespace="test.namespace",
            status="active",
            metadata={"key": "value"},
            config_data={"setting": "value"},
        )

        assert model.namespace == "test.namespace"
        assert model.status == "active"
        assert model.metadata == {"key": "value"}
        assert model.config_data == {"setting": "value"}
        assert model.id is not None

    def test_model_example2_default_values(self):
        """Test ModelExample2 with default values."""
        model = ModelExample2(namespace="test.namespace", status="active")

        assert model.metadata == {}  # default_factory=dict
        assert model.config_data == {}  # default_factory=dict
        assert model.id is not None  # default_factory=uuid4

    def test_model_example2_jsonb_validation(self):
        """Test that JSONB validation works for ModelExample2."""
        model = ModelExample2(namespace="test.namespace", status="active")

        # The _validate_jsonb_fields method should be called during initialization
        # If it's working correctly, no exception should be raised
        assert model is not None

    def test_model_example3_instantiation(self):
        """Test ModelExample3 instantiation with JsonbValidatedModel base class."""
        model = ModelExample3(
            namespace="test.namespace",
            status="active",
            metadata={"key": "value"},
            config_data={"setting": "value"},
        )

        assert model.namespace == "test.namespace"
        assert model.status == "active"
        assert model.metadata == {"key": "value"}
        assert model.config_data == {"setting": "value"}
        assert model.id is not None

    def test_model_example3_default_values(self):
        """Test ModelExample3 with default values."""
        model = ModelExample3(namespace="test.namespace", status="active")

        assert model.metadata == {}  # default_factory=dict
        assert model.config_data == {}  # default_factory=dict
        assert model.id is not None  # default_factory=uuid4

    def test_model_example3_automatic_validation(self):
        """Test that automatic JSONB validation works for ModelExample3."""
        model = ModelExample3(namespace="test.namespace", status="active")

        # The JsonbValidatedModel base class should automatically validate JSONB fields
        # If it's working correctly, no exception should be raised
        assert model is not None

    def test_all_models_have_required_fields(self):
        """Test that all models require the namespace field."""
        # Test that namespace is required for all models
        with pytest.raises(ValueError):
            ModelExample1()  # Missing required namespace field

        with pytest.raises(ValueError):
            ModelExample2()  # Missing required namespace field

        with pytest.raises(ValueError):
            ModelExample3()  # Missing required namespace field

    def test_all_models_accept_valid_namespace(self):
        """Test that all models accept valid namespace values."""
        valid_namespaces = [
            "test.namespace",
            "production",
            "dev",
            "omninode.bridge",
            "a" * 255,  # Maximum length
        ]

        for namespace in valid_namespaces:
            model1 = ModelExample1(namespace=namespace, status="active")
            model2 = ModelExample2(namespace=namespace, status="active")
            model3 = ModelExample3(namespace=namespace, status="active")

            assert model1.namespace == namespace
            assert model2.namespace == namespace
            assert model3.namespace == namespace

    def test_all_models_reject_invalid_namespace(self):
        """Test that all models reject invalid namespace values."""
        invalid_namespaces = [
            "",  # Empty string
            "a" * 256,  # Too long
        ]

        for namespace in invalid_namespaces:
            with pytest.raises(ValueError):
                ModelExample1(namespace=namespace, status="active")

            with pytest.raises(ValueError):
                ModelExample2(namespace=namespace, status="active")

            with pytest.raises(ValueError):
                ModelExample3(namespace=namespace, status="active")

    def test_jsonb_field_serialization(self):
        """Test that JSONB fields can be serialized and deserialized."""
        test_data = {
            "metadata": {"key1": "value1", "key2": 123},
            "config_data": {"setting1": True, "setting2": [1, 2, 3]},
        }

        # Test with ModelExample1
        model1 = ModelExample1(namespace="test", status="active", **test_data)
        data1 = model1.model_dump()
        assert data1["metadata"] == test_data["metadata"]
        assert data1["config_data"] == test_data["config_data"]

        # Test with ModelExample2
        model2 = ModelExample2(namespace="test", status="active", **test_data)
        data2 = model2.model_dump()
        assert data2["metadata"] == test_data["metadata"]
        assert data2["config_data"] == test_data["config_data"]

        # Test with ModelExample3
        model3 = ModelExample3(namespace="test", status="active", **test_data)
        data3 = model3.model_dump()
        assert data3["metadata"] == test_data["metadata"]
        assert data3["config_data"] == test_data["config_data"]

    def test_jsonb_field_update(self):
        """Test that JSONB fields can be updated."""
        model = ModelExample1(namespace="test", status="active")

        # Update JSONB fields
        model.metadata = {"new": "data"}
        model.config_data = {"new_config": True}

        assert model.metadata == {"new": "data"}
        assert model.config_data == {"new_config": True}

    def test_model_equality(self):
        """Test model equality comparison."""
        # Use a fixed UUID for testing equality
        test_id = uuid4()
        model1a = ModelExample1(
            namespace="test", status="active", metadata={"key": "value"}, id=test_id
        )
        model1b = ModelExample1(
            namespace="test", status="active", metadata={"key": "value"}, id=test_id
        )
        model1c = ModelExample1(
            namespace="test", status="active", metadata={"key": "different"}, id=test_id
        )

        # Models with same data should be equal
        assert model1a.model_dump() == model1b.model_dump()

        # Models with different data should not be equal
        assert model1a.model_dump() != model1c.model_dump()

    def test_model_copy(self):
        """Test model copying."""
        original = ModelExample1(
            namespace="test",
            status="active",
            metadata={"original": "data"},
            config_data={"original": "config"},
        )

        # Copy model
        copied = original.model_copy()

        # Verify copy has same data
        assert copied.model_dump() == original.model_dump()

        # Verify it's a different instance
        assert copied is not original

        # Verify modifications to copy don't affect original
        copied.metadata = {"modified": "data"}
        assert original.metadata == {"original": "data"}
        assert copied.metadata == {"modified": "data"}
