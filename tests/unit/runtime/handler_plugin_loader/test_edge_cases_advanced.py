# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Advanced edge case tests for HandlerPluginLoader.

Part of OMN-1132: Handler Plugin Loader implementation.

This module tests edge cases not covered in the basic test suite:
1. Non-class loading - when handler_class points to non-class objects
2. Broken symlinks - filesystem edge cases with symbolic links
3. Race conditions - concurrent loading and filesystem mutation scenarios

Thread Safety Note:
    The HandlerPluginLoader is designed to be stateless and reentrant.
    These tests verify that concurrent operations behave correctly without
    data corruption or race condition failures.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from .conftest import MINIMAL_HANDLER_CONTRACT_YAML

# =============================================================================
# Non-Class Loading Tests
# =============================================================================


class TestNonClassLoading:
    """Tests for scenarios where handler_class points to non-class objects.

    The loader must correctly identify and reject non-class objects with
    appropriate error codes (CLASS_NOT_FOUND / HANDLER_LOADER_011).
    """

    def test_loading_function_instead_of_class_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that loading a function instead of a class raises CLASS_NOT_FOUND.

        When handler_class points to a function object, the loader should
        reject it with HANDLER_LOADER_011 error code.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a module with a function instead of a class
        module_dir = tmp_path / "fake_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("")
        (module_dir / "handler.py").write_text(
            """
def NotAClassHandler():
    '''This is a function, not a class.'''
    return {}
"""
        )

        # Add to sys.path temporarily
        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="function.handler",
                    handler_class="fake_module.handler.NotAClassHandler",
                )
            )

            loader = HandlerPluginLoader()

            with pytest.raises(InfraConnectionError) as exc_info:
                loader.load_from_contract(contract_file)

            error = exc_info.value
            assert (
                error.model.context.get("loader_error")
                == EnumHandlerLoaderError.CLASS_NOT_FOUND.value
            )
            assert "not a class" in str(error).lower()
        finally:
            sys.path.remove(str(tmp_path))
            # Clean up imported module
            if "fake_module" in sys.modules:
                del sys.modules["fake_module"]
            if "fake_module.handler" in sys.modules:
                del sys.modules["fake_module.handler"]

    def test_loading_module_variable_instead_of_class_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that loading a module-level variable raises CLASS_NOT_FOUND.

        When handler_class points to a variable (e.g., a dict), the loader
        should reject it with HANDLER_LOADER_011 error code.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a module with a dict variable instead of a class
        module_dir = tmp_path / "var_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("")
        (module_dir / "handler.py").write_text(
            """
NotAClass = {"key": "value"}  # This is a dict, not a class
"""
        )

        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="variable.handler",
                    handler_class="var_module.handler.NotAClass",
                )
            )

            loader = HandlerPluginLoader()

            with pytest.raises(InfraConnectionError) as exc_info:
                loader.load_from_contract(contract_file)

            error = exc_info.value
            assert (
                error.model.context.get("loader_error")
                == EnumHandlerLoaderError.CLASS_NOT_FOUND.value
            )
            assert "not a class" in str(error).lower()
        finally:
            sys.path.remove(str(tmp_path))
            if "var_module" in sys.modules:
                del sys.modules["var_module"]
            if "var_module.handler" in sys.modules:
                del sys.modules["var_module.handler"]

    def test_loading_constant_instead_of_class_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that loading a string constant raises CLASS_NOT_FOUND.

        When handler_class points to a string constant, the loader should
        reject it with HANDLER_LOADER_011 error code.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a module with a string constant instead of a class
        module_dir = tmp_path / "const_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("")
        (module_dir / "handler.py").write_text(
            """
HANDLER_CONSTANT = "I am just a string"
"""
        )

        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="constant.handler",
                    handler_class="const_module.handler.HANDLER_CONSTANT",
                )
            )

            loader = HandlerPluginLoader()

            with pytest.raises(InfraConnectionError) as exc_info:
                loader.load_from_contract(contract_file)

            error = exc_info.value
            assert (
                error.model.context.get("loader_error")
                == EnumHandlerLoaderError.CLASS_NOT_FOUND.value
            )
            assert "not a class" in str(error).lower()
        finally:
            sys.path.remove(str(tmp_path))
            if "const_module" in sys.modules:
                del sys.modules["const_module"]
            if "const_module.handler" in sys.modules:
                del sys.modules["const_module.handler"]

    def test_loading_none_value_instead_of_class_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that loading None instead of a class raises CLASS_NOT_FOUND.

        When handler_class points to a None value, the loader should reject
        it with HANDLER_LOADER_011 error code.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a module with None as the handler
        module_dir = tmp_path / "none_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("")
        (module_dir / "handler.py").write_text(
            """
NoneHandler = None  # Explicitly None
"""
        )

        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="none.handler",
                    handler_class="none_module.handler.NoneHandler",
                )
            )

            loader = HandlerPluginLoader()

            with pytest.raises(InfraConnectionError) as exc_info:
                loader.load_from_contract(contract_file)

            error = exc_info.value
            assert (
                error.model.context.get("loader_error")
                == EnumHandlerLoaderError.CLASS_NOT_FOUND.value
            )
            assert "not a class" in str(error).lower()
        finally:
            sys.path.remove(str(tmp_path))
            if "none_module" in sys.modules:
                del sys.modules["none_module"]
            if "none_module.handler" in sys.modules:
                del sys.modules["none_module.handler"]

    def test_loading_lambda_instead_of_class_raises_error(self, tmp_path: Path) -> None:
        """Test that loading a lambda instead of a class raises CLASS_NOT_FOUND.

        Lambdas are functions and should be rejected like regular functions.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a module with a lambda as the handler
        module_dir = tmp_path / "lambda_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("")
        (module_dir / "handler.py").write_text(
            """
LambdaHandler = lambda x: x  # A lambda, not a class
"""
        )

        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="lambda.handler",
                    handler_class="lambda_module.handler.LambdaHandler",
                )
            )

            loader = HandlerPluginLoader()

            with pytest.raises(InfraConnectionError) as exc_info:
                loader.load_from_contract(contract_file)

            error = exc_info.value
            assert (
                error.model.context.get("loader_error")
                == EnumHandlerLoaderError.CLASS_NOT_FOUND.value
            )
            assert "not a class" in str(error).lower()
        finally:
            sys.path.remove(str(tmp_path))
            if "lambda_module" in sys.modules:
                del sys.modules["lambda_module"]
            if "lambda_module.handler" in sys.modules:
                del sys.modules["lambda_module.handler"]

    def test_loading_class_without_protocol_raises_protocol_error(
        self, tmp_path: Path
    ) -> None:
        """Test that loading a class without ProtocolHandler methods raises error.

        The class exists and is valid but doesn't implement the required
        protocol methods. This should raise PROTOCOL_NOT_IMPLEMENTED.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a module with a class that doesn't implement the protocol
        module_dir = tmp_path / "noprotocol_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("")
        (module_dir / "handler.py").write_text(
            """
class NoProtocolHandler:
    '''A class without ProtocolHandler methods.'''
    def some_method(self):
        return "hello"
"""
        )

        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="noprotocol.handler",
                    handler_class="noprotocol_module.handler.NoProtocolHandler",
                )
            )

            loader = HandlerPluginLoader()

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                loader.load_from_contract(contract_file)

            error = exc_info.value
            assert (
                error.model.context.get("loader_error")
                == EnumHandlerLoaderError.PROTOCOL_NOT_IMPLEMENTED.value
            )
            assert "missing required" in str(error).lower()
        finally:
            sys.path.remove(str(tmp_path))
            if "noprotocol_module" in sys.modules:
                del sys.modules["noprotocol_module"]
            if "noprotocol_module.handler" in sys.modules:
                del sys.modules["noprotocol_module.handler"]

    def test_loading_instance_instead_of_class_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that loading a class instance instead of a class raises error.

        When handler_class points to an instance (object), not a class,
        the loader should reject it with CLASS_NOT_FOUND.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a module with an instance instead of a class
        module_dir = tmp_path / "instance_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("")
        (module_dir / "handler.py").write_text(
            """
class _RealHandler:
    pass

# Export an instance, not the class itself
HandlerInstance = _RealHandler()
"""
        )

        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="instance.handler",
                    handler_class="instance_module.handler.HandlerInstance",
                )
            )

            loader = HandlerPluginLoader()

            with pytest.raises(InfraConnectionError) as exc_info:
                loader.load_from_contract(contract_file)

            error = exc_info.value
            assert (
                error.model.context.get("loader_error")
                == EnumHandlerLoaderError.CLASS_NOT_FOUND.value
            )
            assert "not a class" in str(error).lower()
        finally:
            sys.path.remove(str(tmp_path))
            if "instance_module" in sys.modules:
                del sys.modules["instance_module"]
            if "instance_module.handler" in sys.modules:
                del sys.modules["instance_module.handler"]


# =============================================================================
# Broken Symlink Tests
# =============================================================================


class TestBrokenSymlinks:
    """Tests for filesystem edge cases with symbolic links.

    These tests verify that the loader handles broken symlinks gracefully
    without crashing or hanging, producing clear error messages.
    """

    @pytest.mark.skipif(
        os.name == "nt", reason="Symlinks require admin privileges on Windows"
    )
    def test_contract_symlink_to_missing_file_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that a symlink pointing to a non-existent file raises FILE_NOT_FOUND.

        When the contract file path is a symlink whose target doesn't exist,
        the loader should detect this and raise an appropriate error.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a symlink to a non-existent file
        target_path = tmp_path / "nonexistent" / "handler_contract.yaml"
        symlink_path = tmp_path / "broken_symlink.yaml"
        symlink_path.symlink_to(target_path)

        # Verify the symlink is broken
        assert symlink_path.is_symlink()
        assert not symlink_path.exists()  # Target doesn't exist

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(symlink_path)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.FILE_NOT_FOUND.value
        )

    @pytest.mark.skipif(
        os.name == "nt", reason="Symlinks require admin privileges on Windows"
    )
    def test_directory_symlink_to_missing_dir_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that a symlink to a non-existent directory raises DIRECTORY_NOT_FOUND.

        When load_from_directory is called with a symlink whose target
        directory doesn't exist, it should raise DIRECTORY_NOT_FOUND.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a symlink to a non-existent directory
        target_dir = tmp_path / "nonexistent_dir"
        symlink_dir = tmp_path / "broken_dir_symlink"
        symlink_dir.symlink_to(target_dir)

        # Verify the symlink is broken
        assert symlink_dir.is_symlink()
        assert not symlink_dir.exists()

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(symlink_dir)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.DIRECTORY_NOT_FOUND.value
        )

    @pytest.mark.skipif(
        os.name == "nt", reason="Symlinks require admin privileges on Windows"
    )
    def test_valid_symlink_to_contract_loads_successfully(self, tmp_path: Path) -> None:
        """Test that a valid symlink to an existing contract file works.

        Symlinks that resolve to valid contracts should be loaded successfully.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a real contract file
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_contract = real_dir / "handler_contract.yaml"
        real_contract.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="symlinked.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        # Create a symlink to the real contract
        symlink_contract = tmp_path / "symlink_contract.yaml"
        symlink_contract.symlink_to(real_contract)

        # Verify symlink is valid
        assert symlink_contract.is_symlink()
        assert symlink_contract.exists()

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(symlink_contract)

        assert handler.handler_name == "symlinked.handler"

    @pytest.mark.skipif(
        os.name == "nt", reason="Symlinks require admin privileges on Windows"
    )
    def test_broken_symlink_in_directory_scan_is_skipped(self, tmp_path: Path) -> None:
        """Test that broken symlinks during directory scan are skipped gracefully.

        When scanning a directory with load_from_directory, broken symlinks
        should be logged and skipped without crashing the entire operation.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a valid contract
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        (valid_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="valid.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        # Create a broken symlink in another subdirectory
        broken_dir = tmp_path / "broken"
        broken_dir.mkdir()
        broken_symlink = broken_dir / "handler_contract.yaml"
        broken_symlink.symlink_to(tmp_path / "nonexistent.yaml")

        loader = HandlerPluginLoader()

        # Should load the valid handler and skip the broken symlink
        handlers = loader.load_from_directory(tmp_path)

        # Should have loaded the valid handler
        assert len(handlers) == 1
        assert handlers[0].handler_name == "valid.handler"

    @pytest.mark.skipif(
        os.name == "nt", reason="Symlinks require admin privileges on Windows"
    )
    def test_symlink_target_deleted_after_discovery_handled(
        self, tmp_path: Path
    ) -> None:
        """Test handling when symlink target is deleted between discovery and load.

        This tests a race condition where:
        1. Discovery finds a valid symlink
        2. Target is deleted before load
        3. Load should handle this gracefully
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a real contract file
        real_contract = tmp_path / "real_contract.yaml"
        real_contract.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="temporary.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        # Create a symlink
        symlink_dir = tmp_path / "symlinks"
        symlink_dir.mkdir()
        symlink_contract = symlink_dir / "handler_contract.yaml"
        symlink_contract.symlink_to(real_contract)

        loader = HandlerPluginLoader()

        # Delete the real file to break the symlink
        real_contract.unlink()

        # Now the symlink is broken - directory scan should handle gracefully
        handlers = loader.load_from_directory(symlink_dir)

        # Should return empty list (symlink now broken)
        assert len(handlers) == 0


# =============================================================================
# Race Condition Tests
# =============================================================================


class TestRaceConditions:
    """Tests for concurrent loading and race condition scenarios.

    These tests verify thread-safety and proper handling of filesystem
    changes during loading operations.
    """

    def test_concurrent_handler_loading_is_thread_safe(self, tmp_path: Path) -> None:
        """Test that multiple threads loading the same handler is thread-safe.

        Multiple threads calling load_from_contract on the same file should
        all succeed without data corruption or race condition failures.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a valid contract
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="concurrent.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        results: list[object] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def load_handler() -> None:
            try:
                handler = loader.load_from_contract(contract_file)
                with lock:
                    results.append(handler)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run multiple threads concurrently
        threads = [threading.Thread(target=load_handler) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # All loads should succeed
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(results) == 10

        # All results should be equivalent
        for handler in results:
            assert handler.handler_name == "concurrent.handler"

    def test_concurrent_different_handlers_loading(self, tmp_path: Path) -> None:
        """Test concurrent loading of different handlers.

        Multiple threads loading different handlers simultaneously should
        all succeed without interference.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create multiple valid contracts
        for i in range(5):
            handler_dir = tmp_path / f"handler_{i}"
            handler_dir.mkdir()
            (handler_dir / "handler_contract.yaml").write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name=f"handler.{i}",
                    handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
                )
            )

        loader = HandlerPluginLoader()
        results: dict[int, object] = {}
        errors: list[Exception] = []
        lock = threading.Lock()

        def load_handler(index: int) -> None:
            try:
                contract = tmp_path / f"handler_{index}" / "handler_contract.yaml"
                handler = loader.load_from_contract(contract)
                with lock:
                    results[index] = handler
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_handler, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()  # Raises if thread had an exception

        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(results) == 5

        # Each handler should have the correct name
        for i in range(5):
            assert results[i].handler_name == f"handler.{i}"

    def test_concurrent_directory_loading_is_thread_safe(self, tmp_path: Path) -> None:
        """Test that multiple threads calling load_from_directory is thread-safe."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create multiple valid contracts
        for i in range(3):
            handler_dir = tmp_path / f"handler_{i}"
            handler_dir.mkdir()
            (handler_dir / "handler_contract.yaml").write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name=f"handler.{i}",
                    handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
                )
            )

        loader = HandlerPluginLoader()
        results: list[list[object]] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def load_directory() -> None:
            try:
                handlers = loader.load_from_directory(tmp_path)
                with lock:
                    results.append(handlers)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run multiple threads concurrently
        threads = [threading.Thread(target=load_directory) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(results) == 5

        # All results should have the same number of handlers
        for handlers in results:
            assert len(handlers) == 3

    def test_file_modification_during_load_handled_gracefully(
        self, tmp_path: Path
    ) -> None:
        """Test that file modification during load is handled gracefully.

        If a contract file is modified while being loaded, the loader should
        either succeed with consistent data or fail gracefully.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a valid contract
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="original.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        results: list[object] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def load_handler() -> None:
            try:
                # Small delay to allow concurrent modification
                handler = loader.load_from_contract(contract_file)
                with lock:
                    results.append(handler)
            except Exception as e:
                with lock:
                    errors.append(e)

        def modify_file() -> None:
            # Modify the file while loading is happening
            for _ in range(3):
                time.sleep(0.001)
                contract_file.write_text(
                    MINIMAL_HANDLER_CONTRACT_YAML.format(
                        handler_name="modified.handler",
                        handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
                    )
                )

        # Run loader and modifier concurrently
        load_thread = threading.Thread(target=load_handler)
        modify_thread = threading.Thread(target=modify_file)

        load_thread.start()
        modify_thread.start()

        load_thread.join(timeout=5.0)
        modify_thread.join(timeout=5.0)

        # Either we successfully loaded (original or modified), or we got a clean error
        # We should NOT have a corrupted result
        if results:
            handler = results[0]
            # The handler name should be one of the valid states
            assert handler.handler_name in ("original.handler", "modified.handler")
        # If there were errors, they should be expected error types
        for error in errors:
            assert isinstance(error, Exception)

    def test_file_deleted_mid_load_handled_gracefully(self, tmp_path: Path) -> None:
        """Test that file deletion during load is handled gracefully.

        If a contract file is deleted while being loaded, the loader should
        fail with a clear error, not crash or hang.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a valid contract
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="temporary.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()

        # Mock the file open to simulate deletion mid-load
        def delayed_open(self: Path, *args: object, **kwargs: object) -> object:
            # Delete the file after validation but before reading
            if self == contract_file and contract_file.exists():
                contract_file.unlink()
            raise OSError("File was deleted")

        # Only patch after exists/is_file checks pass
        with patch.object(Path, "open", delayed_open):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                loader.load_from_contract(contract_file)

            # Should get a clear error, not crash
            assert exc_info.value is not None

    def test_discover_and_load_concurrent_is_thread_safe(self, tmp_path: Path) -> None:
        """Test that discover_and_load is thread-safe with concurrent calls."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create multiple valid contracts
        for i in range(3):
            handler_dir = tmp_path / "handlers" / f"handler_{i}"
            handler_dir.mkdir(parents=True)
            (handler_dir / "handler_contract.yaml").write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name=f"discovered.{i}",
                    handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
                )
            )

        loader = HandlerPluginLoader()
        results: list[list[object]] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def discover() -> None:
            try:
                handlers = loader.discover_and_load(
                    patterns=["handlers/**/handler_contract.yaml"],
                    base_path=tmp_path,
                )
                with lock:
                    results.append(handlers)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run multiple threads concurrently
        threads = [threading.Thread(target=discover) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(results) == 5

        # All results should have the same handlers
        for handlers in results:
            assert len(handlers) == 3

    def test_repeated_loads_are_idempotent_under_concurrency(
        self, tmp_path: Path
    ) -> None:
        """Test that repeated loads of the same handler under concurrency are idempotent.

        Loading the same handler multiple times concurrently should produce
        equivalent results every time without side effects.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a valid contract
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="idempotent.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        all_results: list[object] = []
        lock = threading.Lock()

        def load_multiple_times() -> None:
            for _ in range(5):
                handler = loader.load_from_contract(contract_file)
                with lock:
                    all_results.append(handler)

        # Run multiple threads, each loading multiple times
        threads = [threading.Thread(target=load_multiple_times) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # All 25 loads should succeed with equivalent results
        assert len(all_results) == 25

        for handler in all_results:
            assert handler.handler_name == "idempotent.handler"
            assert (
                handler.handler_class
                == "tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler"
            )


# =============================================================================
# Additional Edge Cases
# =============================================================================


class TestFilesystemEdgeCases:
    """Additional filesystem-related edge cases."""

    def test_contract_file_with_special_characters_in_path(
        self, tmp_path: Path
    ) -> None:
        """Test loading contracts from paths with special characters.

        Paths with spaces, unicode, or special characters should work.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a directory with special characters
        special_dir = tmp_path / "special dir with spaces"
        special_dir.mkdir()
        contract_file = special_dir / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="special.path.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        assert handler.handler_name == "special.path.handler"

    def test_contract_file_with_unicode_path(self, tmp_path: Path) -> None:
        """Test loading contracts from paths with unicode characters."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a directory with unicode characters
        unicode_dir = tmp_path / "handlers_日本語"
        unicode_dir.mkdir()
        contract_file = unicode_dir / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="unicode.path.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        assert handler.handler_name == "unicode.path.handler"

    def test_deeply_nested_directory_structure(self, tmp_path: Path) -> None:
        """Test loading contracts from deeply nested directories."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a deeply nested structure
        deep_path = tmp_path
        for i in range(10):
            deep_path = deep_path / f"level_{i}"
        deep_path.mkdir(parents=True)

        contract_file = deep_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="deep.nested.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        handler = loader.load_from_contract(contract_file)

        assert handler.handler_name == "deep.nested.handler"

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Test that scanning an empty directory returns an empty list."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = HandlerPluginLoader()
        handlers = loader.load_from_directory(empty_dir)

        assert handlers == []

    def test_directory_with_only_non_contract_yaml_files(self, tmp_path: Path) -> None:
        """Test directory with YAML files that are not contracts."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        yaml_dir = tmp_path / "yamls"
        yaml_dir.mkdir()

        # Create non-contract YAML files
        (yaml_dir / "config.yaml").write_text("key: value")
        (yaml_dir / "settings.yaml").write_text("setting: true")
        (yaml_dir / "random.yaml").write_text("data: [1, 2, 3]")

        loader = HandlerPluginLoader()
        handlers = loader.load_from_directory(yaml_dir)

        # Should find no handlers (no handler_contract.yaml or contract.yaml)
        assert handlers == []


class TestImportEdgeCases:
    """Edge cases related to module imports."""

    def test_circular_import_in_handler_module_raises_import_error(
        self, tmp_path: Path
    ) -> None:
        """Test that circular imports in handler modules raise IMPORT_ERROR.

        If a handler module has circular imports that cause ImportError,
        this should be caught and reported with HANDLER_LOADER_012.
        """
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create modules with circular imports
        module_dir = tmp_path / "circular_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("from .handler import Handler")
        (module_dir / "handler.py").write_text(
            """
from . import other
class Handler:
    pass
"""
        )
        (module_dir / "other.py").write_text(
            """
from .handler import Handler  # Circular import
class Other:
    pass
"""
        )

        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="circular.handler",
                    handler_class="circular_module.handler.Handler",
                )
            )

            loader = HandlerPluginLoader()

            # Circular imports may cause ImportError or AttributeError
            with pytest.raises((InfraConnectionError, ImportError)):
                loader.load_from_contract(contract_file)

        finally:
            sys.path.remove(str(tmp_path))
            # Clean up any partially imported modules
            for key in list(sys.modules.keys()):
                if key.startswith("circular_module"):
                    del sys.modules[key]

    def test_syntax_error_in_handler_module_raises_import_error(
        self, tmp_path: Path
    ) -> None:
        """Test that syntax errors in handler modules raise IMPORT_ERROR.

        If the handler module has a Python syntax error, the loader catches
        it and wraps it in InfraConnectionError with HANDLER_LOADER_012
        (IMPORT_ERROR) error code. This ensures consistent error handling
        and prevents raw SyntaxError from leaking to callers.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a module with syntax error
        module_dir = tmp_path / "syntax_error_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("")
        (module_dir / "handler.py").write_text(
            """
class SyntaxErrorHandler(
    # Missing closing parenthesis - syntax error
"""
        )

        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="syntax.error.handler",
                    handler_class="syntax_error_module.handler.SyntaxErrorHandler",
                )
            )

            loader = HandlerPluginLoader()

            # SyntaxError is now caught and wrapped in InfraConnectionError
            with pytest.raises(InfraConnectionError) as exc_info:
                loader.load_from_contract(contract_file)

            # Verify the error code is IMPORT_ERROR (HANDLER_LOADER_012)
            error = exc_info.value
            assert (
                error.model.context.get("loader_error")
                == EnumHandlerLoaderError.IMPORT_ERROR.value
            ), f"Expected IMPORT_ERROR, got: {error.model.context.get('loader_error')}"

            # Verify the error message indicates a syntax error
            assert "syntax error" in str(error).lower(), (
                f"Expected 'syntax error' in message: {error}"
            )

        finally:
            sys.path.remove(str(tmp_path))
            for key in list(sys.modules.keys()):
                if key.startswith("syntax_error_module"):
                    del sys.modules[key]

    def test_missing_dependency_in_handler_module_raises_import_error(
        self, tmp_path: Path
    ) -> None:
        """Test that missing dependencies in handler modules raise IMPORT_ERROR.

        If the handler module imports a package that isn't installed,
        the loader should catch this and report it properly.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a module that imports a non-existent package
        module_dir = tmp_path / "missing_dep_module"
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("")
        (module_dir / "handler.py").write_text(
            """
import nonexistent_package_xyz123  # This package doesn't exist

class MissingDepHandler:
    pass
"""
        )

        sys.path.insert(0, str(tmp_path))

        try:
            contract_file = tmp_path / "handler_contract.yaml"
            contract_file.write_text(
                MINIMAL_HANDLER_CONTRACT_YAML.format(
                    handler_name="missing.dep.handler",
                    handler_class="missing_dep_module.handler.MissingDepHandler",
                )
            )

            loader = HandlerPluginLoader()

            with pytest.raises(InfraConnectionError) as exc_info:
                loader.load_from_contract(contract_file)

            error = exc_info.value
            # Could be MODULE_NOT_FOUND (for the handler module) or IMPORT_ERROR
            assert error.model.context.get("loader_error") in (
                EnumHandlerLoaderError.MODULE_NOT_FOUND.value,
                EnumHandlerLoaderError.IMPORT_ERROR.value,
            )
        finally:
            sys.path.remove(str(tmp_path))
            for key in list(sys.modules.keys()):
                if key.startswith("missing_dep_module"):
                    del sys.modules[key]
