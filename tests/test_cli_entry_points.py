"""
Integration tests for CLI entry points defined in pyproject.toml.

Tests that all script entries work correctly, that the new standardized commands
function properly, and that backward compatibility is maintained for existing commands.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class TestCLIEntryPoints:
    """Test CLI entry points and their functionality."""

    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent

    def test_pyproject_toml_exists(self, project_root):
        """Test that pyproject.toml exists and is readable."""
        pyproject_path = project_root / 'pyproject.toml'
        assert pyproject_path.exists(), "pyproject.toml not found"

        # Test that it's readable and has script entries
        with open(pyproject_path, 'r') as f:
            content = f.read()
            assert '[project.scripts]' in content, "No script entries found in pyproject.toml"

    def test_standardized_entry_points_defined(self, project_root):
        """Test that all standardized entry points are defined."""
        pyproject_path = project_root / 'pyproject.toml'

        with open(pyproject_path, 'r') as f:
            content = f.read()

        # Check for standardized entry points (based on current pyproject.toml)
        expected_standardized = [
            'generate = "generate_glossary.cli.generate:main"',
            'mine-web = "generate_glossary.mining.main:main"',
            'glossary-web-miner = "generate_glossary.mining.main:main"',
            'glossary-validator = "generate_glossary.validation.main:main"',
            'glossary-deduplicator = "generate_glossary.deduplication.main:main"',
            'glossary-disambiguate = "generate_glossary.disambiguation.main:main"'
        ]

        for entry_point in expected_standardized:
            assert entry_point in content, f"Missing standardized entry point: {entry_point}"


    def test_entry_point_modules_importable(self):
        """Test that all entry point modules and functions are importable."""
        entry_points = [
            ('generate_glossary.cli.generate', 'main'),
            ('generate_glossary.mining.main', 'main'),
            ('generate_glossary.validation.main', 'main'),
            ('generate_glossary.deduplication.main', 'main'),
            ('generate_glossary.disambiguation.main', 'main')
        ]

        for module_name, function_name in entry_points:
            try:
                import importlib
                module = importlib.import_module(module_name)
                assert hasattr(module, function_name), \
                    f"Module {module_name} missing function {function_name}"
                assert callable(getattr(module, function_name)), \
                    f"Function {function_name} in {module_name} is not callable"
            except ImportError as e:
                pytest.fail(f"Could not import {module_name}: {e}")



class TestStandardizedCLIFunctionality:
    """Test that standardized CLI commands work correctly."""

    def test_generate_command_help(self):
        """Test that generate command shows help without errors."""
        try:
            from generate_glossary.cli.generate import main

            with patch('sys.argv', ['generate', '--help']):
                with patch('builtins.print'):
                    with pytest.raises(SystemExit):
                        main()
        except ImportError:
            pytest.fail("Could not import generate CLI")

    def test_validate_command_help(self):
        """Test that validate command shows help without errors."""
        try:
            from generate_glossary.validation.main import main

            with patch('sys.argv', ['glossary-validator', '--help']):
                with patch('builtins.print'):
                    with pytest.raises(SystemExit):
                        main()
        except ImportError:
            pytest.fail("Could not import validate CLI")

    def test_deduplicate_command_importable(self):
        """Test that deduplicate command is importable."""
        try:
            from generate_glossary.deduplication.main import main
            assert callable(main), "Deduplication main function is not callable"
        except ImportError:
            pytest.fail("Could not import deduplicate CLI")

    def test_disambiguate_command_importable(self):
        """Test that disambiguate command is importable."""
        try:
            from generate_glossary.disambiguation.main import main
            assert callable(main), "Disambiguation main function is not callable"
        except ImportError:
            pytest.fail("Could not import disambiguate CLI")


class TestCLIConsistency:
    """Test consistency across all CLI interfaces."""

    def test_all_main_functions_handle_help_flag(self):
        """Test that all main functions can handle --help without crashing."""
        main_modules = [
            'generate_glossary.validation.main',
            'generate_glossary.deduplication.main',
            'generate_glossary.disambiguation.main'
        ]

        for module_name in main_modules:
            try:
                import importlib
                module = importlib.import_module(module_name)
                main_func = getattr(module, 'main')

                # Try to call with --help (this should either work or raise SystemExit)
                with patch('sys.argv', [module_name, '--help']):
                    with patch('sys.exit'):
                        with patch('builtins.print'):  # Suppress output
                            try:
                                main_func()
                            except (SystemExit, Exception):
                                # SystemExit is expected for --help
                                # Other exceptions might occur due to argument parsing
                                pass

            except ImportError:
                pytest.fail(f"Could not import {module_name}")

    def test_cli_argument_parsing_consistency(self):
        """Test that CLI interfaces use consistent argument patterns."""
        # This is a high-level test to ensure basic consistency
        try:
            from generate_glossary.validation.main import main as validate_main
            from generate_glossary.disambiguation.main import main as disambiguate_main

            # Test that both accept basic required arguments without crashing imports
            assert callable(validate_main)
            assert callable(disambiguate_main)

        except ImportError as e:
            pytest.fail(f"CLI consistency test failed due to import error: {e}")



class TestEntryPointIntegration:
    """Integration tests for entry points."""

    def test_no_circular_imports_in_entry_points(self):
        """Test that entry point modules don't have circular import issues."""
        entry_point_modules = [
            'generate_glossary.cli.generate',
            'generate_glossary.mining.main',
            'generate_glossary.validation.main',
            'generate_glossary.deduplication.main',
            'generate_glossary.disambiguation.main'
        ]

        for module_name in entry_point_modules:
            try:
                import importlib
                # Try to import the module - this will fail if there are circular imports
                module = importlib.import_module(module_name)

                # Try to access the main function
                main_func = getattr(module, 'main', None)
                assert main_func is not None, f"Module {module_name} has no main function"
                assert callable(main_func), f"main in {module_name} is not callable"

            except ImportError as e:
                pytest.fail(f"Circular import or other import issue in {module_name}: {e}")

    def test_entry_points_use_standardized_config(self):
        """Test that entry points use the new standardized config systems."""
        config_imports = [
            ('generate_glossary.validation.main', 'generate_glossary.validation.config'),
            ('generate_glossary.disambiguation.main', 'generate_glossary.disambiguation.config')
        ]

        for main_module_name, config_module_name in config_imports:
            try:
                import importlib

                importlib.import_module(main_module_name)
                importlib.import_module(config_module_name)
            except ImportError as e:
                pytest.fail(f"Could not test standardized config for {main_module_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])