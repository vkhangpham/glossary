#!/usr/bin/env python3
"""
Test runner script for the glossary generation system

This script provides convenient ways to run different types of tests:
- Unit tests (fast, no external dependencies)
- Integration tests (require API keys)
- All tests
- Specific test categories

Usage:
    python tests/run_tests.py                 # Run all tests
    python tests/run_tests.py --unit          # Run only unit tests  
    python tests/run_tests.py --integration   # Run only integration tests
    python tests/run_tests.py --migration     # Run only migration compatibility tests
    python tests/run_tests.py --coverage      # Run with coverage report
    python tests/run_tests.py --fast          # Run fast tests only (unit tests)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description or ' '.join(cmd)}")
    print(f"{'='*50}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        return False
    return True


def check_requirements():
    """Check that required packages are installed"""
    required_packages = ["pytest", "pytest-cov"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Install with: uv add --dev pytest pytest-cov")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run tests for the glossary system")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--migration", action="store_true", help="Run only migration compatibility tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--file", "-f", help="Run specific test file")
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Base pytest command
    cmd = ["uv", "run", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=generate_glossary.utils.llm_simple",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Determine which tests to run
    if args.file:
        cmd.append(f"tests/{args.file}")
        description = f"Running specific test file: {args.file}"
    elif args.unit:
        cmd.extend(["-m", "unit", "tests/unit/"])
        description = "Running unit tests only"
    elif args.integration:
        cmd.extend(["-m", "integration", "tests/integration/"])
        description = "Running integration tests only (requires API keys)"
    elif args.migration:
        cmd.append("tests/unit/test_migration_compatibility.py")
        description = "Running migration compatibility tests"
    elif args.fast:
        cmd.extend(["-m", "not slow", "tests/unit/"])
        description = "Running fast tests only"
    else:
        cmd.append("tests/")
        description = "Running all tests"
    
    # Run the tests
    success = run_command(cmd, description)
    
    if success:
        print("\n🎉 All tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")
        
        if args.integration and not any(os.getenv(key) for key in ["OPENAI_API_KEY", "GEMINI_API_KEY"]):
            print("\n⚠️  Integration tests were skipped due to missing API keys")
            print("   Set OPENAI_API_KEY and/or GEMINI_API_KEY to run integration tests")
            
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())