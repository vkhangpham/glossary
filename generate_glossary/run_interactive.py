#!/usr/bin/env python3
"""
Interactive Glossary Generation Pipeline Runner

This script provides an interactive interface to run the glossary generation pipeline.
It prompts the user for the level and other options, then runs the pipeline script.
"""

import os
import sys
import subprocess
from typing import Dict, Any, List

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the script header."""
    clear_screen()
    print("=" * 80)
    print("Glossary Generation Pipeline - Interactive Runner".center(80))
    print("=" * 80)
    print()

def get_user_input(prompt: str, options: List[str] = None, default: str = None) -> str:
    """
    Get user input with validation.
    
    Args:
        prompt: The prompt to display
        options: Optional list of valid options
        default: Optional default value
        
    Returns:
        The user's input
    """
    prompt_text = f"{prompt}"
    if options:
        prompt_text += f" ({'/'.join(options)})"
    if default:
        prompt_text += f" [default: {default}]"
    prompt_text += ": "
    
    while True:
        user_input = input(prompt_text).strip()
        
        if not user_input and default:
            return default
        
        if not options or user_input in options:
            return user_input
        
        print(f"Invalid input. Please enter one of: {', '.join(options)}")

def get_boolean_input(prompt: str, default: bool = False) -> bool:
    """
    Get a yes/no input from the user.
    
    Args:
        prompt: The prompt to display
        default: The default value (True for yes, False for no)
        
    Returns:
        True for yes, False for no
    """
    default_str = "y" if default else "n"
    options = ["y", "n"]
    
    result = get_user_input(f"{prompt} (y/n)", options, default_str)
    return result.lower() == "y"

def run_pipeline(config: Dict[str, Any]) -> None:
    """
    Run the pipeline with the given configuration.
    
    Args:
        config: Dictionary of configuration options
    """
    # Build the command
    cmd = [
        sys.executable,
        "-m",
        "generate_glossary.run_pipeline",
        "--level", str(config["level"]),
        "--provider", config["provider"],
        "--cooldown", str(config["cooldown"]),
        "--dedup-mode", config["dedup_mode"]
    ]
    
    # Add skip flags
    if config["skip_generation"]:
        cmd.append("--skip-generation")
    if config["skip_web_mining"]:
        cmd.append("--skip-web-mining")
    if config["skip_validation"]:
        cmd.append("--skip-validation")
    if config["skip_deduplication"]:
        cmd.append("--skip-deduplication")
    
    # Print the command
    print("\nRunning command:")
    print(" ".join(cmd))
    print("\nStarting pipeline execution...\n")
    
    # Run the command
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\nPipeline failed with exit code {process.returncode}")
        else:
            print("\nPipeline completed successfully!")
    
    except KeyboardInterrupt:
        print("\nPipeline execution interrupted by user")
        process.terminate()
    except Exception as e:
        print(f"\nError running pipeline: {e}")

def main() -> None:
    """Main function to run the interactive pipeline."""
    print_header()
    
    # Get configuration from user
    config = {}
    
    # Level
    level_options = ["0", "1", "2", "3", "4"]
    config["level"] = int(get_user_input("Enter the level to process", level_options, "0"))
    
    # Provider
    provider_options = ["openai", "gemini", "anthropic", "deepseek"]
    config["provider"] = get_user_input("Enter the LLM provider to use", provider_options, "gemini")
    
    # Cooldown
    cooldown_str = get_user_input("Enter the cooldown period in seconds between steps", default="5")
    config["cooldown"] = int(cooldown_str)
    
    # Deduplication mode
    dedup_options = ["graph", "rule", "web", "llm"]
    config["dedup_mode"] = get_user_input("Enter the deduplication mode to use", dedup_options, "graph")
    
    # Skip options
    print("\nSelect which phases to skip (if any):")
    config["skip_generation"] = get_boolean_input("Skip generation phase?", False)
    config["skip_web_mining"] = get_boolean_input("Skip web mining?", False)
    config["skip_validation"] = get_boolean_input("Skip validation?", False)
    config["skip_deduplication"] = get_boolean_input("Skip deduplication?", False)
    
    # Confirm
    print("\nPipeline Configuration:")
    print(f"  Level: {config['level']}")
    print(f"  Provider: {config['provider']}")
    print(f"  Cooldown: {config['cooldown']} seconds")
    print(f"  Deduplication mode: {config['dedup_mode']}")
    print(f"  Skip generation: {config['skip_generation']}")
    print(f"  Skip web mining: {config['skip_web_mining']}")
    print(f"  Skip validation: {config['skip_validation']}")
    print(f"  Skip deduplication: {config['skip_deduplication']}")
    
    if get_boolean_input("\nProceed with this configuration?", True):
        run_pipeline(config)
    else:
        print("\nPipeline execution cancelled")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(1) 