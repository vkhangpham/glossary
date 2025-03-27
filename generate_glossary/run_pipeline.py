#!/usr/bin/env python3
"""
Glossary Generation Pipeline Runner

This script automates the entire glossary generation pipeline for any level (0-4).
It handles both the generation phase and the processing phase.

Usage:
    python -m generate_glossary.run_pipeline --level 0 --provider openai

The script will:
1. Run the appropriate generation scripts for the specified level
2. Mine web content for the generated concepts
3. Run validation steps (rule, web, llm)
4. Run deduplication (graph mode)
"""

import argparse
import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
VALID_LEVELS = [0, 1, 2, 3, 4]
VALID_PROVIDERS = ["openai", "gemini", "anthropic", "deepseek"]
DEFAULT_PROVIDER = "gemini"

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the glossary generation pipeline for a specific level")
    
    parser.add_argument(
        "--level", "-l",
        type=int,
        choices=VALID_LEVELS,
        required=True,
        help="Level to process (0-4)"
    )
    
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=VALID_PROVIDERS,
        default=DEFAULT_PROVIDER,
        help=f"LLM provider to use (default: {DEFAULT_PROVIDER})"
    )
    
    parser.add_argument(
        "--skip-generation", "-sg",
        action="store_true",
        help="Skip the generation phase and only run the processing phase"
    )
    
    parser.add_argument(
        "--skip-web-mining", "-sw",
        action="store_true",
        help="Skip the web mining step"
    )
    
    parser.add_argument(
        "--skip-validation", "-sv",
        action="store_true",
        help="Skip the validation steps"
    )
    
    parser.add_argument(
        "--skip-deduplication", "-sd",
        action="store_true",
        help="Skip the deduplication step"
    )
    
    parser.add_argument(
        "--dedup-mode", "-dm",
        type=str,
        choices=["graph", "rule", "web", "llm"],
        default="graph",
        help="Deduplication mode to use (default: graph)"
    )
    
    parser.add_argument(
        "--cooldown", "-c",
        type=int,
        default=5,
        help="Cooldown period in seconds between steps (default: 5)"
    )
    
    return parser.parse_args()

def run_command(command: List[str], description: str) -> Tuple[int, str, str]:
    """Run a command and return the exit code, stdout, and stderr."""
    logging.info(f"Running: {description}")
    logging.debug(f"Command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    exit_code = process.returncode
    
    if exit_code != 0:
        logging.error(f"Command failed with exit code {exit_code}")
        logging.error(f"Error output: {stderr}")
    else:
        logging.info(f"Command completed successfully")
    
    return exit_code, stdout, stderr

def ensure_directory_exists(directory: str) -> None:
    """Ensure that a directory exists, creating it if necessary."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def run_generation_phase(level: int, provider: str) -> bool:
    """Run the generation phase for the specified level."""
    logging.info(f"Starting generation phase for level {level}")
    
    # Ensure data directories exist
    data_dir = f"data/lv{level}/raw"
    ensure_directory_exists(data_dir)
    
    # Define the generation scripts for each level
    generation_scripts = {
        0: [
            ["python", f"generate_glossary/generation/lv0/lv0_s0_get_college_names.py"],
            ["python", f"generate_glossary/generation/lv0/lv0_s1_extract_concepts.py", "--provider", provider],
            ["python", f"generate_glossary/generation/lv0/lv0_s2_filter_by_institution_freq.py"],
            ["python", f"generate_glossary/generation/lv0/lv0_s3_verify_single_token.py", "--provider", provider]
        ],
        1: [
            ["python", f"generate_glossary/generation/lv1/lv1_s0_get_dept_names.py"],
            ["python", f"generate_glossary/generation/lv1/lv1_s1_extract_concepts.py", "--provider", provider],
            ["python", f"generate_glossary/generation/lv1/lv1_s2_filter_by_institution_freq.py"],
            ["python", f"generate_glossary/generation/lv1/lv1_s3_verify_single_token.py", "--provider", provider]
        ],
        2: [
            ["python", f"generate_glossary/generation/lv2/lv2_s0_get_journal_names.py"],
            ["python", f"generate_glossary/generation/lv2/lv2_s1_extract_concepts.py", "--provider", provider],
            ["python", f"generate_glossary/generation/lv2/lv2_s2_filter_by_frequency.py"],
            ["python", f"generate_glossary/generation/lv2/lv2_s3_verify_concepts.py", "--provider", provider]
        ],
        3: [
            ["python", f"generate_glossary/generation/lv3/lv3_s0_get_paper_titles.py"],
            ["python", f"generate_glossary/generation/lv3/lv3_s1_extract_concepts.py", "--provider", provider],
            ["python", f"generate_glossary/generation/lv3/lv3_s2_filter_by_frequency.py"],
            ["python", f"generate_glossary/generation/lv3/lv3_s3_verify_concepts.py", "--provider", provider]
        ],
        4: [
            ["python", f"generate_glossary/generation/lv4/lv4_s0_get_paper_abstracts.py"],
            ["python", f"generate_glossary/generation/lv4/lv4_s1_extract_concepts.py", "--provider", provider],
            ["python", f"generate_glossary/generation/lv4/lv4_s2_filter_by_frequency.py"],
            ["python", f"generate_glossary/generation/lv4/lv4_s3_verify_concepts.py", "--provider", provider]
        ]
    }
    
    # Run each script in sequence
    for i, script in enumerate(generation_scripts[level]):
        step_name = f"Step {i}"
        exit_code, stdout, stderr = run_command(script, f"{step_name} of generation phase")
        
        if exit_code != 0:
            logging.error(f"Generation phase failed at {step_name}")
            return False
        
        # Add a small cooldown between steps
        time.sleep(2)
    
    logging.info(f"Generation phase completed successfully for level {level}")
    return True

def run_web_mining(level: int) -> bool:
    """Run the web mining step for the specified level."""
    logging.info(f"Starting web mining for level {level}")
    
    # Ensure output directory exists
    data_dir = f"data/lv{level}"
    ensure_directory_exists(data_dir)
    
    # Define the input and output files
    input_file = f"data/lv{level}/raw/lv{level}_s3_verified_concepts.txt"
    output_file = f"data/lv{level}/lv{level}_resources.json"
    
    # Run the web mining command
    command = [
        "python", "-m", "generate_glossary.web_miner_cli",
        "--input", input_file,
        "--output", output_file
    ]
    
    exit_code, stdout, stderr = run_command(command, "Web mining")
    
    if exit_code != 0:
        logging.error("Web mining failed")
        return False
    
    logging.info(f"Web mining completed successfully for level {level}")
    return True

def run_validation(level: int, provider: str, cooldown: int) -> bool:
    """Run the validation steps for the specified level."""
    logging.info(f"Starting validation for level {level}")
    
    # Ensure output directory exists
    output_dir = f"data/lv{level}/postprocessed"
    ensure_directory_exists(output_dir)
    
    # Define the validation steps
    validation_steps = [
        {
            "name": "Rule-based validation",
            "command": [
                "python", "-m", "generate_glossary.validator.cli",
                f"data/lv{level}/raw/lv{level}_s3_verified_concepts.txt",
                "-m", "rule",
                "-o", f"data/lv{level}/postprocessed/lv{level}_rv"
            ]
        },
        {
            "name": "Web-based validation",
            "command": [
                "python", "-m", "generate_glossary.validator.cli",
                f"data/lv{level}/postprocessed/lv{level}_rv.txt",
                "-m", "web",
                "-w", f"data/lv{level}/lv{level}_resources.json",
                "-o", f"data/lv{level}/postprocessed/lv{level}_wv"
            ]
        },
        {
            "name": "LLM-based validation",
            "command": [
                "python", "-m", "generate_glossary.validator.cli",
                f"data/lv{level}/postprocessed/lv{level}_wv.txt",
                "-m", "llm",
                "-p", provider,
                "-o", f"data/lv{level}/postprocessed/lv{level}_lv"
            ]
        }
    ]
    
    # Run each validation step in sequence
    for step in validation_steps:
        exit_code, stdout, stderr = run_command(step["command"], step["name"])
        
        if exit_code != 0:
            logging.error(f"Validation failed at {step['name']}")
            return False
        
        # Add a cooldown between steps
        time.sleep(cooldown)
    
    logging.info(f"Validation completed successfully for level {level}")
    return True

def run_deduplication(level: int, provider: str, mode: str) -> bool:
    """Run the deduplication step for the specified level."""
    logging.info(f"Starting deduplication for level {level} using {mode} mode")
    
    # Ensure output directory exists
    output_dir = f"data/lv{level}/postprocessed"
    ensure_directory_exists(output_dir)
    
    # Build the deduplication command
    command = [
        "python", "-m", "generate_glossary.deduplicator.cli",
        f"data/lv{level}/postprocessed/lv{level}_lv.txt",
        "-m", mode,
        "-w", f"data/lv{level}/lv{level}_resources.json",
        "-o", f"data/lv{level}/postprocessed/lv{level}_final"
    ]
    
    # Add provider for LLM mode
    if mode == "llm":
        command.extend(["-p", provider])
    
    # Add higher level terms for cross-level deduplication (if not level 0)
    if level > 0:
        higher_level_terms = []
        higher_level_web_content = []
        
        for i in range(level):
            higher_level_terms.append(f"{i}:data/lv{i}/postprocessed/lv{i}_final.txt")
            higher_level_web_content.append(f"{i}:data/lv{i}/lv{i}_resources.json")
        
        if higher_level_terms:
            command.extend(["-t"] + higher_level_terms)
            command.extend(["-c"] + higher_level_web_content)
    
    exit_code, stdout, stderr = run_command(command, "Deduplication")
    
    if exit_code != 0:
        logging.error("Deduplication failed")
        return False
    
    logging.info(f"Deduplication completed successfully for level {level}")
    return True

def count_concepts(file_path: str) -> int:
    """Count the number of concepts in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        logging.error(f"Error counting concepts: {e}")
        return 0

def main() -> None:
    """Main function to run the pipeline."""
    args = parse_args()
    
    # Set PYTHONPATH to project root
    os.environ["PYTHONPATH"] = "."
    
    level = args.level
    provider = args.provider
    cooldown = args.cooldown
    
    logging.info(f"Starting glossary generation pipeline for level {level}")
    logging.info(f"Using LLM provider: {provider}")
    
    # Run the generation phase
    if not args.skip_generation:
        if not run_generation_phase(level, provider):
            logging.error("Pipeline failed during generation phase")
            sys.exit(1)
        time.sleep(cooldown)
    else:
        logging.info("Skipping generation phase")
    
    # Run web mining
    if not args.skip_web_mining:
        if not run_web_mining(level):
            logging.error("Pipeline failed during web mining")
            sys.exit(1)
        time.sleep(cooldown)
    else:
        logging.info("Skipping web mining")
    
    # Run validation
    if not args.skip_validation:
        if not run_validation(level, provider, cooldown):
            logging.error("Pipeline failed during validation")
            sys.exit(1)
        time.sleep(cooldown)
    else:
        logging.info("Skipping validation")
    
    # Run deduplication
    if not args.skip_deduplication:
        if not run_deduplication(level, provider, args.dedup_mode):
            logging.error("Pipeline failed during deduplication")
            sys.exit(1)
    else:
        logging.info("Skipping deduplication")
    
    # Print summary
    logging.info(f"Level {level} pipeline completed successfully!")
    
    # Count concepts at different stages
    raw_concepts = count_concepts(f"data/lv{level}/raw/lv{level}_s3_verified_concepts.txt")
    rule_validated = count_concepts(f"data/lv{level}/postprocessed/lv{level}_rv.txt")
    web_validated = count_concepts(f"data/lv{level}/postprocessed/lv{level}_wv.txt")
    llm_validated = count_concepts(f"data/lv{level}/postprocessed/lv{level}_lv.txt")
    final_concepts = count_concepts(f"data/lv{level}/postprocessed/lv{level}_final.txt")
    
    logging.info("Concept counts at different stages:")
    logging.info(f"  Raw concepts: {raw_concepts}")
    logging.info(f"  Rule-validated: {rule_validated}")
    logging.info(f"  Web-validated: {web_validated}")
    logging.info(f"  LLM-validated: {llm_validated}")
    logging.info(f"  Final concepts: {final_concepts}")
    
    logging.info("================================================")

if __name__ == "__main__":
    main() 