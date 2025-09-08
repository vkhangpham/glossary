#!/usr/bin/env python
"""
Simple runner script for lv0_s1 concept extraction prompt optimization.

Usage:
    python run_lv0_s1_optimization.py
    
Requires:
    - OPENAI_API_KEY (in .env file or environment variable)
    - Training data in data/prompts_training_data/lv0_s1.json
    
Output:
    - Optimized prompts saved to data/prompts/lv0_s1_system_latest.json
    - Optimized prompts saved to data/prompts/lv0_s1_user_latest.json
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Run the lv0_s1 prompt optimization."""
    print("=" * 60)
    print("LV0_S1 Concept Extraction Prompt Optimization")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        print("Please set it in .env file or with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Check for training data
    training_file = Path("data/prompts_training_data/lv0_s1.json")
    if not training_file.exists():
        print(f"❌ Error: Training data not found at {training_file}")
        print("Please ensure training data is available")
        sys.exit(1)
    
    print(f"✓ Found training data at {training_file}")
    print(f"✓ OPENAI_API_KEY is set")
    print()
    
    try:
        # Import and run the optimizer
        from prompt_optimization.optimizers.lv0_s1 import optimize_prompts
        
        print("Starting optimization process...")
        print("This may take several minutes depending on the training data size.")
        print()
        
        # Run optimization
        system_prompt, user_prompt = optimize_prompts()
        
        print()
        print("=" * 60)
        print("✅ Optimization completed successfully!")
        print("=" * 60)
        print()
        print("Optimized prompts have been saved to:")
        print("  • data/prompts/lv0_s1_system_latest.json")
        print("  • data/prompts/lv0_s1_user_latest.json")
        print()
        print("You can now use these prompts in your generation scripts.")
        
    except ImportError as e:
        print(f"❌ Error importing optimizer: {e}")
        print("Please ensure all dependencies are installed with: uv sync")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()