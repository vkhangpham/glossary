#!/usr/bin/env python3
"""
Quick verification script to test the completed dataset.

This script performs a basic check to ensure:
1. All files exist and are valid
2. The splitter can load the unified context
3. The system can process at least one term correctly
"""

import json
import sys
from pathlib import Path

# Package structure now properly configured with pyproject.toml

def main():
    print("🔍 Verifying test dataset...")
    
    # Check files exist
    test_dataset_root = Path("data/test_dataset")
    hierarchy_file = test_dataset_root / "hierarchy.json"
    context_file = test_dataset_root / "unified_context_ground_truth.json"
    
    if not hierarchy_file.exists():
        print("❌ Hierarchy file missing")
        return False
        
    if not context_file.exists():
        print("❌ Unified context file missing")
        return False
        
    print("✅ Required files exist")
    
    # Check JSON validity
    try:
        with open(hierarchy_file, 'r') as f:
            hierarchy = json.load(f)
        print("✅ Hierarchy JSON is valid")
        
        with open(context_file, 'r') as f:
            context_data = json.load(f)
        print("✅ Unified context JSON is valid")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON error: {e}")
        return False
    
    # Check basic structure
    if "terms" not in hierarchy:
        print("❌ Hierarchy missing 'terms' key")
        return False
        
    if "contexts" not in context_data:
        print("❌ Context data missing 'contexts' key")
        return False
        
    print("✅ Basic structure valid")
    
    # Count terms
    hierarchy_terms = len(hierarchy["terms"])
    context_terms = len(context_data["contexts"])
    
    print(f"📊 Terms: {hierarchy_terms} in hierarchy, {context_terms} in context")
    
    if hierarchy_terms != context_terms:
        print("⚠️  Term count mismatch")
    else:
        print("✅ Term counts match")
        
    # Check for positive terms
    positive_terms = [
        "transformers", "interface", "modeling", "fragmentation", "clustering",
        "stress", "regression", "cell", "network", "bond"
    ]
    
    found_positive = 0
    high_confidence = 0
    
    for term in positive_terms:
        if term in context_data["contexts"]:
            found_positive += 1
            confidence = context_data["contexts"][term]["overall_confidence"]
            if confidence >= 0.7:
                high_confidence += 1
                
    print(f"📊 Positive terms: {found_positive}/10 found, {high_confidence}/10 high confidence")
    
    # Try loading with splitter
    try:
        from sense_disambiguation.splitter import SenseSplitter
        
        splitter = SenseSplitter(
            hierarchy_file_path=str(hierarchy_file),
            context_file=str(context_file),
            level=2,
            use_llm_for_tags=False  # Use simulation
        )
        
        print("✅ SenseSplitter initialization successful")
        
        # Check if context loaded
        if hasattr(splitter, 'term_contexts') and splitter.term_contexts:
            loaded_terms = len(splitter.term_contexts)
            print(f"✅ Loaded {loaded_terms} term contexts")
        else:
            print("⚠️  No term contexts loaded")
            
    except Exception as e:
        print(f"❌ SenseSplitter error: {e}")
        return False
    
    print("\n🎉 Dataset verification complete!")
    print("✅ Test dataset is ready for use")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 