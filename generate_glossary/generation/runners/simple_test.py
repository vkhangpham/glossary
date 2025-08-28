#!/usr/bin/env python3
"""
Simple architectural test to validate the functional consolidation.

This test focuses on validating the architecture without requiring 
external dependencies like LiteLLM, web search APIs, etc.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from generate_glossary.generation.shared.level_config import get_level_config, get_step_file_paths


def test_level_configs():
    """Test that level configurations are properly defined."""
    print("Testing level configurations...")
    
    for level in [1, 2, 3]:
        try:
            config = get_level_config(level)
            print(f"  Level {level}:")
            print(f"    - Batch size: {config.batch_size}")
            print(f"    - Agreement threshold: {config.agreement_threshold}")
            print(f"    - Search patterns: {len(config.search_patterns)}")
            print(f"    - Quality keywords: {len(config.quality_keywords)}")
            print(f"    - Frequency threshold: {config.frequency_threshold}")
            print(f"    - Description: {config.processing_description}")
            
        except Exception as e:
            print(f"  ❌ Level {level} config failed: {e}")
            return False
    
    print("  ✅ All level configurations loaded successfully")
    return True


def test_file_paths():
    """Test that file paths are generated correctly."""
    print("\nTesting file path generation...")
    
    for level in [1, 2, 3]:
        for step in ["s0", "s1", "s2", "s3"]:
            try:
                input_file, output_file, metadata_file = get_step_file_paths(level, step)
                print(f"  Level {level} Step {step}:")
                print(f"    - Input: {input_file}")
                print(f"    - Output: {output_file}")
                print(f"    - Metadata: {metadata_file}")
                
                # Validate file paths make sense
                assert f"lv{level}" in output_file, f"Output file should contain lv{level}"
                assert f"lv{level}" in metadata_file, f"Metadata file should contain lv{level}"
                assert output_file.endswith(".txt"), f"Output should be .txt file"
                assert metadata_file.endswith(".json"), f"Metadata should be .json file"
                
            except Exception as e:
                print(f"  ❌ Level {level} Step {step} failed: {e}")
                return False
    
    print("  ✅ All file paths generated correctly")
    return True


def test_configuration_differences():
    """Test that configurations differ appropriately between levels."""
    print("\nTesting configuration differences...")
    
    configs = {}
    for level in [1, 2, 3]:
        configs[level] = get_level_config(level)
    
    # Test that batch sizes decrease for higher levels (more complexity)
    assert configs[1].batch_size >= configs[2].batch_size, "Level 1 should have larger or equal batch size than Level 2"
    assert configs[2].batch_size >= configs[3].batch_size, "Level 2 should have larger or equal batch size than Level 3"
    
    # Test that agreement thresholds increase for higher levels (more precision)
    assert configs[2].agreement_threshold >= configs[1].agreement_threshold, "Level 2 should have higher or equal agreement threshold than Level 1"
    assert configs[3].agreement_threshold >= configs[2].agreement_threshold, "Level 3 should have higher or equal agreement threshold than Level 2"
    
    # Test that search patterns are different
    patterns_1 = set(configs[1].search_patterns)
    patterns_2 = set(configs[2].search_patterns)
    patterns_3 = set(configs[3].search_patterns)
    
    assert patterns_1 != patterns_2, "Level 1 and 2 should have different search patterns"
    assert patterns_2 != patterns_3, "Level 2 and 3 should have different search patterns"
    
    # Test that Level 3 has special frequency handling
    assert configs[3].frequency_threshold == "venue_based", "Level 3 should use venue-based frequency"
    assert isinstance(configs[1].frequency_threshold, float), "Level 1 should use numeric frequency threshold"
    assert isinstance(configs[2].frequency_threshold, float), "Level 2 should use numeric frequency threshold"
    
    print("  ✅ Configuration differences validated")
    return True


def test_architecture_consolidation():
    """Test that the architecture consolidates properly.""" 
    print("\nTesting architecture consolidation...")
    
    # Test that we can import shared utilities
    try:
        from generate_glossary.generation.shared.web_extraction import extract_web_content_simple
        print("  ✅ Web extraction utility imported")
    except Exception as e:
        print(f"  ❌ Web extraction import failed: {e}")
        return False
    
    try:
        from generate_glossary.generation.shared.level_config import LEVEL_CONFIGS
        print(f"  ✅ Level configs loaded: {list(LEVEL_CONFIGS.keys())}")
    except Exception as e:
        print(f"  ❌ Level config import failed: {e}")
        return False
    
    # Test that the configuration covers all expected levels
    expected_levels = {1, 2, 3}
    actual_levels = set(LEVEL_CONFIGS.keys())
    
    assert expected_levels == actual_levels, f"Expected levels {expected_levels}, got {actual_levels}"
    
    print("  ✅ Architecture consolidation validated")
    return True


def main():
    """Run all architectural tests."""
    print("🧪 Testing Functional Consolidation Architecture")
    print("=" * 60)
    
    tests = [
        ("Level Configurations", test_level_configs),
        ("File Path Generation", test_file_paths),
        ("Configuration Differences", test_configuration_differences),
        ("Architecture Consolidation", test_architecture_consolidation)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All architectural tests passed!")
        print("\n✅ Functional consolidation architecture is working correctly")
        print("✅ Levels 1-3 can now share common processing logic")
        print("✅ Configuration-driven approach eliminates code duplication")
        return True
    else:
        print("❌ Some architectural tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)