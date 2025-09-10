"""Level 1 Step 3: Verify single-token department concepts."""

from ..token_verification import verify_single_tokens
from ..wrapper_utils import to_test_path, create_cli_wrapper, validate_input_file


def main(provider="openai"):
    """Main function for Level 1 token verification."""
    input_file = "data/lv1/processed/lv1_s2_filtered.json"
    
    # Validate input file exists
    if not validate_input_file(input_file):
        return {"error": "Input file not found", "file": input_file}
    
    level = 1
    output_file = "data/lv1/final/lv1_final_departments.json"
    metadata_file = "data/lv1/final/lv1_s3_metadata.json"
    
    return verify_single_tokens(input_file, level, output_file, metadata_file, provider)


def test(provider="openai", **kwargs):
    """Test function with small dataset."""
    # Use test paths
    input_file = to_test_path("data/lv1/processed/lv1_s2_filtered.json", 1)
    level = 1
    output_file = to_test_path("data/lv1/final/lv1_final_departments.json", 1)
    metadata_file = to_test_path("data/lv1/final/lv1_s3_metadata.json", 1)
    
    return verify_single_tokens(input_file, level, output_file, metadata_file, provider)


if __name__ == "__main__":
    create_cli_wrapper(1, "s3", main, test)