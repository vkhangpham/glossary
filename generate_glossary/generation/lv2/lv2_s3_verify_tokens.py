"""Level 2 Step 3: Verify single-token research area concepts."""

from ..token_verification import verify_single_tokens
from ..wrapper_utils import to_test_path, create_cli_wrapper, validate_input_file


def main(provider: str = "openai") -> None:
    """Main function for Level 2 token verification."""
    input_file = "data/lv2/processed/lv2_s2_filtered.json"
    
    # Validate input file exists
    validation_result = validate_input_file(input_file)
    if validation_result is False:
        return {"error": "Input file not found", "file": input_file}
    elif isinstance(validation_result, dict):
        return validation_result
    
    level = 2
    output_file = "data/lv2/final/lv2_final_research_areas.json"
    metadata_file = "data/lv2/final/lv2_s3_metadata.json"
    
    return verify_single_tokens(input_file, level, output_file, metadata_file, provider)


def test(provider="openai", **kwargs):
    """Test function with small dataset."""
    # Use test paths
    input_file = to_test_path("data/lv2/processed/lv2_s2_filtered.json", 2)
    level = 2
    output_file = to_test_path("data/lv2/final/lv2_final_research_areas.json", 2)
    metadata_file = to_test_path("data/lv2/final/lv2_s3_metadata.json", 2)
    
    return verify_single_tokens(input_file, level, output_file, metadata_file, provider)


if __name__ == "__main__":
    create_cli_wrapper(2, "s3", main, test)