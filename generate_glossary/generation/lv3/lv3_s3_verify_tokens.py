"""Level 3 Step 3: Verify single-token conference topic concepts."""

from ..token_verification import verify_single_tokens
from ..wrapper_utils import to_test_path, create_cli_wrapper, handle_dry_run, validate_input_file


def main(provider="openai", dry_run=False):
    """Main function for Level 3 token verification."""
    input_file = "data/lv3/processed/lv3_s2_filtered.json"
    
    # Validate input file exists
    if not validate_input_file(input_file):
        return {"error": "Input file not found", "file": input_file}
    
    level = 3
    output_file = "data/lv3/final/lv3_final_conference_topics.json"
    metadata_file = "data/lv3/final/lv3_s3_metadata.json"
    
    return handle_dry_run(
        dry_run,
        verify_single_tokens,
        input_file, level, output_file, metadata_file, provider
    )


def test(provider="openai", **kwargs):
    """Test function with small dataset."""
    # Use test paths
    input_file = to_test_path("data/lv3/processed/lv3_s2_filtered.json", 3)
    level = 3
    output_file = to_test_path("data/lv3/final/lv3_final_conference_topics.json", 3)
    metadata_file = to_test_path("data/lv3/final/lv3_s3_metadata.json", 3)
    
    return verify_single_tokens(input_file, level, output_file, metadata_file, provider)


if __name__ == "__main__":
    def add_args(parser):
        parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    create_cli_wrapper(3, "s3", main, test, add_args)