"""Level 2 Step 2: Filter research area concepts by frequency."""

from ..frequency_filtering import filter_by_frequency
from ..wrapper_utils import to_test_path, create_cli_wrapper, validate_input_file


def main(provider="openai", min_frequency=None, threshold_percent=None):
    """Main function for Level 2 frequency filtering."""
    input_file = "data/lv2/processed/lv2_s1_research_areas.json"
    
    # Validate input file exists
    if not validate_input_file(input_file):
        return {"error": "Input file not found", "file": input_file}
    
    level = 2
    output_file = "data/lv2/processed/lv2_s2_filtered.json"
    metadata_file = "data/lv2/processed/lv2_s2_metadata.json"
    
    return filter_by_frequency(input_file, level, output_file, metadata_file, min_frequency, threshold_percent)


def test(provider="openai", **kwargs):
    """Test function with small dataset."""
    # Use test paths
    input_file = to_test_path("data/lv2/processed/lv2_s1_research_areas.json", 2)
    level = 2
    output_file = to_test_path("data/lv2/processed/lv2_s2_filtered.json", 2)
    metadata_file = to_test_path("data/lv2/processed/lv2_s2_metadata.json", 2)
    
    return filter_by_frequency(input_file, level, output_file, metadata_file, None, None)


if __name__ == "__main__":
    def add_args(parser):
        parser.add_argument("--min-frequency", type=int, help="Minimum frequency")
        parser.add_argument("--threshold-percent", type=float, help="Threshold percentage")
    
    create_cli_wrapper(2, "s2", main, test, add_args)