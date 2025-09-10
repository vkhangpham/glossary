"""Level 3 Step 2: Filter conference topic concepts by frequency."""

from ..frequency_filtering import filter_by_frequency
from ..wrapper_utils import to_test_path, create_cli_wrapper, validate_input_file


def main(provider="openai", min_frequency=None, threshold_percent=None):
    """Main function for Level 3 frequency filtering."""
    input_file = "data/lv3/processed/lv3_s1_conference_topics.json"
    
    # Validate input file exists
    if not validate_input_file(input_file):
        return {"error": "Input file not found", "file": input_file}
    
    # Validate parameters
    if min_frequency is not None and (not isinstance(min_frequency, int) or min_frequency < 1):
        return {"error": "Invalid parameter", "param": "min_frequency", "value": min_frequency}
    
    if threshold_percent is not None and (not isinstance(threshold_percent, (int, float)) or not (0 <= threshold_percent <= 100)):
        return {"error": "Invalid parameter", "param": "threshold_percent", "value": threshold_percent}
    
    level = 3
    output_file = "data/lv3/processed/lv3_s2_filtered.json"
    metadata_file = "data/lv3/processed/lv3_s2_metadata.json"
    
    return filter_by_frequency(input_file, level, output_file, metadata_file, min_frequency, threshold_percent)


def test(provider="openai", **kwargs):
    """Test function with small dataset."""
    # Use test paths
    input_file = to_test_path("data/lv3/processed/lv3_s1_conference_topics.json", 3)
    level = 3
    output_file = to_test_path("data/lv3/processed/lv3_s2_filtered.json", 3)
    metadata_file = to_test_path("data/lv3/processed/lv3_s2_metadata.json", 3)
    
    return filter_by_frequency(input_file, level, output_file, metadata_file, None, None)


if __name__ == "__main__":
    def add_args(parser):
        parser.add_argument("--min-frequency", type=int, help="Minimum frequency")
        parser.add_argument("--threshold-percent", type=float, help="Threshold percentage")
    
    create_cli_wrapper(3, "s2", main, test, add_args)