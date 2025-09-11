import random
import os
import sys
import json
import polars as pl
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm

from generate_glossary.utils.logger import get_logger

TOP_N_INSTITUTIONS = 30  # Number of top institutions to select
EXCEL_FILE = "data/Faculty Extraction Report.xlsx"  # Source Excel file
OUTPUT_DIR = "data/generation/lv0"  # Output directory
OUTPUT_FILE = "lv0_s0_output.txt"  # Output filename
METADATA_FILE = "lv0_s0_metadata.json"  # Metadata filename
RANDOM_SEED = 42  # For reproducible shuffling

# Test mode paths
TEST_OUTPUT_DIR = "data/generation/tests"  # Test output directory
TEST_TOP_N = 3  # 10% of 30 institutions for test mode

SHEETS_TO_PROCESS = [
    "US-R1-Top20",
    "US-R1-Top20-50",
    "US-R1-Top50-100",
    "US-R1-Top100-end",
    "Quantity US-R1-Top20",
]

random.seed(RANDOM_SEED)

logger = get_logger("lv0.s0")


def count_colleges_per_institution(df: pl.DataFrame) -> Dict[str, List[str]]:
    """
    Count the number of unique colleges per institution.

    Optimized version using Polars group_by for O(n) performance.

    Args:
        df: DataFrame containing institution and department_path columns

    Returns:
        Dictionary mapping institutions to their list of unique colleges
    """
    if "institution" not in df.columns or "department_path" not in df.columns:
        logger.error("Missing required columns: institution or department_path")
        return {}

    result = (
        df.select(["institution", "department_path"])
        .filter(pl.col("department_path").is_not_null())
        .with_columns(
            pl.col("department_path")
            .str.split("->")
            .list.get(0)
            .str.strip_chars()
            .str.to_lowercase()
            .alias("college")
        )
        .filter(pl.col("college").is_not_null() & (pl.col("college") != ""))
        .group_by("institution")
        .agg(pl.col("college").unique().alias("colleges"))
        .with_columns(pl.col("institution").str.strip_chars().str.to_lowercase())
    )

    institution_colleges = {}
    for row in result.iter_rows(named=True):
        institution_colleges[row["institution"]] = sorted(row["colleges"])

    total_paths = len(df)
    valid_paths = len(df.filter(pl.col("department_path").str.contains("->")))
    logger.info(f"Processed {valid_paths}/{total_paths} valid department paths")

    return institution_colleges


def select_top_institutions(
    institution_colleges: Dict[str, List[str]], top_n: int = TOP_N_INSTITUTIONS
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Select top N institutions with the most colleges.

    Args:
        institution_colleges: Dictionary mapping institutions to their colleges
        top_n: Number of top institutions to select

    Returns:
        Tuple of (filtered dictionary, list of selected institutions)
    """
    college_counts = {
        inst: len(colleges) for inst, colleges in institution_colleges.items()
    }

    sorted_institutions = sorted(
        college_counts.keys(), key=lambda x: college_counts[x], reverse=True
    )

    selected_institutions = sorted_institutions[:top_n]
    filtered_dict = {inst: institution_colleges[inst] for inst in selected_institutions}

    return filtered_dict, selected_institutions


def test(provider="openai", **kwargs):
    """Test mode: Process 10% of data and save to test directory"""
    global OUTPUT_DIR, TOP_N_INSTITUTIONS

    # Save original values
    original_output_dir = OUTPUT_DIR
    original_top_n = TOP_N_INSTITUTIONS

    # Set test values
    OUTPUT_DIR = TEST_OUTPUT_DIR
    TOP_N_INSTITUTIONS = TEST_TOP_N

    logger.info("Running in TEST MODE")

    try:
        # Run main with test settings
        main()
    finally:
        # Restore original values
        OUTPUT_DIR = original_output_dir
        TOP_N_INSTITUTIONS = original_top_n


def main():
    """Main function to extract college names from Excel file."""
    logger.info("Starting college names extraction")
    logger.info(f"Configuration: top_n={TOP_N_INSTITUTIONS}, seed={RANDOM_SEED}")

    excel_file = EXCEL_FILE

    if not Path(excel_file).exists():
        logger.error(f"Excel file not found: {excel_file}")
        logger.info(
            "Please ensure 'Faculty Extraction Report.xlsx' exists in the data/ directory"
        )
        sys.exit(1)

    try:
        logger.info(f"Reading Excel file from: {excel_file}")
        logger.info(
            f"Processing {len(SHEETS_TO_PROCESS)} sheets: {', '.join(SHEETS_TO_PROCESS)}"
        )

        main_df = None

        for sheet in tqdm(SHEETS_TO_PROCESS, desc="Processing Excel sheets"):
            try:
                sheet_df = pl.read_excel(excel_file, sheet_name=sheet)
                sheet_df = sheet_df.select(["institution", "department_path"])
                logger.info(f"Read {len(sheet_df)} rows from sheet: {sheet}")

                if main_df is None:
                    main_df = sheet_df
                else:
                    main_df = pl.concat([main_df, sheet_df])

            except Exception as e:
                logger.warning(f"Could not read sheet {sheet}: {str(e)}")
                continue

        if main_df is None:
            logger.error("No data could be read from Excel sheets")
            sys.exit(1)

        logger.info(f"Combined data: total {len(main_df)} rows")

        main_df = main_df.filter(pl.col("department_path").is_not_null())
        logger.info(
            f"After removing rows with missing department_path: {len(main_df)} rows"
        )

        logger.info("Extracting colleges from department paths...")
        institution_colleges = count_colleges_per_institution(main_df)
        logger.info(f"Found colleges for {len(institution_colleges)} institutions")

        college_counts = [len(colleges) for colleges in institution_colleges.values()]
        if college_counts:
            avg_colleges = sum(college_counts) / len(college_counts)
            logger.info(f"Average colleges per institution: {avg_colleges:.1f}")
            logger.info(
                f"Max colleges: {max(college_counts)}, Min colleges: {min(college_counts)}"
            )

        top_institutions, selected_inst_list = select_top_institutions(
            institution_colleges, top_n=TOP_N_INSTITUTIONS
        )
        logger.info(f"Selected top {TOP_N_INSTITUTIONS} institutions")

        for inst in selected_inst_list:
            logger.info(f"{inst}: {len(top_institutions[inst])} colleges")

        college_entries = []
        for inst, colleges in top_institutions.items():
            for college in colleges:
                college_entries.append(f"{inst} - {college}")

        random.shuffle(college_entries)

        output_path = Path(OUTPUT_DIR) / OUTPUT_FILE
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(output_path, "w") as file:
            for entry in college_entries:
                file.write(entry + "\n")

        logger.info(
            f"Successfully wrote {len(college_entries)} college entries to {output_path}"
        )

        metadata_path = Path(OUTPUT_DIR) / METADATA_FILE

        metadata = {
            "selected_institutions": selected_inst_list,
            "institution_college_counts": {
                inst: len(colleges) for inst, colleges in top_institutions.items()
            },
            "total_colleges": len(college_entries),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Successfully wrote metadata to {metadata_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

    logger.info("College names extraction completed")


if __name__ == "__main__":
    main()
