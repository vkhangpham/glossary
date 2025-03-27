import random
import os
import sys
import polars as pl
from collections import Counter
from typing import Dict, List, Tuple
import pandas as pd  # Use pandas to get sheet names

# Add the parent directory of the current file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, project_root)

from generate_glossary.utils.logger import setup_logger

random.seed(42)

logger = setup_logger("lv0.s0")

def extract_college_from_path(path: str) -> str:
    """Extract the college/school/division from a department path."""
    if not path or "->" not in path:
        return None
    
    # Extract the first part before the arrow
    parts = path.split("->")
    if not parts:
        return None
    
    college = parts[0].strip()
    return college.lower() if college else None

def count_colleges_per_institution(df: pl.DataFrame) -> Dict[str, List[str]]:
    """
    Count the number of unique colleges per institution.
    
    Args:
        df: DataFrame containing institution and department_path columns
        
    Returns:
        Dictionary mapping institutions to their list of unique colleges
    """
    institution_colleges = {}
    
    # Group by institution
    for institution in df["institution"].unique():
        # Filter rows for this institution
        inst_df = df.filter(pl.col("institution") == institution)
        
        # Extract colleges from department paths
        colleges = []
        for path in inst_df["department_path"]:
            college = extract_college_from_path(path)
            if college:
                colleges.append(college)
        
        # Store unique colleges for this institution
        institution_colleges[institution.strip().lower()] = list(set(colleges))
    
    return institution_colleges

def select_top_institutions(
    institution_colleges: Dict[str, List[str]], 
    top_n: int = 30
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Select top N institutions with the most colleges.
    
    Args:
        institution_colleges: Dictionary mapping institutions to their colleges
        top_n: Number of top institutions to select
        
    Returns:
        Tuple of (filtered dictionary, list of selected institutions)
    """
    # Count colleges per institution
    college_counts = {inst: len(colleges) for inst, colleges in institution_colleges.items()}
    
    # Sort institutions by college count (descending)
    sorted_institutions = sorted(
        college_counts.keys(), 
        key=lambda x: college_counts[x], 
        reverse=True
    )
    
    # Select top N institutions
    selected_institutions = sorted_institutions[:top_n]
    
    # Filter dictionary to include only selected institutions
    filtered_dict = {
        inst: institution_colleges[inst] 
        for inst in selected_institutions
    }
    
    return filtered_dict, selected_institutions

if __name__ == "__main__":
    logger.info("Starting college names extraction")
    
    try:
        logger.info("Reading Excel file...")
        
        # Get all sheet names from the Excel file using pandas
        excel_file = "data/Faculty Extraction Report.xlsx"
        try:
            # Using pandas to read sheet names
            xl = pd.ExcelFile(excel_file)
            all_sheets = xl.sheet_names
            logger.info(f"Found {len(all_sheets)} sheets in the Excel file: {', '.join(all_sheets)}")
        except Exception as e:
            logger.error(f"Failed to read sheet names with pandas: {str(e)}")
            # Fallback to known sheet names if pandas fails
            all_sheets = ["US-R1-Top20", "US-R1-Top20-50", "US-R1-Top50-100", "US-R1-Top100-end"]
            logger.info(f"Using fallback sheet list: {', '.join(all_sheets)}")
        
        # Read all relevant sheets and combine them
        main_df = None
        valid_dfs_count = 0
        
        for sheet in all_sheets:
            try:
                # Only process sheets with 'R1' in their name
                if "R1" not in sheet:
                    logger.info(f"Skipping sheet: {sheet} (does not contain 'R1')")
                    continue
                
                # Skip sheets that don't contain department data
                if "summary" in sheet.lower() or "logs" in sheet.lower() or "eval" in sheet.lower():
                    logger.info(f"Skipping sheet: {sheet}")
                    continue
                
                sheet_df = pl.read_excel(excel_file, sheet_name=sheet)
                
                # Check if sheet has the required columns
                if "institution" not in sheet_df.columns or "department_path" not in sheet_df.columns:
                    logger.info(f"Skipping sheet {sheet} - missing required columns")
                    continue
                
                # Select only the columns we need to avoid concatenation issues
                sheet_df = sheet_df.select(["institution", "department_path"])
                
                logger.info(f"Read {len(sheet_df)} rows from sheet: {sheet}")
                
                # If this is the first valid DataFrame, set it as the main one
                if main_df is None:
                    main_df = sheet_df
                else:
                    # Otherwise, append to the main DataFrame
                    main_df = pl.concat([main_df, sheet_df])
                
                valid_dfs_count += 1
                
            except Exception as e:
                logger.warning(f"Could not read or process sheet {sheet}: {str(e)}")
                continue
        
        # Check if we have any valid data
        if valid_dfs_count == 0:
            logger.error("No valid sheets found with department data")
            sys.exit(1)
            
        logger.info(f"Combined data from {valid_dfs_count} sheets, total {len(main_df)} rows")
        
        # Remove any rows with missing department_path
        main_df = main_df.filter(pl.col("department_path").is_not_null())
        logger.info(f"After removing rows with missing department_path: {len(main_df)} rows")
        
        # Count colleges per institution
        institution_colleges = count_colleges_per_institution(main_df)
        logger.info(f"Found colleges for {len(institution_colleges)} institutions")
        
        # Select top 30 institutions with most colleges
        top_institutions, selected_inst_list = select_top_institutions(
            institution_colleges, 
            top_n=30
        )
        
        # Log college counts for selected institutions
        for inst in selected_inst_list:
            logger.info(f"{inst}: {len(top_institutions[inst])} colleges")
        
        # Prepare output format: "institution - college"
        college_entries = []
        for inst, colleges in top_institutions.items():
            for college in colleges:
                college_entries.append(f"{inst} - {college}")
        
        # Shuffle entries
        random.shuffle(college_entries)
        
        # Save to output file
        output_path = "data/lv0/lv0_s0_college_names.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as file:
            for entry in college_entries:
                file.write(entry + "\n")
                
        logger.info(f"Successfully wrote {len(college_entries)} college entries to {output_path}")
        
        # Save metadata about selected institutions
        metadata_path = "data/lv0/lv0_s0_metadata.json"
        import json
        
        metadata = {
            "selected_institutions": selected_inst_list,
            "institution_college_counts": {
                inst: len(colleges) for inst, colleges in top_institutions.items()
            },
            "total_colleges": len(college_entries)
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Successfully wrote metadata to {metadata_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    
    logger.info("College names extraction completed") 