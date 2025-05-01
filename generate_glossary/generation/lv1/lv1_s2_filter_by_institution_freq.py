import json
from typing import Dict, List, Any, Set, Counter as CounterType
import sys
import os
import csv
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from generate_glossary.utils.logger import setup_logger
from generate_glossary.deduplicator.dedup_utils import normalize_text

# Setup logging
logger = setup_logger("lv1.s2")

class Config:
    """Configuration for concept filtering"""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    INPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s1_department_concepts.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s2_filtered_concepts.txt")
    VALIDATION_META_FILE = os.path.join(BASE_DIR, "data/lv1/raw/lv1_s2_metadata.json")
    PLOTS_DIR = os.path.join(BASE_DIR, "data/lv1/raw/plots")
    
    # Filtering thresholds
    MIN_COLLEGE_APPEARANCE = 1  # Concept must appear in at least this many colleges
    MIN_COLLEGE_FREQ_PERCENT = 1  # Concept must appear in at least this % of departments within a college
    
    # Concept validation
    MIN_CONCEPT_LENGTH = 3  # Minimum length of a valid concept
    MAX_CONCEPT_LENGTH = 50  # Maximum length of a valid concept
    
    # Visualization
    MAX_CONCEPTS_PER_PLOT = 30  # Maximum number of concepts to show in distribution plots

def is_valid_concept(concept: str) -> bool:
    """
    Check if a concept is valid based on basic criteria
    
    Args:
        concept: The concept string to validate
        
    Returns:
        Boolean indicating if the concept is valid
    """
    # Check length criteria
    if not concept or len(concept) < Config.MIN_CONCEPT_LENGTH or len(concept) > Config.MAX_CONCEPT_LENGTH:
        return False
        
    # Exclude single characters and numbers
    if len(concept) <= 2 and (concept.isdigit() or concept.isalpha()):
        return False
        
    # Exclude concepts that are likely not academic terms
    non_academic_terms = [
        "page", "home", "about", "contact", "staff", "faculty", "links",
        "click", "here", "website", "portal", "login", "apply", "apply now",
        "register", "registration", "more", "learn more", "read more",
        "back", "next", "previous", "link", "site"
    ]
    
    if concept.lower() in non_academic_terms:
        return False
        
    # Check if concept has more than just punctuation and spaces
    if not any(c.isalnum() for c in concept):
        return False
        
    return True

def count_department_frequencies(input_file: str) -> tuple[Dict[str, int], Dict[str, str], Dict[str, CounterType[str]], Dict[str, Set[str]]]:
    """
    Count how many departments each concept appears in and track college-level statistics
    
    Args:
        input_file: Path to CSV file containing department-concept mappings
        
    Returns:
        Tuple containing:
        - Dictionary mapping concepts to their total department frequency
        - Dictionary mapping normalized forms to original forms
        - Dictionary mapping concepts to their frequency per college
        - Dictionary mapping concepts to the set of colleges they appear in
    """
    # Use Counter for efficient counting
    concept_counts = Counter()
    college_concept_counts = defaultdict(Counter)  # college -> concept -> count
    concept_colleges = defaultdict(set)  # concept -> set of colleges
    
    # Normalize concepts and track original forms
    normalized_to_original = {}
    
    # Track skipped college-name concepts for logging
    skipped_college_concepts = set()
    
    # Read CSV file using csv module to handle quoted fields properly
    with open(input_file, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        
        # Skip header
        next(csv_reader)
        
        # Process each line
        for row in csv_reader:
            if len(row) >= 3:  # Ensure we have at least department, college, concept
                department, college, concept = row[0], row[1], row[2]
                
                # Skip invalid concepts
                if not is_valid_concept(concept):
                    continue
                    
                # Normalize the concept and college
                norm_concept = normalize_text(concept)
                norm_college = normalize_text(college)
                
                # Extract college name without "college of" prefix
                college_name = norm_college.replace("college of ", "").strip()
                
                # Skip if concept matches college name
                # Also check if concept is part of a compound college name
                # e.g., "law" in "college of law and policy"
                if (norm_concept == college_name or 
                    any(norm_concept == part.strip() for part in college_name.split(" and "))):
                    skipped_college_concepts.add((concept, college))
                    continue
                
                # Track the original form (prefer shorter versions)
                if norm_concept in normalized_to_original:
                    current = normalized_to_original[norm_concept]
                    if len(concept) < len(current):
                        normalized_to_original[norm_concept] = concept
                else:
                    normalized_to_original[norm_concept] = concept
                    
                # Update counters
                concept_counts[norm_concept] += 1
                college_concept_counts[norm_college][norm_concept] += 1
                concept_colleges[norm_concept].add(norm_college)
    
    # Log skipped concepts
    if skipped_college_concepts:
        logger.info("\nSkipped concepts that match their college names:")
        for concept, college in sorted(skipped_college_concepts):
            logger.info(f"  Skipped '{concept}' from {college}")
    
    return dict(concept_counts), normalized_to_original, dict(college_concept_counts), dict(concept_colleges)

def plot_concept_distribution(
    concept: str,
    college_concept_counts: Dict[str, CounterType[str]],
    college_dept_counts: Dict[str, int],
    output_dir: str
):
    """
    Create a bar plot showing the distribution of a concept across colleges
    
    Args:
        concept: The concept to plot
        college_concept_counts: Dictionary mapping colleges to their concept frequencies
        college_dept_counts: Dictionary mapping colleges to their total department count
        output_dir: Directory to save the plot
    """
    # Calculate percentage in each college
    college_percentages = []
    colleges = []
    
    for college, concept_counts in college_concept_counts.items():
        if concept in concept_counts and college in college_dept_counts:
            percentage = (concept_counts[concept] / college_dept_counts[college]) * 100
            college_percentages.append(percentage)
            colleges.append(college.replace('college of ', '').title())
    
    if not colleges:
        return
        
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(colleges, college_percentages)
    plt.title(f'Distribution of "{concept}" Across Colleges')
    plt.xlabel('College')
    plt.ylabel('Percentage of Departments (%)')
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'concept_dist_{concept.replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()

def plot_top_concepts_per_college(
    college: str,
    concept_counts: CounterType[str],
    total_departments: int,
    output_dir: str,
    max_concepts: int = 20
):
    """
    Create a bar plot showing the distribution of top concepts in a college
    
    Args:
        college: The college name
        concept_counts: Counter of concept frequencies for this college
        total_departments: Total number of departments in this college
        output_dir: Directory to save the plot
        max_concepts: Maximum number of concepts to show
    """
    if not concept_counts:
        return
        
    # Get top concepts
    top_concepts = concept_counts.most_common(max_concepts)
    concepts, counts = zip(*top_concepts)
    
    # Calculate percentages
    percentages = [(count / total_departments) * 100 for count in counts]
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(concepts)), percentages)
    plt.title(f'Top Concepts in {college.replace("college of ", "").title()}')
    plt.xlabel('Concept')
    plt.ylabel('Percentage of Departments (%)')
    plt.xticks(range(len(concepts)), concepts, rotation=45, ha='right')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'college_concepts_{college.replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()

def filter_by_college_distribution(
    concept_frequencies: Dict[str, int],
    college_concept_counts: Dict[str, CounterType[str]],
    concept_colleges: Dict[str, Set[str]],
    normalized_to_original: Dict[str, str]
) -> List[str]:
    """
    Filter concepts based on their distribution across colleges
    
    Args:
        concept_frequencies: Dictionary mapping concepts to their total frequency
        college_concept_counts: Dictionary mapping colleges to their concept frequencies
        concept_colleges: Dictionary mapping concepts to the set of colleges they appear in
        normalized_to_original: Dictionary mapping normalized forms to original forms
        
    Returns:
        List of concepts that meet the college distribution criteria
    """
    filtered_concepts = []
    
    # Calculate total departments per college
    college_dept_counts = {
        college: sum(counts.values())
        for college, counts in college_concept_counts.items()
    }
    
    # Create plots directory
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    
    # Plot distribution for each college
    for college, concept_counts in tqdm(college_concept_counts.items(), desc="Generating college plots"):
        plot_top_concepts_per_college(
            college,
            concept_counts,
            college_dept_counts[college],
            Config.PLOTS_DIR
        )
    
    # Filter and plot concepts
    for norm_concept, freq in tqdm(concept_frequencies.items(), desc="Filtering concepts by college distribution"):
        colleges = concept_colleges[norm_concept]
        
        # Check if concept appears in enough colleges
        if len(colleges) >= Config.MIN_COLLEGE_APPEARANCE:
            # Check frequency within each college
            college_percentages = []
            for college in colleges:
                if college in college_dept_counts and college_dept_counts[college] > 0:
                    concept_freq = college_concept_counts[college][norm_concept]
                    percentage = (concept_freq / college_dept_counts[college]) * 100
                    college_percentages.append(percentage)
            
            # Concept must have significant presence in at least one college
            if any(pct >= Config.MIN_COLLEGE_FREQ_PERCENT for pct in college_percentages):
                orig_concept = normalized_to_original.get(norm_concept, norm_concept)
                filtered_concepts.append(orig_concept)
                
                # Plot distribution for this concept
                plot_concept_distribution(
                    orig_concept,
                    college_concept_counts,
                    college_dept_counts,
                    Config.PLOTS_DIR
                )
    
    return sorted(filtered_concepts)

def ensure_dirs_exist():
    """Ensure all required directories exist"""
    dirs_to_create = [
        os.path.dirname(Config.OUTPUT_FILE),
        os.path.dirname(Config.VALIDATION_META_FILE)
    ]
    
    for directory in dirs_to_create:
        try:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise

def main():
    """Main execution function"""
    try:
        logger.info("Starting concept filtering by department frequency")
        
        # Ensure directories exist
        ensure_dirs_exist()
        
        # Count concept frequencies across departments and colleges
        concept_frequencies, normalized_to_original, college_concept_counts, concept_colleges = count_department_frequencies(Config.INPUT_FILE)
        logger.info(f"Counted frequencies for {len(concept_frequencies)} unique normalized concepts")
        
        # Filter concepts based on college distribution
        filtered_concepts = filter_by_college_distribution(
            concept_frequencies,
            college_concept_counts,
            concept_colleges,
            normalized_to_original
        )
        
        # Log frequency distribution
        freq_dist = Counter(concept_frequencies.values())
        logger.info("Department frequency distribution:")
        for freq, count in sorted(freq_dist.items()):
            logger.info(f"  {count} concepts appear in {freq} departments")
            
        # Log college distribution
        logger.info("\nCollege distribution:")
        for college, concept_counts in college_concept_counts.items():
            logger.info(f"  {college}: {len(concept_counts)} concepts in {sum(concept_counts.values())} departments")
        
        logger.info(f"\nFiltered to {len(filtered_concepts)} concepts")
        
        # Create output directories if needed
        for path in [Config.OUTPUT_FILE, Config.VALIDATION_META_FILE, Config.PLOTS_DIR]:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save filtered concepts
        with open(Config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            for concept in filtered_concepts:
                f.write(f"{concept}\n")
        
        # Calculate college-level statistics for filtered concepts
        college_stats = {}
        for college, concept_counts in college_concept_counts.items():
            total_deps = sum(concept_counts.values())
            filtered_concepts_in_college = sum(1 for c in concept_counts if c in filtered_concepts)
            college_stats[college] = {
                "total_departments": total_deps,
                "total_concepts": len(concept_counts),
                "filtered_concepts": filtered_concepts_in_college,
                "top_concepts": [
                    {
                        "concept": c,
                        "frequency": f,
                        "percentage": (f / total_deps * 100) if total_deps > 0 else 0
                    }
                    for c, f in concept_counts.most_common(Config.MAX_CONCEPTS_PER_PLOT)
                    if c in filtered_concepts
                ]
            }
        
        # Save validation metadata
        validation_metadata = {
            "metadata": {
                "output_count": len(filtered_concepts),
                "normalized_concepts_count": len(concept_frequencies),
                "min_college_appearance": Config.MIN_COLLEGE_APPEARANCE,
                "min_college_freq_percent": Config.MIN_COLLEGE_FREQ_PERCENT,
                "min_concept_length": Config.MIN_CONCEPT_LENGTH,
                "max_concept_length": Config.MAX_CONCEPT_LENGTH,
                "frequency_distribution": {
                    str(k): v for k, v in freq_dist.items()
                }
            },
            "college_statistics": college_stats,
            "concept_college_distribution": {
                concept: {
                    "total_colleges": len(concept_colleges.get(normalize_text(concept), [])),
                    "colleges": sorted(list(concept_colleges.get(normalize_text(concept), [])))
                }
                for concept in filtered_concepts
            }
        }
        
        with open(Config.VALIDATION_META_FILE, "w", encoding="utf-8") as f:
            json.dump(validation_metadata, f, indent=4, ensure_ascii=False)

        logger.info("Concept filtering completed successfully")
        logger.info(f"Plots have been saved to {Config.PLOTS_DIR}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 