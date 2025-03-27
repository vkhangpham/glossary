# Level 0 Concept Generation

This package generates level 0 concepts, which represent broad academic disciplines extracted from college/school/division names.

## Overview

Level 0 represents the highest level in the concept hierarchy, focusing on broad academic disciplines like "humanities", "sciences", "engineering", etc. These concepts are extracted from college/school/division names from top academic institutions.

## Pipeline Steps

The level 0 generation pipeline consists of four main steps:

1. **College Name Extraction** (`lv0_s0_get_college_names.py`):
   - Extracts college/school/division names from the "Faculty Extraction Report.xlsx" file
   - Selects the top 30 institutions with the most colleges
   - Outputs college names in the format "institution - college"

2. **Concept Extraction** (`lv0_s1_extract_concepts.py`):
   - Uses LLM to extract broad academic disciplines from college names
   - Implements parallel processing for efficiency
   - Applies a frequency threshold to filter concepts

3. **Institution Frequency Filtering** (`lv0_s2_filter_by_institution_freq.py`):
   - Filters concepts based on their appearance across institutions
   - Requires concepts to appear in at least 60% of the selected institutions

4. **Single-Token Verification** (`lv0_s3_verify_single_token.py`):
   - Verifies single-word concepts using LLM
   - Automatically passes multi-word concepts through without verification
   - Normalizes concepts to remove duplicates

## Usage

Run the scripts in sequence:

```bash
# Step 0: Extract college names
python generate_glossary/generation/lv0/lv0_s0_get_college_names.py

# Step 1: Extract concepts from college names
python generate_glossary/generation/lv0/lv0_s1_extract_concepts.py [--provider openai|gemini]

# Step 2: Filter concepts by institution frequency
python generate_glossary/generation/lv0/lv0_s2_filter_by_institution_freq.py

# Step 3: Verify single-token concepts
python generate_glossary/generation/lv0/lv0_s3_verify_single_token.py [--provider openai|gemini]
```

## Output

The final output is a list of verified level 0 concepts in `data/lv0/lv0_s3_verified_concepts.txt`.

Detailed metadata for each step is stored in the corresponding JSON files in the `data/lv0/` directory. 