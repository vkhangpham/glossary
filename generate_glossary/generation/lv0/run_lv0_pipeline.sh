#!/bin/bash

# Run Level 0 Pipeline
# This script runs all steps of the level 0 concept generation pipeline

# Set the provider (default: openai)
PROVIDER=${1:-openai}

echo "Running Level 0 pipeline with provider: $PROVIDER"
echo "================================================"

# Set PYTHONPATH to project root
export PYTHONPATH=.

# Step 0: Extract college names
echo "Step 0: Extracting college names..."
python generate_glossary/generation/lv0/lv0_s0_get_college_names.py
if [ $? -ne 0 ]; then
    echo "Error in Step 0. Exiting."
    exit 1
fi
echo "Step 0 completed successfully."
echo

# Step 1: Extract concepts from college names
echo "Step 1: Extracting concepts from college names..."
python generate_glossary/generation/lv0/lv0_s1_extract_concepts.py --provider $PROVIDER
if [ $? -ne 0 ]; then
    echo "Error in Step 1. Exiting."
    exit 1
fi
echo "Step 1 completed successfully."
echo

# Step 2: Filter concepts by institution frequency
echo "Step 2: Filtering concepts by institution frequency..."
python generate_glossary/generation/lv0/lv0_s2_filter_by_institution_freq.py
if [ $? -ne 0 ]; then
    echo "Error in Step 2. Exiting."
    exit 1
fi
echo "Step 2 completed successfully."
echo

# Step 3: Verify single-token concepts
echo "Step 3: Verifying single-token concepts..."
python generate_glossary/generation/lv0/lv0_s3_verify_single_token.py --provider $PROVIDER
if [ $? -ne 0 ]; then
    echo "Error in Step 3. Exiting."
    exit 1
fi
echo "Step 3 completed successfully."
echo

# Print summary
echo "Level 0 pipeline completed successfully!"
echo "Output file: data/lv0/lv0_s3_verified_concepts.txt"

# Count concepts
NUM_CONCEPTS=$(wc -l < data/lv0/lv0_s3_verified_concepts.txt)
echo "Generated $NUM_CONCEPTS level 0 concepts."
echo "================================================" 