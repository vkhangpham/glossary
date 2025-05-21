#!/bin/bash
# Script to set up the necessary data directories for sense_disambiguation

# Base directory
BASE_DIR="sense_disambiguation/data"

# Create base directory if it doesn't exist
mkdir -p "$BASE_DIR"

# Create subdirectories
mkdir -p "$BASE_DIR/ambiguity_detection_results"
mkdir -p "$BASE_DIR/sense_disambiguation_results"
mkdir -p "$BASE_DIR/global_clustering_results"
mkdir -p "$BASE_DIR/final"
mkdir -p "$BASE_DIR/vector_store"
mkdir -p "$BASE_DIR/corpus_cache"

# Create level-specific final directories 
for level in {0..3}; do
  mkdir -p "$BASE_DIR/final/lv$level"
done

echo "Created data directories for sense_disambiguation"
echo "Note: Input files like hierarchy.json should remain in /data/ directory" 