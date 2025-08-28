# Unified Context Schema

This document describes the schema used for transferring data between detectors and the splitter in the sense disambiguation pipeline.

## Overview

The unified context schema provides a standardized format for all ambiguity detectors to communicate their findings to the splitter. It follows a two-level structure:

1. A top-level map of terms to their context information
2. Each term context contains general metadata and a list of evidence blocks from different detectors

## Schema Definition

```jsonc
{
  "<term1>": {                      // Term string as key (e.g., "machine_learning")
    "canonical_name": "string",     // Canonical form of the term
    "level": 2,                     // Hierarchy level (0-3) or null if unknown
    "overall_confidence": 0.83,     // Aggregated confidence score [0-1]
    "evidence": [                   // List of evidence blocks from various detectors
      {
        "source": "resource_cluster",       // Detector source
        "detector_version": "2025.05.22",   // Version string
        "confidence": 0.72,                 // Per-detector confidence [0-1]
        "metrics": {                        // Numeric metrics (optional)
          "separation_score": 0.42,
          "silhouette_score": 0.31
        },
        "payload": {                        // Detector-specific data (schema varies by source)
          "cluster_labels": [0, 1, 0, -1],  // Resource clustering example
          "eps": 0.4,
          "min_samples": 2
        }
      },
      {
        "source": "parent_context",
        "detector_version": "2025.05.22",
        "confidence": 0.65,
        "metrics": {
          "distinct_ancestors_count": 2
        },
        "payload": {
          "divergent": true,
          "parents": ["artificial_intelligence", "computer_science"],
          "distinct_ancestors": [["science", "computer_science"], ["mathematics", "statistics"]]
        }
      }
    ]
  },
  "<term2>": { /* ... */ }
}
```

## Source-Specific Payloads

Each detector provides a specific payload format:

### 1. Resource Cluster Detector

```jsonc
{
  "cluster_labels": [0, 1, 0, -1],        // Integer labels for each resource
  "eps": 0.4,                             // DBSCAN eps parameter
  "min_samples": 2,                       // DBSCAN min_samples parameter
  "cluster_details": {                    // Optional: detailed resource info
    "0": [                                // Cluster ID → resources
      {
        "url": "https://example.com/resource1",
        "processed_content": "truncated content..."
      }
    ],
    "1": [ /* ... */ ]
  }
}
```

### 2. Parent Context Detector

```jsonc
{
  "divergent": true,                      // Whether parent contexts diverge
  "parents": ["ai", "algorithms"],        // Direct parent terms
  "distinct_ancestors": [                 // Distinct lineage paths
    ["science", "computer_science", "ai"],
    ["mathematics", "algorithms"]
  ]
}
```

### 3. Radial Polysemy Detector

```jsonc
{
  "polysemy_index": 0.78,                 // Polysemy score [0-1]
  "context_count": 15,                    // Number of contexts analyzed
  "sample_contexts": [                    // Example contexts (≤5)
    "When training the model, we need to...",
    "In cognitive psychology, mental models represent..."
  ]
}
```

## File Format

The context information is stored in a JSON file with the following naming convention:

```
unified_context_YYYYMMDD_HHMMSS.json
```

This file is produced by the Hybrid Detector and consumed by the Splitter. 