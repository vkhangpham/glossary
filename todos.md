# TODOS

## GEPA Prompt Optimization Issues

### Problem Description
GEPA optimization for Level 0 Step 1 (concept extraction) improved F1-score by 21.1% but introduced systematic semantic errors in academic term extraction.

### Critical Issues Found

#### 1. Semantic Context Loss
- **Issue**: Shared nouns not properly distributed across compound terms
- **Example**: "development, regeneration, and stem cell biology" → ["development", "regeneration", "stem cell biology"]
- **Expected**: ["developmental biology", "regenerative biology", "stem cell biology"]
- **Impact**: Missing biological context for first two terms

#### 2. Invalid Term Formation
- **Issue**: Incorrect pairing of terms in compound structures
- **Example**: "systems, populations and leadership" → ["systems leadership", "populations leadership"]
- **Expected**: ["systems science", "population science", "leadership studies"]
- **Impact**: Creates non-existent academic fields

#### 3. Linguistic Incorrectness
- **Issue**: Morphologically invalid terms generated
- **Example**: "regeneration biology" instead of "regenerative biology"
- **Impact**: Produces linguistically incorrect academic terminology

### Root Cause
Pure F1-score optimization without semantic validation constraints. The metric rewards quantity of extracted concepts over semantic correctness.

### Affected Files
- `data/generation/tests/lv0_s1_metadata.json` (optimized version with issues)
- `data/generation/tests/lv0_s1_metadata_subop.json` (sub-optimal but semantically better)

### Impact Assessment
- **Severity**: HIGH - Systematic corruption of academic taxonomy
- **Scope**: 15-20% of extracted concepts have semantic validity issues
- **Propagation**: Errors cascade through all 4 hierarchy levels

### Technical Details
- **Optimization Settings**: light mode, openai/gpt-4o-mini task model
- **Training Examples**: 52 training, 23 validation
- **Metric**: Pure F1-score with exact string matching
- **Missing**: Semantic validation, linguistic checks, domain constraints

### Next Steps
- [ ] Verify current production prompts are not using GEPA-optimized versions
- [ ] Document specific failure patterns for future reference
- [ ] Consider alternative optimization approaches for semantic tasks
- [ ] Review other GEPA-optimized prompts for similar issues

### Notes
- Training data quality was not the issue - examples were well-designed
- GEPA learned patterns correctly but over-applied them mechanically
- Fundamental limitation of statistical optimization for semantic tasks

---
*Created: 2025-09-06*
*Status: Open*
*Priority: High*