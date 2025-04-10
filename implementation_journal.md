# Implementation Journal: Handling Ambiguity in Academic Glossary

**Date:** 2024-08-01

**Goal:** Enhance the academic glossary generation pipeline to accurately capture and represent distinct senses of ambiguous terms based on their hierarchical context (College -> Department -> Research Area -> Venue Topic). This is critical for downstream applications like annotation, ensuring terms like "stress" are correctly identified based on their domain (e.g., Psychology vs. Materials Science).

## Current Plan (v2 - Adopted 2024-08-01)

**Approach:** Introduce a new, distinct module (`SenseDisambiguation`) dedicated to sense disambiguation and contextual refinement. This module runs *after* initial term generation/processing (including aggressive deduplication) and *before* final hierarchy building.

### Phase 1: Develop `SenseDisambiguation` Module

**Objective:** Create a new module that identifies and resolves ambiguities based on context and resources.

1. **Task 1.1: Define Input/Output**

   * **Status:** Defined (2024-08-01)
   * **Input:** Preliminary `hierarchy.json` (or aggregated `data/final/lv*` data) *after* existing Processing Phase steps (Web Mining, Validation, Aggressive Deduplication, Metadata Collection).
   * **Output:** A modified `hierarchy.json` structure where ambiguous nodes identified in the input are split into distinct sense nodes. Each sense node will have a unique ID, the original term string, clear parent linkage, a `disambiguator_tag` (e.g., "psychology"), and appropriately partitioned resources/metadata.
2. **Task 1.2: Implement Ambiguity Detection Logic**

   * **Status:** Completed (Candidate List Generated) (2024-08-01)
   * **Action:** Develop logic to identify potentially merged canonical nodes.
     * **PoC 1 (Parent Context Distance):** Implemented. Generated 1375 candidates. Analysis showed high sensitivity but low precision due to noise in input hierarchy parent links; results deemed unreliable for primary use.
     * **PoC 2 (Resource Topic/Cluster Analysis):** Implemented using DBSCAN (`eps=0.4`, `min_samples=2`). Generated 134 candidates (`resource_cluster_ambiguous_eps0.4.txt`). Analysis showed this list contains high-quality, plausible candidates for ambiguity based on content diversity.
   * **Decision:** The list of 134 terms from the Resource Cluster Detector will be used as the primary input candidate list for Task 1.3.
3. **Task 1.3: Implement Contextual Comparison & Splitting Logic**

   * **Status:** In Progress (2024-08-01)
   * **Action:** For the 134 ambiguous candidates identified in Task 1.2:
     * Analyze parent contexts and associated resource clusters (from DBSCAN results in Task 1.2).
     * Group original resources/metadata according to the identified clusters/contexts.
     * Implement logic to confirm if splitting is justified (based on cluster separation, context differences).
     * Implement enhanced sense tagging (Task 1.3b) using LLMs (primary) with fallbacks (TF-IDF, parent context) to generate meaningful `disambiguator_tag`s for each confirmed sense.
   * **Refinement Plan (Task 1.3b - Improving Sense Tagging):**
     1. **Approach Options:**
        * **Option A - Content-Based Tag Generation:** Extract domain keywords from the clustered resources themselves using:
          - TF-IDF to identify distinctive terms in each cluster
          - Topic modeling (LDA) to identify themes within clusters
          - Named entity recognition to extract domain-specific entities
        * **Option B - Embedding-Based Tag Generation:** Use nearest neighbors in embedding space to:
          - Compare cluster centroids to a known taxonomy of academic fields
          - Find closest academic domain terms for each cluster
        * **Option C - LLM-Based Tag Generation:** Leverage the SenseSplitter's planned LLM capability to:
          - Analyze resource snippets in each cluster
          - Generate appropriate domain label based on content analysis
          - Ensure consistency with academic terminology
     2. **Implementation Priority:** Option C (LLM-Based) appears most promising as it can:
        - Handle nuanced semantic differences between clusters
        - Generate human-readable, academically appropriate labels
        - Operate with or without parent context information
     3. **Technical Requirements:**
        * Implement resource content sampling for each cluster (limit to 3-5 resources per cluster)
        * Create prompt template for domain tag extraction
        * Incorporate fallback to Option A if LLM is unavailable
        * Add post-processing to standardize tags (lowercase, underscores, remove special characters)
     4. **Evaluation Criteria:**
        * Tags should be domain-specific academic terms (not generic descriptions)
        * Tags should clearly distinguish between senses
        * Tags should be consistent with academic terminology
        * No more than 10% of terms should have generic "sense_X" labels in final output
   * **Implementation Updates (2024-08-01):**
     * Implemented enhanced tag generation in `splitter.py` using a multi-layered approach:
       1. Primary: LLM-based tag generation that analyzes resource snippets and context
       2. Fallback: TF-IDF based domain extraction for when LLM is unavailable
       3. Last resort: Parent-context based tags when available
       4. Final fallback: Generic "sense_N" tags only when all else fails
     * Added a comprehensive prompt template for LLM tag generation
     * Added a domain dictionary for TF-IDF based tagging
     * Included resource content collection for each cluster
     * Code has been structured to seamlessly integrate with actual LLM API in production
     * Set both `use_llm_for_tags` and `improved_tagging` to `True` by default
   * **Next Steps:**
     * Run the updated SenseSplitter on the 133 terms from ResourceClusterDetector (eps=0.4)
     * Evaluate the quality of generated tags
     * Integrate with actual LLM API for production use (OpenAI, Anthropic, etc.)
   * **Evaluation Results & Enhancement Plan (2024-08-09):**
     * Successfully tested SenseSplitter with actual LLM integration
     * Generated 133 split proposals saved to JSON for analysis
     * **Identified Issues:**
       1. **Same-Tag Senses:** Some terms have multiple clusters tagged with the same domain (e.g., two "biology" senses)
       2. **Over-Splitting:** Some terms split into very closely related domains (e.g., "artificial_intelligence" and "computer_science")
       3. **Insufficient Context:** LLM tagging based only on resource content lacks hierarchical awareness
     * **Enhancement Strategy:**
       1. **Level-Specific Processing:** Implement hierarchy-level awareness (L0-L3) to adjust:
          - Clustering parameters (higher eps for broad domains, lower for specific topics)
          - Tag generation context (e.g., college vs. department vs. research area)
          - Split validation thresholds
       2. **Source Type Integration:** Each level has different sources (as per README.md):
          - L0: College/School names from faculty data
          - L1: Department names from web searches using L0 terms
          - L2: Research areas from web searches using L1 terms
          - L3: Conference/journal topics from web searches using L2 terms
       3. **Enhanced Validation:**
          - Create multi-signal validators that combine clustering, parent context, and semantic analysis
          - Implement level-specific thresholds to prevent over-splitting
          - Add same-tag detection to prevent redundant splits
       4. **Improved LLM Prompting:**
          - Include level context in prompts
          - Add parent context information
          - Request appropriate granularity based on term level
   * **Bug Fix & Improvement (2024-08-10):**
     * **Issue Identified:** Discovered that SenseSplitter was not respecting term hierarchy levels, causing terms like "law" (level 0) to appear in level 2 processing.
     * **Fix Implemented:** Added level filtering to the SenseSplitter class:
       1. Created a new method `_get_term_level()` to extract a term's level from hierarchy data
       2. Implemented `_filter_candidate_terms_by_level()` to filter candidate terms by their actual level
       3. Updated the `generate_split_proposals()` method to use level-filtered candidates
     * **Expected Impact:** Each level's output JSON will now only contain terms that actually belong to that hierarchy level, ensuring accurate level-specific processing and preventing inappropriate splits across hierarchy levels.
   * **Integration Improvements (2024-08-11):**
     * **Issue Identified:** The detector and splitter modules were not properly integrated, with redundant code in the splitter that simulated what the detector was already doing.
     * **Improvements Implemented:**
       1. Enhanced `ResourceClusterDetector` to:
          - Store detailed cluster information (labels, metrics, separation)
          - Save this data to a structured JSON file
          - Calculate and save clustering metrics like inter-cluster distances
       2. Updated `SenseSplitter` to:
          - Load pre-computed cluster results from the detector's JSON output
          - Remove redundant simulation code
          - Use the detector's metrics for improved validation
       3. Updated both modules' main execution sections for a smoother pipeline
     * **Expected Impact:** Proper separation of concerns where:
       - Detector: Identifies ambiguous terms and computes resource clusters
       - Splitter: Analyzes clusters to propose meaningful term splits
       - This improves efficiency, maintainability, and avoids redundant computation
   * **Validation Refinement (2024-08-12):**
     * **Issue Identified:** Some terms were being split even when the sense tags represented closely related academic fields (e.g., "mathematical optimization" vs. "numerical analysis").
     * **Improvement Implemented:** Enhanced split validation with advanced tag distinctness checking:
       1. Added a new `_check_tags_distinctness()` method that uses multiple strategies to identify related academic fields:
          - LLM-based determination of field distinctness
          - Detection of parent-child field relationships
          - Semantic similarity calculation using embeddings (as fallback)
       2. Updated `_validate_split()` to reject splits where tags aren't truly distinct academic domains
     * **Expected Impact:** Only terms with genuinely distinct academic field senses (like "linear_algebra" vs. "social_determinants" for "determinants") will be split, preventing unnecessary fragmentation of the taxonomy.
     * **Implementation Detail:**
       - The LLM-based approach is implemented in a new method `_check_field_distinctness_with_llm()`
       - This method uses a carefully crafted prompt instructing the LLM to consider:
         1. Whether fields represent the same domain with different terminology
         2. Whether fields are closely related subfields
         3. Whether fields have a parent-child relationship
         4. Whether fields are different expressions of the same concept
       - The LLM responds with a yes/no decision and reasoning, which is parsed and returned
       - This approach is more comprehensive and adaptable than hardcoded rules, leveraging the LLM's broad knowledge of academic fields
   * **Core Meaning Validation (2024-08-12):**
     * **Issue Identified:** Even with LLM-based field distinctness checking, some terms were still being split into senses that represent different aspects of the same core concept (e.g., "coastal_ecology" vs. "coastal_management").
     * **Improvement Implemented:** Refined the LLM prompt to focus specifically on differentiating between:
       1. TRUE AMBIGUITY: Where the core meaning of a term is fundamentally different between senses (e.g., "bank" as financial institution vs. river bank)
       2. DIFFERENT ASPECTS: Where a term refers to the same core concept but with different facets or applications (e.g., "coastal zones" as ecology vs. management)
     * **Concrete Examples Added to Prompt:**
       - Valid splits: "Indian law" (Native American law vs. India's legal system), "cell" (biology vs. prison), "determinants" (mathematics vs. social factors)
       - Invalid splits: "coastal zones" (ecology vs. management), "machine learning" (supervised vs. reinforcement), "legal theory" (positivism vs. natural law)
     * **Default Behavior Changed:** Modified the validation to default to "not distinct" in ambiguous cases, being conservative about splitting and thus preventing taxonomy fragmentation
     * **Expected Impact:** Significantly reduced over-splitting by focusing only on true semantic ambiguities where the core meaning differs, not just different perspectives on the same concept
   * **Conservative Threshold Adjustment (2024-08-13):**
     * **Issue Identified:** Despite improved LLM-based validation, additional quantitative controls were needed to prevent over-splitting.
     * **Improvement Implemented:** Adjusted thresholds in the cluster separation calculation and validation:
       1. Increased level-specific threshold values in `_validate_split` method:
          - Level 0 (colleges): 0.6 → 0.7 (requiring very strong evidence)
          - Level 1 (departments): 0.5 → 0.6 (requiring strong evidence)
          - Level 2 (research areas): 0.3 → 0.5 (requiring moderate evidence)
          - Level 3 (conference topics): 0.2 → 0.4 (requiring substantial evidence)
       2. Made the `_calculate_cluster_separation` method more conservative:
          - Increased level scaling values from [0.7, 0.6, 0.5, 0.4] to [0.75, 0.65, 0.6, 0.5]
          - Reduced score multiplier from 1.5 to 1.2
       3. Updated DBSCAN clustering to use eps=0.2 (tighter clusters) instead of eps=0.4
     * **Expected Impact:** Significantly higher bar for term splitting, ensuring only truly ambiguous terms with strong evidence are split, preventing taxonomy fragmentation
   * **Enhanced LLM Field Distinctness Prompt (2024-08-13):**
     * **Issue Identified:** LLM responses for field distinctness checks sometimes lacked consistency and clear decision boundaries.
     * **Improvement Implemented:** Completely redesigned the `_check_field_distinctness_with_llm` method:
       1. Created a more structured system prompt with a clear decision algorithm
       2. Required a rigid output format with explicit ANALYSIS and DECISION sections
       3. Set temperature to 0.1 (very low) to minimize response variation
       4. Added regex pattern matching for reliable extraction of decisions
       5. Improved examples showing clear distinctions between different meanings vs. different aspects
     * **Expected Impact:** More consistent and deterministic decisions about field distinctness, reducing false positives where different aspects of the same concept are incorrectly identified as distinct
   * **Placeholder Implementation (2024-08-14):**
     * **Issue Identified:** The fallback methods for TF-IDF tagging and parent context tagging were only defined as placeholders, reducing the system's effectiveness when LLM tagging fails.
     * **Improvement Implemented:** Fully implemented both placeholder methods:
       1. Added `_extract_domain_keywords_with_tfidf` method that:
          - Creates a corpus from all term resources
          - Applies TF-IDF vectorization to identify distinctive keywords
          - Filters for domain-specific academic terms
          - Returns the most relevant domain keyword as a tag
       2. Implemented `_extract_tag_from_parent_context` method that:
          - Tracks which resources in a cluster match which parent terms
          - Identifies the dominant parent context for a cluster
          - Uses appropriate tagging strategies based on parent level
          - Creates compound tags when needed for more specificity
       3. Updated the `_generate_sense_tags` method to properly integrate these approaches in the fallback chain
     * **Expected Impact:** Robust multi-layered tagging approach that maintains high-quality tags even when LLM is unavailable, reducing reliance on generic "sense_X" fallback tags
4. **Task 1.4: Implement Sense Splitting & Metadata Augmentation**

   * **Status:** Pending
   * **Action:** If splitting is confirmed:
     * Replace the original merged node with multiple new sense-specific nodes.
     * Assign new unique IDs.
     * Distribute original metadata (parents, sources, variations, resources) to the correct new nodes.
     * Add the `disambiguator_tag` metadata to each new node.
5. **Task 1.5: Integrate Module into Pipeline**

   * **Status:** Placement Defined (2024-08-01)
   * **Action:** Place this module as the *last step* of the Processing Phase, before `hierarchy_builder.py`. Modify runner scripts (`run_pipeline.py`, `run_interactive.py`).
6. **Task 1.6: Adapt `hierarchy_builder.py` (if needed)**

   * **Status:** Pending
   * **Action:** Ensure `hierarchy_builder.py` correctly consumes the output of the `SenseDisambiguation` module.

### Phase 2: Integration of Past Annotation Data (Future Consideration)

* **Objective:** Leverage real-world usage data (annotated papers) to further refine sense definitions and validate the glossary.
* **Status:** Blocked (Pending Phase 1)
* **Tasks:** Develop analysis framework; Design corpus-based refinement logic.

### Testing and Evaluation Strategy

* Select known ambiguous terms (e.g., "stress", "network", "model").
* Manually review their representation in the final `hierarchy.json` after the `SenseDisambiguation` step.
* Develop metrics to evaluate correct sense mapping from sample texts.

---

## Discussion Log & Key Decisions

*(Condensed log focusing on current plan)*

* **2024-08-01 (Decision):** Shifted from modifying existing pipeline components (v1) to creating a dedicated `SenseDisambiguation` module (v2) for better modularity.
* **2024-08-01 (Decision):** Adopted **Option B** for interaction with deduplication: The existing deduplicator runs aggressively first, and the new module identifies and *splits* incorrectly merged nodes based on context.
* **2024-08-01 (Decision):** Confirmed pipeline placement for the new module: Runs after all other Processing Phase steps (Web Mining, Validation, Deduplication, Metadata Collection) and before the final Hierarchy Construction.
* **2024-08-01 (Decision):** Defined basic Input/Output for the `SenseDisambiguation` module (using preliminary hierarchy/aggregated data as input, producing refined hierarchy with split senses as output).
* **2024-08-01 (AI Reflection):** The chosen approach (v2, Option B) concentrates complexity in the new module, specifically in accurately detecting merged nodes (Task 1.2) and partitioning metadata/resources during splits (Task 1.4). Defining the comparison logic and threshold (Task 1.3) is critical. Using preliminary `hierarchy.json` as input seems viable.
* **2024-08-01 (User Input):** Input format choice (preliminary `hierarchy.json` or aggregated files) flexible, ensure canonical forms vs variations are handled. Prefer PoCs for detection logic options. Annotated papers confirmed accessible for Phase 2.
* **2024-08-01 (Action):** Implemented and tested PoC detectors for Task 1.2 (Parent Context & Resource Clustering).
* **2024-08-01 (Finding & Decision):** Parent Context detector produced a large, noisy list (1375 terms). Resource Cluster detector (DBSCAN, eps=0.4) produced a higher-quality, manageable list (134 terms). Decided to use the Resource Cluster list as input for Task 1.3.
* **2024-08-09 (Decision):** After evaluating initial implementation results, decided to enhance SenseSplitter with level-specific processing (L0-L3) and additional context information. Will implement DBSCAN parameter adjustment based on level, enhanced LLM prompting with level context, and multi-signal validation to prevent over-splitting and redundant splits.
* **2024-08-10 (Bug Fix):** Fixed an issue where SenseSplitter was not filtering candidate terms by their actual hierarchy level, causing terms like "law" (level 0) to be processed at level 2. Implemented proper level filtering to ensure each level-specific output contains only appropriate terms.
* **2024-08-11 (Architecture Improvement):** Improved integration between `detector.py` and `splitter.py` by implementing proper separation of concerns. Enhanced the detector to store and output detailed cluster information in a structured JSON format, including metrics like inter-cluster distances. Updated the splitter to load and use these pre-computed results instead of redundantly simulating the clustering process. This eliminates code duplication, improves efficiency, and makes the pipeline more maintainable.
* **2024-08-12 (Validation Enhancement):** Implemented stricter validation of tag distinctness to ensure we only split terms when the sense labels represent truly distinct academic fields. Added comprehensive checks to prevent splitting when tags represent the same field (e.g., "law"/"legal_studies"), closely related fields (e.g., "optimization"/"numerical_analysis"), or parent-child fields (e.g., "computer_science"/"machine_learning"). These checks combine LLM evaluation, pattern detection, and semantic similarity using embeddings to provide a robust filtering mechanism.
* **2024-08-12 (Core Meaning Distinction):** Further refined the validation criteria to focus specifically on whether the term's core meaning is fundamentally different between senses, not just different aspects or applications of the same concept. Updated the LLM prompt with concrete examples distinguishing true ambiguity (like "Indian law" referring to Native American law vs. India's legal system) from mere facets of the same concept (like "coastal zones" ecology vs. management). Changed defaults to be conservative about splitting, preventing unnecessary taxonomy fragmentation.
* **2024-08-13 (Conservative Thresholds):** Adjusted numerical thresholds in cluster separation calculation and validation to be significantly more conservative, requiring stronger evidence before splitting terms. Changed DBSCAN parameters to use eps=0.2 for tighter clusters, making the system more resistant to over-splitting.
* **2024-08-13 (Deterministic LLM Prompts):** Redesigned the field distinctness LLM prompting to enforce a structured output format and more explicit decision algorithm, resulting in more consistent and reliable determinations about whether two academic fields represent distinct meanings or not.
* **2024-08-14 (Placeholder Implementation):** Fully implemented the previously placeholder methods for TF-IDF tagging and parent context tagging, creating a robust multi-layered approach to sense tagging that works effectively even when LLM is unavailable. This ensures high-quality tags with reduced reliance on generic fallbacks.

---

### Phase 3: Enhance Ambiguity Detection (`detector.py`) (Started 2024-08-15)

**Objective:** Improve the accuracy and robustness of the `ResourceClusterDetector` based on recent findings and splitter refinements.

**Planned Enhancements:**

1.  **Task 3.1: Upgrade Embedding Model**
    *   **Status:** In Progress
    *   **Action:** Replace the current default model (`all-MiniLM-L6-v2`) with a more powerful semantic model like `all-mpnet-base-v2` to better capture nuances in resource content. Evaluate impact on cluster quality and separation.
2.  **Task 3.2: Implement Adaptive Clustering Parameters**
    *   **Status:** In Progress
    *   **Action:** Modify `ResourceClusterDetector` to accept and utilize level-specific `dbscan_eps` and `dbscan_min_samples`, similar to the `SenseSplitter`. This allows tailoring cluster sensitivity to the expected granularity at each hierarchy level (e.g., tighter clusters for specific topics, looser for broad domains).
3.  **Task 3.3: Improve Content Handling**
    *   **Status:** Pending
    *   **Action:** Enhance the resource content extraction logic within the detector to be more robust against different data types (lists, JSON fragments) and potentially improve snippet truncation strategies.
4.  **Task 3.4: Explore Advanced Clustering (Future)**
    *   **Status:** Future Consideration
    *   **Action:** Investigate using HDBSCAN or hierarchical clustering algorithms, which might offer better performance on varying density clusters, potentially reducing reliance on fixed `eps` values. Requires assessing dependency additions.
5.  **Task 3.5: Hybrid Detection Strategy (Future)**
    *   **Status:** Future Consideration
    *   **Action:** Explore combining results from `ParentContextDetector` and `ResourceClusterDetector` (and potentially others) into a hybrid model with confidence scores.

---

## Discussion Log & Key Decisions

*(Condensed log focusing on current plan)*

* **2024-08-14 (Placeholder Implementation):** Fully implemented the previously placeholder methods for TF-IDF tagging and parent context tagging, creating a robust multi-layered approach to sense tagging that works effectively even when LLM is unavailable. This ensures high-quality tags with reduced reliance on generic fallbacks.

---
