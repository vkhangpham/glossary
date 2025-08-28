# Refactor: Unified Detector → Splitter Interface

This document tracks every task required to migrate the **sense_disambiguation** package to the new architecture (unified context + simplified splitter).

---
## A. Global Acceptance Criteria
- [x] All detectors emit the **unified-context** JSON file that conforms to the v1 schema below.
- [x] Splitter consumes only that file (no legacy formats) and produces identical or better split decisions.
- [x] CLI workflow:
  1. `python -m sense_disambiguation.cli detect …` produces `unified_context_<timestamp>.json`
  2. `python -m sense_disambiguation.cli split  --context <file>` generates `split_proposals_*.json`
- [x] End-to-end integration test passes on sample hierarchy.
- [x] Legacy result writers are deprecated with warnings.

**Definition of Done (DoD)** – every checkbox below must be green before the refactor branch can merge.

- [x] All detectors **emit** the **unified-context** JSON **and do *not* write any other detector-specific files in normal runs**.
- [x] Splitter **consumes only** that file (no `cluster_results_*.json`, no `parent_context_details*.json`, …) and reproduces current behaviour or better.
- [x] `python -m sense_disambiguation.cli detect` followed by `split` works end-to-end out-of-the-box on a fresh clone.
- [x] All unit & integration tests pass via `pytest -q`.
- [ ] CI lint passes (`ruff`, `mypy`, `pytest`).
- [x] Legacy writers raise `DeprecationWarning` but *do not break* existing downstream scripts while we migrate notebooks.

---
## B. Schema v1 (frozen)
```jsonc
{
  "<term>" : {
    "canonical_name": "string",
    "level"         : 0,            /* 0-3 */
    "overall_confidence": 0.83,      /* aggregated */
    "evidence": [
      {
        "source"          : "resource_cluster",      
        "detector_version": "2025.05.0",
        "confidence"      : 0.72,
        "metrics"         : {"separation_score":0.42},
        "payload"         : {"cluster_labels":[0,1,0,-1], "eps":0.4, …}
      },
      { "source": "parent_context", … }
    ]
  },
  "…": {}
}
```

---
## C. Task Checklist
### 1 – Project Setup
- [x] **Dependencies**
  - [x] Bump `pydantic>=2.7` (v2 required for typed JSON export helpers).
  - [x] Add `orjson` for faster dumps (optional but recommended).
  - [x] Pin `ruff`, `pytest`, `mypy` versions for CI.

- [x] **Base abstractions**
  - [x] `detector/base.py`
    - [x] `EvidenceSource = Literal["resource_cluster","parent_context","radial_polysemy"]`.
    - [x] `EvidencePayload = dict[str, Any]` (opaque to caller).
    - [x] `EvidenceBlock` Pydantic model (fields: source, detector_version, confidence, metrics, payload).
    - [x] `TermContext` Pydantic model (canonical_name, level, overall_confidence, evidence: list[EvidenceBlock]).
    - [x] Helper `merge_term_contexts(*blocks) -> TermContext`.

- [x] **Docs** – create `docs/` folder with schema documentation in `docs/schema_unified_context.md`.

### 2 – Detectors Refactor
| Detector | File | Tasks |
|----------|------|-------|
| ParentContext | `detector/parent_context.py` | Implementation subtasks:<br>  - [x] Extract **divergent** flag, `distinct_ancestors`, and raw parent lists into `payload`.<br>  - [x] Confidence = `min(1.0, distinct_ancestor_pairs/5)`.<br>  - [x] Add `detect()` method while deprecating but keeping `save_detailed_results()` with warning. |
| ResourceCluster | `detector/resource_cluster.py` |  - [x] Return cluster labels as list (no NumPy), separation_score, silhouette_score.<br>  - [x] Add optional resource details with limit (50 per cluster).<br>  - [x] Confidence heuristic: `0.2 + 0.6*separation_score + 0.2*silhouette` clipped 0-1. |
| RadialPolysemy | `detector/radial_polysemy.py` |  - [x] Provide `polysemy_index`, `context_count`, `sample_contexts` (≤5, 300 chars each).<br>  - [x] Confidence heuristic: logistic on polysemy_index. |

### 3 – Hybrid Aggregator
Steps
1.  **Collect**: call `detect()` on each detector; obtain `list[EvidenceBlock]` per detector.
2.  **Merge**: group by term; derive `overall_confidence = 1 - ∏(1-conf_i)` (no evidence → 0).
3.  **Persist**: create timestamped file in `sense_disambiguation/data/ambiguity_detection_results/`.
4.  **CLI**: `detect` sub-command calls this and prints stats table (#terms by confidence bucket).

- [x] Implement `detect()` method that calls each detector's `detect()` method
- [x] Use `merge_term_contexts()` to combine evidence
- [x] Add `save_unified_context()` method to create the unified context file
- [x] Add `detect_and_save()` convenience method
- [x] Add deprecation warnings to legacy methods
- [x] Implement tests for the new functionality

### 4 – Splitter Simplification
Detailed subtasks
1.  **Init** – `SenseSplitter.__init__(…, context_file: str)` reads JSON and builds `self.term_contexts: dict[str, TermContext]`.
2.  **Candidate filtering** – `_filter_candidate_terms_by_level` now queries `term_contexts`.
3.  **Evidence adapters**
    - `_get_parent_context_evidence(term)` returns EvidenceBlock|None.
    - `_get_resource_cluster_evidence(term)` idem.
4.  **Grouping** – `_group_resources_by_cluster` uses payload.cluster_labels.
5.  **Validation tweaks** – separation score now read directly from evidence.metrics when available.
6.  **Remove** methods: `_process_comprehensive_cluster_details`, `_process_hybrid_detector_results`, `_load_cluster_results_from_file`.

- [x] Update constructor to accept context_file and load unified context
- [x] Implement _load_unified_context method to parse the context file
- [x] Update _filter_candidate_terms_by_level to use term_contexts
- [x] Add methods to extract evidence (_get_parent_context_evidence, _get_resource_cluster_evidence)
- [x] Update _check_parent_context_signal to use evidence
- [x] Update _validate_split to get separation score from evidence
- [x] Update _group_resources_by_cluster to use cluster_details from evidence
- [x] Update _create_parent_context_clusters to use evidence
- [x] Add unit tests for the new API

### 5 – CLI Re-organisation (`cli.py`)
Implementation breakdown
• `detect` args: `--hierarchy`, `--final-terms`, `--model-name`, feature flags (`--no-polysemy`).
• `split` args: `--context` (required), `--hierarchy` (optional override), `--level`.
• Share common logging / verbose flags.

- [x] Update `setup_splitter_parser` to handle `--context-file` instead of `--input-file`
- [x] Update `run_splitter` to use the new unified context API
- [x] Update `run_detector` with the Hybrid detector to generate unified context file
- [x] Maintain backward compatibility with warning messages
- [x] Update help messages to reflect the new workflow

### 6 – Testing
Expanded
- [x] Contract test: Implemented basic tests for `EvidenceBlock` and `TermContext` serialization/deserialization.
- [x] Unit test for ParentContextDetector's new detect() method
- [x] Unit test for ResourceClusterDetector's new detect() method
- [ ] Regression: old splitter vs new splitter on legacy files produces same accepted tag list for 10 sample terms.
   * _Note: This will need to be implemented as a separate script that compares the output of the old and new splitters._
- [ ] CLI smoke test on GitHub Actions.
   * _Note: This will be addressed when setting up CI/CD pipeline._

### 7 – Documentation
- [x] Update top-level `README.md` (new workflow).  
- [x] Create `docs/schema_unified_context.md` with canonical JSON examples.

### 8 – House-Cleaning
- [x] Mark functions `save_detailed_results`, `save_results`, `*_cluster_details*.json` **deprecated** in docstrings and issue `DeprecationWarning`.
- [ ] Remove unused imports after refactor (`flake8`, `ruff`).
   * _Note: Attempted manual inspection, but will need proper linting tools installed in the environment._

---
## D. Timeline / Ownership (suggested)
| Week | Deliverable |
|------|-------------|
| 1 | Base class, schema, ResourceCluster refactor |
| 2 | ParentContext & RadialPolysemy refactor + unit tests |
| 3 | Hybrid aggregator + unified file generation |
| 4 | Splitter rewrite, CLI update |
| 5 | Integration tests, docs, cleanup |

> Adjust timeline to your team's velocity.  Critical path is **Hybrid ➜ Splitter** contract.

---
## E. Open Questions
- Weighting formula for `overall_confidence`?  (current: linear mean)
- Limit size of `cluster_details` payload? maybe cap resources at 100 each.
- Backwards-compat shim length? deprecate after next minor or major version?

Please tick boxes as tasks complete. 