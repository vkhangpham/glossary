# Major Refactoring Plan - Glossary Analysis System

## Overview
This document tracks the comprehensive refactoring effort to streamline and improve the entire glossary analysis codebase. The refactoring focuses on 8 major modules with emphasis on code standardization, performance optimization, and maintainability.

## Status Summary
- **Completed**: 3 modules ✅
- **In Progress**: 1 module 🔄
- **Pending**: 4 modules ⏳
- **Estimated Completion**: TBD

---

## Module Breakdown

### 1. LLM: Streamlined Solution ✅
**Status**: COMPLETED
**Objective**: Create unified LLM interface with consistent error handling and cost optimization

**Completed Items**:
- [ ] Details to be documented

**Impact**: Reduced LLM costs by ~60%, unified provider interface

---

### 2. Web Mining: Speed & Cost Improvements ✅
**Status**: COMPLETED  
**Objective**: Migrate to Firecrawl SDK for 4x performance improvement

**Completed Items**:
- [x] Integrated Firecrawl SDK
- [x] Removed redundant web scraping code
- [x] 4x speed improvement achieved
- [x] 80% cost reduction vs separate search + LLM calls

**Impact**: $83/month for 10,000 concepts vs previous ~$400

---

### 3. Generation Scripts: Standardization (lv1-2-3) ✅
**Status**: COMPLETED
**Objective**: Standardize generation pipeline across all levels

**Completed Items**:
- [x] UV setup with pyproject.toml
- [x] Console scripts for all generation steps
- [x] Centralized configuration system
- [x] Checkpoint/recovery system with batch-level granularity

**Impact**: Consistent 4-step pipeline across all levels, protected against failures

---

### 4. Utils: Cleanup 🔄
**Status**: IN PROGRESS
**Objective**: Remove redundancy, consolidate utilities, improve organization

**Current State Analysis**:
- [ ] Identify all utility functions and their usage
- [ ] Map duplicate functionality
- [ ] Plan consolidation strategy

**Planned Changes**:
- [ ] Remove duplicate logger setup code
- [ ] Consolidate web search utilities
- [ ] Extract shared constants to config
- [ ] Remove unused utility functions
- [ ] Standardize error handling utilities
- [ ] Create clear module boundaries

**Target Structure**:
```
utils/
├── llm.py          # LLM interface only
├── config.py       # All configuration
├── logger.py       # Single logger setup
├── file_io.py      # File operations
└── validators.py   # Input validation
```

---

### 5. Post-Generation Processing: Cleanup & Improve ⏳
**Status**: PENDING
**Objective**: Refactor deduplicator, validator, and sense disambiguation

**Sub-modules**:

#### 5.1 Deduplicator
**Current Issues**:
- graph_dedup.py is 2,598 lines (monolithic)
- Mixed responsibilities
- Poor testability

**Planned Changes**:
- [ ] Break into focused modules
- [ ] Separate graph building from deduplication logic
- [ ] Extract similarity calculations
- [ ] Improve memory efficiency
- [ ] Add streaming support for large datasets

#### 5.2 Validator
**Current Issues**:
- Multiple validation modes with inconsistent interfaces
- Hardcoded thresholds

**Planned Changes**:
- [ ] Unified validation interface
- [ ] Configurable validation rules
- [ ] Batch validation support
- [ ] Better error reporting

#### 5.3 Sense Disambiguation
**Current Issues**:
- Complex logic in splitter.py (2,134 lines)
- Unclear separation of concerns

**Planned Changes**:
- [ ] Modularize disambiguation logic
- [ ] Separate detection from splitting
- [ ] Improve disambiguation rules
- [ ] Add confidence scoring

---

### 6. Hierarchy Scripts: Cleanup & Improve ⏳
**Status**: PENDING
**Objective**: Streamline hierarchy building and evaluation

**Current Issues**:
- Scattered hierarchy logic
- Inconsistent data structures
- Complex visualization setup

**Planned Changes**:
- [ ] Consolidate hierarchy building logic
- [ ] Standardize parent-child relationship handling
- [ ] Improve variation consolidation
- [ ] Optimize resource transfer
- [ ] Simplify evaluation metrics
- [ ] Modernize visualization interface

---

### 7. Definition & Embeddings: Cleanup & Improve ⏳
**Status**: PENDING
**Objective**: Optimize definition generation and embedding management

**Current Issues**:
- Large embedding files in Git LFS
- Inconsistent definition formats
- Memory issues with large embeddings

**Planned Changes**:
- [ ] Implement streaming for embedding generation
- [ ] Standardize definition format
- [ ] Add caching layer for embeddings
- [ ] Optimize vector storage with FAISS
- [ ] Improve definition quality with better prompts
- [ ] Add definition validation

---

### 8. Standardize All Scripts ⏳
**Status**: PENDING
**Objective**: Create consistent interfaces and patterns across all scripts

**Standardization Goals**:
- [ ] Consistent CLI argument parsing
- [ ] Unified logging format
- [ ] Standard error codes
- [ ] Common input/output formats
- [ ] Shared configuration loading
- [ ] Consistent progress reporting
- [ ] Standard documentation format

**Target Patterns**:
```python
# Standard CLI structure
def main():
    args = parse_args()
    config = load_config(args.level)
    setup_logging(args.verbose)
    
    try:
        result = process(args, config)
        save_output(result, args.output)
    except Exception as e:
        log_error(e)
        sys.exit(ERROR_CODES[type(e)])
```

---

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-2)
1. Complete Utils cleanup (Module 4)
2. Establish standard patterns (Module 8 foundation)
3. Create shared testing framework

### Phase 2: Core Refactoring (Weeks 3-5)
1. Refactor post-generation processing (Module 5)
2. Apply standard patterns to refactored code
3. Add comprehensive tests

### Phase 3: Advanced Features (Weeks 6-7)
1. Improve hierarchy scripts (Module 6)
2. Optimize definitions & embeddings (Module 7)
3. Performance testing and optimization

### Phase 4: Polish & Documentation (Week 8)
1. Complete standardization (Module 8)
2. Update all documentation
3. Create migration guide
4. Final testing and validation

---

## Success Metrics

1. **Code Quality**
   - Reduce average file size to <500 lines
   - Achieve 80% test coverage
   - Zero duplicate code blocks

2. **Performance**
   - 50% reduction in memory usage
   - 30% faster pipeline execution
   - Support for 10x larger datasets

3. **Maintainability**
   - Single source of truth for configs
   - Consistent error handling
   - Clear module boundaries

4. **Reliability**
   - Full checkpoint/recovery coverage
   - Graceful failure handling
   - Comprehensive logging

---

## Risk Mitigation

1. **Backward Compatibility**
   - Maintain old interfaces during transition
   - Provide migration scripts
   - Document breaking changes

2. **Data Integrity**
   - Validate output consistency
   - Keep backup of original code
   - Test with production data

3. **Performance Regression**
   - Benchmark before/after each change
   - Profile memory usage
   - Monitor LLM costs

---

## Notes & Decisions

### Decisions Made:
- Firecrawl SDK chosen for web mining (4x speed improvement)
- UV package manager for dependency management
- Functional programming style (no OOP)
- Console scripts via pyproject.toml

### Open Questions:
- [ ] Should we migrate to async for all I/O operations?
- [ ] Consider using Pydantic for all data models?
- [ ] Should we add a database layer for metadata?
- [ ] Consider containerization for deployment?

---

## Session Progress Tracking

### Session 1 (2025-08-29)
- Created refactoring plan
- Identified 8 major modules
- Established implementation phases
- Set success metrics
- **Completed comprehensive architecture analysis** (ARCHITECTURE_ANALYSIS.md)
- **Identified key anti-patterns**:
  - Monolithic files (graph_dedup.py: 2,574 lines)
  - Module boundary violations (metadata_collector.py doing 5+ concerns)
  - Missing abstraction layers
  - Data flow coupling with hardcoded paths
- **Proposed domain-driven architecture** with clear separation:
  - Core (domain models & business rules)
  - Application (use cases & workflows)
  - Infrastructure (technical implementations)
  - Interfaces (CLI, web, API entry points)

### Architecture Insights Applied to Modules:

**Module 4 (Utils)** - Now has clear target structure:
- Extract to infrastructure layer (persistence, integrations, config)
- Move business logic to core/services
- Keep only cross-cutting concerns in shared/

**Module 5 (Post-Generation)** - Major refactoring needed:
- graph_dedup.py (2,574 lines) → Split into 5+ focused modules
- splitter.py (2,105 lines) → Separate detection from splitting logic
- Apply repository pattern for data access

**Module 6 (Hierarchy)** - Needs domain modeling:
- Create Hierarchy aggregate root
- Define Relationship entities
- Separate building logic from persistence

### Next Steps:
1. Begin Module 4 implementation with new architecture patterns
2. Create core domain models as foundation
3. Extract infrastructure layer from utils
4. Set up proper test framework

---

*Last Updated: 2025-08-29*
*Session Owner: Kyle*