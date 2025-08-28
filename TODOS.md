# TODO List

Simple tracking for code improvements and technical debt.

## Critical Priority 🔴

### [Architecture] Path Handling Anti-Pattern
**Problem**: Manual `sys.path.insert()` in 20+ scripts breaks when run from different directories  
**Files**: `lv0_s0_get_college_names.py:12`, `lv0_s1_extract_concepts.py:19`, `lv0_s2_filter_by_institution_freq.py`, `lv0_s3_verify_single_token.py`, all lv1/lv2/lv3 generation scripts  
**Impact**: Pipeline fails in CI/CD, Docker containers, or different execution contexts  
**Fix**: Replace with setuptools entry points or PYTHONPATH management  
**Status**: **Fixed** ✓ (UV setup + pyproject.toml + console scripts)  

### [Architecture] Config Duplication  
**Problem**: File paths, thresholds, batch sizes hardcoded across 12+ scripts  
**Files**: `lv0_s1_extract_concepts.py:30-42`, similar patterns across all generation steps  
**Impact**: Changes require touching multiple files, high maintenance burden  
**Fix**: Extract to central config file or environment variables  
**Status**: **Fixed** ✓ (Centralized config.py with level-specific overrides)  

### [Reliability] No Rollback Strategy
**Problem**: Pipeline failures lose expensive LLM work (~$500+ per full run)  
**Files**: All generation steps, `lv0_s0_get_college_names.py:221`  
**Impact**: Must restart entire pipeline on any failure  
**Fix**: Implement granular checkpointing within steps, not just between steps  
**Status**: **Fixed** ✓ (Checkpoint system with batch-level recovery + CLI management tool)  

## High Priority 🟡

### [Security] API Key Exposure Risk
**Problem**: Keys loaded directly from .env without validation or masking in logs  
**Files**: `utils/llm.py`, `utils/web_miner_runner.py:78-86`, `utils/tavily_miner.py`  
**Impact**: Potential credential exposure in error logs or debug output  
**Fix**: Add key validation and consistent masking (first 4 + last 4 chars only)  
**Status**: **Fixed** ✓ (Secure config system + key validation + masking + CLI tools)  

### [Performance] Mixed Async/Sync Inefficiency
**Problem**: Creating new event loops in multiprocessing workers  
**Files**: `lv0_s1_extract_concepts.py:301-308`  
**Impact**: Degraded performance, resource contention at scale  
**Fix**: Pure async with proper event loop management or pure multiprocessing  
**Status**: Open  

### [Architecture] Monolithic File Complexity
**Problem**: Large files violate single responsibility principle  
**Files**: `graph_dedup.py` (2,598 lines), `splitter.py` (2,134 lines)  
**Impact**: Difficult to test, debug, and maintain  
**Fix**: Break into focused modules with clear responsibilities  
**Status**: Open  

## Medium Priority 🟢

### [Performance] Memory Management Missing
**Problem**: No memory management for large datasets  
**Files**: Web scraping and embedding generation modules  
**Impact**: OOM errors on large institution datasets  
**Fix**: Implement streaming/chunked processing  
**Status**: Open  

### [Reliability] Error Handling Inconsistency
**Problem**: Some failures silent, others crash entire pipeline  
**Files**: Various LLM integration points  
**Impact**: Difficult to debug failures in production  
**Fix**: Standardize error handling with proper logging levels  
**Status**: Open  

### [Performance] Duplicate Logger Setup
**Problem**: Logger setup code duplicated across multiple files  
**Files**: `utils/llm.py`, `utils/logger.py`  
**Fix**: Centralize to one location  
**Status**: Open  

## Low Priority

### [Documentation] Incorrect Directory Structure
**Problem**: README references non-existent pipeline scripts  
**Files**: `README.md:40`  
**Fix**: Update directory structure diagram  
**Status**: **Fixed** ✓  

---

## Implementation Priorities

**Immediate (Deploy Blockers)**:
1. ✓ Fix path management anti-pattern - breaks in production containers
2. ✓ Centralize configuration - reduces deployment risk

**Short-term (Operational)**:
3. ✓ Add API key security measures
4. ✓ Implement checkpoint/rollback strategy to protect LLM investments

**Medium-term (Scale)**:
5. Refactor monolithic files
6. Fix async/sync performance issues
7. Add memory management

## Quick Stats
- **Total**: 10 items
- **Open**: 5  
- **Fixed**: 5
- **Critical**: 3 (3 fixed)
- **High Priority**: 3 (1 fixed)
- **Medium Priority**: 3

*Last updated: 2025-08-28 (Code Review)*