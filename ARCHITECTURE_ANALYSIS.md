# Architecture Analysis - Glossary Analysis System

## Executive Summary

**Current State**: 94 Python files, ~31K lines of code, organically grown pipeline with 4 hierarchy levels  
**Main Issues**: Module boundary violations, monolithic files (2.5K+ lines), inconsistent patterns  
**Recommendation**: Domain-driven restructuring with clear separation of concerns

---

## Current Architecture Overview

### The Pipeline Flow
```
Level 0 (Colleges) → Level 1 (Depts) → Level 2 (Research) → Level 3 (Topics)
     ↓                    ↓                  ↓                   ↓
 4-Step Gen          4-Step Gen         4-Step Gen         4-Step Gen
     ↓                    ↓                  ↓                   ↓
 Web Mining          Web Mining         Web Mining         Web Mining
     ↓                    ↓                  ↓                   ↓
 Validation          Validation         Validation         Validation
     ↓                    ↓                  ↓                   ↓
 Deduplication       Deduplication      Deduplication      Deduplication
     ↓                    ↓                  ↓                   ↓
                    Hierarchy Building
                            ↓
                     Sense Disambiguation
                            ↓
                    Definition Generation
                            ↓
                    Embedding Generation
```

### Current Directory Structure Analysis

```
glossary/
├── generate_glossary/       # MAIN MODULE (17K lines)
│   ├── generation/          # Pipeline scripts
│   │   ├── lv0/            # 4 scripts per level
│   │   ├── runners/        # Level orchestrators
│   │   └── shared/         # Shared extraction logic
│   ├── deduplicator/       # 3.7K lines total
│   ├── validator/          # Validation logic
│   └── utils/              # 3.2K lines - THE DUMPING GROUND
├── hierarchy/              # 3.1K lines - Hierarchy building
├── sense_disambiguation/   # 6.5K lines - Ambiguity detection
├── generate_definition/    # 1.1K lines - Definition generation
├── generate_embeddings/    # 2K lines - Vector embeddings
└── tests/                  # Minimal coverage
```

---

## Architectural Problems

### 1. **Module Boundary Violations**
```python
# Example: metadata_collector.py (1,256 lines) doing:
- File discovery (I/O concern)
- Data extraction (Business logic)
- Parent relationship inference (Domain logic)
- CSV/JSON operations (Serialization)
- Metadata consolidation (Aggregation)
```
**Impact**: Single change affects multiple concerns, high regression risk

### 2. **Monolithic Files (The "God Object" Anti-Pattern)**
- `graph_dedup.py`: 2,574 lines
- `splitter.py`: 2,105 lines
- `resource_cluster.py`: 1,377 lines

**Why this happened**: Feature accretion without refactoring cycles  
**Impact**: Impossible to test in isolation, high cognitive load

### 3. **Inconsistent Processing Patterns**
```python
# Pattern 1: Direct processing
def process_level_0():
    data = load()
    result = transform(data)
    save(result)

# Pattern 2: Class-based processing
class ConceptExtractionProcessor(ResilientProcessor):
    def process_sources(self, sources):
        # Different approach

# Pattern 3: Runner pattern
class LevelRunner:
    def run_pipeline(self):
        # Yet another approach
```
**Impact**: New developers can't predict code organization

### 4. **Data Flow Coupling**
Each level depends on exact file names from previous level:
```python
# lv1_s0 expects:
input_file = "data/lv0/lv0_final.txt"  # Hardcoded!
```
**Impact**: Can't run levels independently, can't parallelize

### 5. **Missing Abstraction Layers**
No clear separation between:
- Infrastructure (file I/O, logging)
- Application (orchestration, workflow)
- Domain (glossary concepts, hierarchy rules)
- Integration (LLM, web services)

---

## Proposed Architecture

### Domain-Driven Design Structure

```
glossary/
├── core/                    # Domain model & business rules
│   ├── models/
│   │   ├── concept.py      # Concept entity
│   │   ├── hierarchy.py    # Hierarchy aggregate
│   │   ├── level.py        # Level value object
│   │   └── relationship.py # Relationship entity
│   ├── services/
│   │   ├── extraction.py   # Concept extraction logic
│   │   ├── validation.py   # Validation rules
│   │   ├── deduplication.py # Dedup algorithms
│   │   └── disambiguation.py # Sense splitting
│   └── repositories/
│       ├── concept_repo.py # Concept persistence interface
│       └── hierarchy_repo.py # Hierarchy persistence
│
├── application/            # Use cases & workflows
│   ├── pipelines/
│   │   ├── generation.py   # Generation pipeline
│   │   ├── processing.py   # Post-processing pipeline
│   │   └── enrichment.py   # Definition/embedding pipeline
│   ├── commands/          # Command handlers
│   └── queries/           # Query handlers
│
├── infrastructure/         # Technical implementations
│   ├── persistence/
│   │   ├── file_storage.py # File-based storage
│   │   └── checkpoints.py  # Checkpoint system
│   ├── integrations/
│   │   ├── llm/
│   │   │   ├── client.py   # LLM client abstraction
│   │   │   └── providers.py # OpenAI, Gemini, etc.
│   │   └── web/
│   │       ├── firecrawl.py # Firecrawl integration
│   │       └── tavily.py    # Tavily integration
│   └── config/
│       ├── settings.py     # Configuration management
│       └── security.py     # API key management
│
├── interfaces/            # Entry points
│   ├── cli/              # Command-line interfaces
│   │   ├── generate.py   # Generation commands
│   │   ├── process.py    # Processing commands
│   │   └── analyze.py    # Analysis commands
│   ├── web/              # Web interfaces
│   │   └── visualizer/   # Hierarchy visualization
│   └── api/              # Future API endpoints
│
└── shared/               # Cross-cutting concerns
    ├── logging.py        # Logging setup
    ├── metrics.py        # Performance metrics
    └── exceptions.py     # Custom exceptions
```

### Key Architectural Principles

#### 1. **Dependency Inversion**
```python
# Core doesn't depend on infrastructure
class ConceptRepository(Protocol):
    def save(self, concept: Concept) -> None: ...
    def find_by_id(self, id: str) -> Optional[Concept]: ...

# Infrastructure implements the interface
class FileConceptRepository(ConceptRepository):
    def save(self, concept: Concept) -> None:
        # File-specific implementation
```

#### 2. **Single Responsibility**
```python
# Before: One class doing everything
class MetadataCollector:
    def find_files(self): ...
    def extract_data(self): ...
    def save_metadata(self): ...

# After: Focused classes
class FileDiscovery:
    def find_level_files(self, level: int): ...

class MetadataExtractor:
    def extract(self, file_path: Path): ...

class MetadataPersistence:
    def save(self, metadata: Metadata): ...
```

#### 3. **Functional Core, Imperative Shell**
```python
# Pure functional core
def deduplicate_concepts(
    concepts: List[Concept], 
    similarity_threshold: float
) -> List[ConceptGroup]:
    # Pure function, no side effects
    return grouped_concepts

# Imperative shell handles I/O
def process_deduplication(input_file: Path, output_file: Path):
    concepts = load_concepts(input_file)  # I/O
    groups = deduplicate_concepts(concepts, 0.8)  # Pure
    save_groups(groups, output_file)  # I/O
```

#### 4. **Event-Driven Pipeline**
```python
@dataclass
class PipelineEvent:
    level: int
    step: str
    status: str
    data: Optional[Dict]

class Pipeline:
    def __init__(self):
        self.events = []
        
    def emit(self, event: PipelineEvent):
        # Checkpoint automatically
        # Log progress
        # Update metrics
        self.events.append(event)
```

---

## Migration Strategy

### Phase 1: Foundation (Week 1)
1. **Create core domain models**
   - Define Concept, Hierarchy, Level entities
   - No behavior yet, just data structures

2. **Extract infrastructure layer**
   - Move all file I/O to persistence module
   - Centralize LLM calls to integration layer

3. **Setup new test framework**
   - Unit tests for domain models
   - Integration tests for infrastructure

### Phase 2: Vertical Slice (Week 2)
1. **Implement one complete flow in new architecture**
   - Choose Level 0 generation as pilot
   - Full implementation: domain → application → infrastructure

2. **Parallel operation**
   - New architecture runs alongside old
   - Compare outputs for validation

### Phase 3: Progressive Migration (Weeks 3-4)
1. **Migrate level by level**
   - Level 1, then 2, then 3
   - Each level is independent

2. **Refactor monolithic files**
   - Break down graph_dedup.py
   - Split splitter.py into focused modules

### Phase 4: Cleanup (Week 5)
1. **Remove old code**
   - Delete migrated modules
   - Update documentation

2. **Performance optimization**
   - Profile new architecture
   - Optimize bottlenecks

---

## Benefits of New Architecture

### 1. **Testability**
- Pure functions are trivial to test
- Mocked repositories for integration tests
- 80%+ code coverage achievable

### 2. **Maintainability**
- Change in LLM provider affects only integration layer
- New validation rule doesn't touch file I/O
- Clear location for any new feature

### 3. **Scalability**
- Parallel processing per level
- Switch to database without changing domain
- Add caching layer transparently

### 4. **Developer Experience**
- Predictable code organization
- Clear dependency flow
- Reduced cognitive load

---

## Anti-Patterns to Avoid

### 1. **The "Utils" Trap**
```python
# BAD: Dumping ground
utils/
├── everything.py
├── helpers.py
└── misc.py

# GOOD: Specific purposes
infrastructure/
├── persistence/
├── integrations/
└── config/
```

### 2. **Leaky Abstractions**
```python
# BAD: Domain knows about files
class Concept:
    def save_to_file(self, path: str):
        # Domain shouldn't know about files!

# GOOD: Domain stays pure
class Concept:
    # Just domain logic
    
class ConceptRepository:
    def save(self, concept: Concept):
        # Infrastructure handles persistence
```

### 3. **Circular Dependencies**
```python
# BAD: Circular imports
# deduplicator.py
from validator import validate

# validator.py  
from deduplicator import deduplicate

# GOOD: Dependency injection
class Deduplicator:
    def __init__(self, validator: Validator):
        self.validator = validator
```

---

## Immediate Actions

### Quick Wins (Do Today)
1. **Extract file paths to config**
   ```python
   # config.py
   PATHS = {
       'lv0_final': 'data/lv0/lv0_final.txt',
       'lv1_input': 'data/lv1/raw/lv1_s0_input.txt'
   }
   ```

2. **Create error exception hierarchy**
   ```python
   class GlossaryError(Exception): pass
   class ValidationError(GlossaryError): pass
   class DeduplicationError(GlossaryError): pass
   ```

3. **Standardize logging**
   ```python
   # One logger factory
   def get_logger(name: str) -> Logger:
       return setup_logger(name)
   ```

### Critical Refactors (This Week)
1. Break down `metadata_collector.py` (1,256 lines)
2. Split `graph_dedup.py` (2,574 lines)
3. Extract business logic from utils

### Strategic Changes (This Month)
1. Implement repository pattern for data access
2. Create domain model layer
3. Build comprehensive test suite

---

## Success Metrics

### Code Quality
- **File size**: No file > 500 lines
- **Function size**: No function > 50 lines
- **Cyclomatic complexity**: < 10 per function
- **Test coverage**: > 80%

### Architecture Health
- **Coupling**: Low (dependency injection)
- **Cohesion**: High (single responsibility)
- **Dependencies**: Unidirectional (no cycles)
- **Abstraction**: Appropriate (no leaky abstractions)

### Performance
- **Memory**: < 2GB for full pipeline
- **Speed**: < 30 min for complete run
- **Parallelization**: All levels can run in parallel
- **Recovery**: < 1 min from checkpoint

---

## Conclusion

The current architecture shows signs of organic growth without deliberate design. The proposed domain-driven architecture will:

1. **Reduce complexity** through separation of concerns
2. **Improve testability** with pure functional core
3. **Enable scalability** through proper abstractions
4. **Enhance maintainability** with clear boundaries

The migration can be done incrementally without disrupting ongoing work, with immediate benefits visible after Phase 1.

---

*"Architecture is about the important stuff. Whatever that is."* - Ralph Johnson

The important stuff here is:
- **Protecting expensive LLM operations** (checkpoint system ✓)
- **Managing complex hierarchies** (domain model needed)
- **Scaling to more concepts** (better abstractions needed)
- **Enabling team development** (clear boundaries needed)