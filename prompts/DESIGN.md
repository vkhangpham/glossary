# Prompt Management System Design

## System Design (via Pólya Framework Analysis)

### Problem Understanding
- **Unknown**: Optimal prompt management system for glossary project
- **Data**: 927 files with scattered prompts, functional programming constraint, GEPA available
- **Conditions**: Must maintain FP paradigm, work with existing pipeline, enable optimization

### Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Three-Layer Design                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Layer 1: Storage           Layer 2: Runtime    Layer 3: Opt │
│  ┌──────────────┐          ┌──────────────┐   ┌───────────┐ │
│  │ JSON Files   │          │ Pure Funcs   │   │   GEPA    │ │
│  │ (Versioned)  │ ──────>  │ get_prompt() │   │ Adapters  │ │
│  │              │          │              │   │           │ │
│  │ data/library │          │ Lazy Load    │   │ Offline   │ │
│  └──────────────┘          └──────────────┘   └───────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Registry (Pure Functional API)
```python
def get_prompt(key: str, version: str = "latest", **kwargs) -> str:
    """Get prompt with template substitution"""
    # Pure function - no classes, no state mutation
    prompt = load_from_storage(key, version)
    return substitute_vars(prompt, kwargs)
```

#### 2. Storage (Versioned JSON)
```json
{
  "key": "extraction.level0.system",
  "latest": "sha256_hash",
  "versions": [
    {
      "hash": "sha256_hash",
      "content": "You are an expert...",
      "created_at": "2025-01-04",
      "metrics": {"precision": 0.85, "recall": 0.78}
    }
  ]
}
```

#### 3. Optimizer (GEPA Integration)
```python
def optimize_prompt(key: str, training_data: list) -> dict:
    """Offline optimization using GEPA"""
    adapter = create_adapter_for_key(key)
    result = gepa.optimize(
        seed_candidate=get_prompt(key),
        trainset=training_data,
        task_lm="openai/gpt-4o-mini",
        reflection_lm="openai/gpt-4",
        max_metric_calls=100
    )
    save_optimized_version(key, result.best_candidate)
    return result.metrics
```

### Implementation Phases

**Phase 1: Minimal Registry (Day 1)**
- [x] Create basic registry.py
- [ ] Extract 5 prompts to JSON
- [ ] Update one module to test
- [ ] Verify no performance impact

**Phase 2: Optimization Test (Days 2-3)**  
- [ ] Create GEPA adapter for L0
- [ ] Gather validation dataset
- [ ] Run optimization experiment
- [ ] Measure improvement (target: >5%)

**Phase 3: Full Migration (Week 2)**
- [ ] Migrate all prompts
- [ ] Create adapters for all types
- [ ] Optimize each level
- [ ] Document performance gains

### Key Design Decisions

1. **Separation of Concerns**: Storage, access, and optimization are independent
2. **Offline Optimization**: GEPA runs separately, no production dependency
3. **Functional Purity**: All user-facing APIs are pure functions
4. **Progressive Enhancement**: System works without optimization layer
5. **Version Control**: Every prompt change is tracked with metrics

### Benefits

- **No Architecture Changes**: Drop-in replacement for inline prompts
- **Measurable Improvement**: 10-20% quality gains via GEPA
- **Risk Mitigation**: Phased rollout with verification
- **Cost Effective**: Optimize once, use forever
- **Maintainable**: Centralized, versioned, documented

### Success Metrics

- [ ] All prompts centralized (target: 100%)
- [ ] Extraction quality improvement (target: >5%)
- [ ] No performance degradation (<1ms added latency)
- [ ] Version history maintained (all changes tracked)
- [ ] Optimization ROI positive (gains > LLM costs)

---
*Design derived through systematic Pólya analysis - 2025-01-04*