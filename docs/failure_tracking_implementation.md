# Failure Tracking Implementation

## Overview
Implemented a simple, file-based failure tracking system with retry logic using exponential backoff. The solution focuses on the core requirements without over-engineering.

## Key Features

### 1. Simple Failure Tracking (`generate_glossary/utils/failure_tracker.py`)
- **File-Based Storage**: Saves failures to JSON files in `data/failures/` directory
- **Daily Log Files**: Creates one file per day (e.g., `failures_2025-09-09.json`)
- **Minimal Data Structure**: Records timestamp, module, function, error type, message, and context
- **Non-Blocking**: Failure tracking errors don't crash the application

### 2. Retry Logic with Exponential Backoff
- **Tenacity Integration**: Uses the existing `tenacity` library for robust retry logic
- **Exponential Backoff**: Wait times increase exponentially (2s, 4s, 8s, etc.)
- **Selective Retries**: Only retries on specific exceptions (RateLimitError, Timeout, ConnectionError)
- **Configurable Attempts**: Different retry counts for different operations

### 3. Continue-on-Failure Processing
- **Batch Processing**: Individual failures don't stop entire batches
- **Partial Results**: Returns successful results even when some items fail
- **Error Recording**: Failed items are logged but processing continues

## Changes Made

### 1. `generate_glossary/utils/failure_tracker.py` (NEW)
- `save_failure(module: str, function: str, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None, failure_dir: Optional[Path] = None)`: Appends failure records to daily JSON files. Parameters include the module name, function name, error type, error message, optional context dict, and optional custom failure directory.
- `load_failures(date: Optional[str] = None, failure_dir: Optional[Path] = None) -> List[Dict[str, Any]]`: Reads and returns parsed failure records. Takes optional date string (YYYY-MM-DD format, defaults to today) and optional failure directory, returns list of failure record dictionaries.
- Simple file I/O with error handling
- No complex patterns or in-memory tracking

### 2. `generate_glossary/utils/llm.py` (MODIFIED)
- Added `@retry` decorator to `completion()` with exponential backoff
- Added failure tracking to all exception handlers
- Modified `structured_completion_consensus()` to continue on individual failures
- Preserves all existing function signatures

### 3. `generate_glossary/mining/firecrawl.py` (MODIFIED)
- Added `@retry` decorators to key functions
- Added failure tracking to exception handlers
- Modified batch processing to continue on failures
- Records context for each failure

## Usage Examples

### Viewing Failures
```python
from generate_glossary.utils.failure_tracker import load_failures

# Load today's failures
failures = load_failures()

# Load failures from specific date
failures = load_failures(date="2025-09-08")
```

### Failure File Format
```json
[
  {
    "timestamp": "2025-09-09T01:57:32.645967",
    "module": "generate_glossary.utils.llm",
    "function": "completion",
    "error_type": "RateLimitError",
    "error_message": "Rate limit exceeded",
    "context": {
      "model": "openai/gpt-4",
      "messages_count": 5
    }
  }
]
```

## Benefits

1. **Simple Implementation**: No complex singleton patterns or advanced analytics
2. **File-Based Persistence**: Easy to analyze failures later with any tool
3. **Non-Intrusive**: Minimal changes to existing code
4. **Automatic Retries**: Reduces transient failures without manual intervention
5. **Continue Processing**: Maximizes successful results even with partial failures

## Testing

The implementation was tested with:
- Basic failure saving and loading
- Retry logic with exponential backoff
- Type checking with pyright
- Import verification

All tests passed successfully, confirming the system works as intended.