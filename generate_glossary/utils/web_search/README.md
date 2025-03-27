# Web Search Utilities

This module provides a shared interface for web search, HTML fetching, list extraction, and filtering that can be used by both Level 1 (department extraction) and Level 2 (research area extraction) components of the generate_glossary pipeline.

## Components

### 1. Search (`search.py`)

Handles web search operations using external APIs:

- `WebSearchConfig`: Configuration for web search operations
- `web_search_bulk()`: Performs bulk web searches for multiple queries
- `save_term_search_results()`: Extracts term from query and saves raw search results

### 2. HTML Fetching (`html_fetch.py`)

Robust HTML fetching with error handling, caching, and encoding detection:

- `HTMLFetchConfig`: Configuration for HTML fetching
- `fetch_webpage()`: Main function to fetch webpage content with retry logic
- `check_cache()` / `save_to_cache()`: Cache management for web content
- Various utility functions for header generation, SSL handling, and encoding detection

### 3. List Extraction (`list_extractor.py`)

Extracts lists from HTML content with various heuristics:

- `ListExtractionConfig`: Configuration for list extraction parameters
- `extract_lists_from_html()`: Main function to extract lists from HTML content
- `score_list()`: Score a list based on multiple heuristics and metadata
- Various utility functions for HTML structure analysis and quality metrics

### 4. Filtering (`filtering.py`)

Filters extracted lists based on quality and optionally validates with LLM:

- `FilterConfig`: Configuration for filtering parameters
- `filter_lists()`: Main function to filter extracted lists
- `validate_lists_with_llm_binary()` / `validate_lists_with_llm()`: LLM validation functions
- `consolidate_lists()`: Consolidates items from multiple lists into a single list

## Usage

See `example.py` for a complete example of how to use these utilities for both Level 1 and Level 2 processing.

### Basic Usage

```python
import asyncio
from generate_glossary.utils.web_search.search import WebSearchConfig, web_search_bulk
from generate_glossary.utils.web_search.html_fetch import HTMLFetchConfig, fetch_webpage
from generate_glossary.utils.web_search.list_extractor import ListExtractionConfig, extract_lists_from_html
from generate_glossary.utils.web_search.filtering import FilterConfig, filter_lists, consolidate_lists

async def process_term(term):
    # 1. Configure components
    search_config = WebSearchConfig(...)
    html_config = HTMLFetchConfig(...)
    list_config = ListExtractionConfig(...)
    filter_config = FilterConfig(...)
    
    # 2. Perform web search
    query = f"site:.edu {term} list"
    search_results = web_search_bulk([query], search_config)
    
    # 3. Extract URLs from search results
    urls = [r.get("url") for r in search_results.get("data", [])[0].get("results", [])]
    
    # 4. Fetch and process HTML
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(5)
        fetch_tasks = [fetch_webpage(url, session, semaphore, html_config) for url in urls]
        html_contents = await asyncio.gather(*fetch_tasks)
        
        # 5. Extract lists from HTML
        all_extracted_lists = []
        for html_content in html_contents:
            if html_content:
                extracted_lists = extract_lists_from_html(html_content, list_config)
                all_extracted_lists.extend(extracted_lists)
        
        # 6. Filter and validate lists
        filtered_lists = await filter_lists(all_extracted_lists, term, filter_config)
        
        # 7. Consolidate to get final items
        final_items = consolidate_lists(filtered_lists, term)
        
        return final_items
```

## Customizing for Different Levels

The utilities can be customized for different levels by providing:

1. Level-specific keywords and patterns
2. Custom scoring functions
3. Custom item cleaning functions
4. Different LLM prompts for validation

See `example.py` for specific customizations for Level 1 (departments) and Level 2 (research areas). 