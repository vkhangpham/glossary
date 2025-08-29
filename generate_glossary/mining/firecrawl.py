"""
Firecrawl SDK-based web mining module for academic glossary extraction.
Replaces the complex HTML parsing pipeline with Firecrawl's AI-powered extraction.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
import time

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Firecrawl client
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not FIRECRAWL_API_KEY:
    logger.warning("FIRECRAWL_API_KEY not found in environment. Set it to use Firecrawl.")
    
# Constants
MAX_URLS_PER_CONCEPT = 3
MAX_CONCURRENT_OPERATIONS = 5
BATCH_SIZE = 25

class ConceptDefinition(BaseModel):
    """Schema for extracted academic concept definitions."""
    concept: str = Field(description="The academic term or concept")
    definition: str = Field(description="Clear, comprehensive definition")
    context: str = Field(description="Academic field or domain")
    key_points: List[str] = Field(default=[], description="Key characteristics")
    related_concepts: List[str] = Field(default=[], description="Related terms")
    source_quality: str = Field(default="general", description="Quality: authoritative/reliable/general")

class WebResource(BaseModel):
    """Schema for web resources containing definitions."""
    url: str
    title: str = ""
    definitions: List[ConceptDefinition] = []
    domain: str = ""
    
    def __init__(self, **data):
        super().__init__(**data)
        from urllib.parse import urlparse
        self.domain = urlparse(self.url).netloc

def initialize_firecrawl() -> Optional[FirecrawlApp]:
    """Initialize Firecrawl client with API key."""
    if not FIRECRAWL_API_KEY:
        logger.error("Firecrawl API key not configured")
        return None
    
    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        logger.info("Firecrawl client initialized successfully")
        return app
    except Exception as e:
        logger.error(f"Failed to initialize Firecrawl: {e}")
        return None

def search_concept_firecrawl(app: FirecrawlApp, concept: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for a concept using Firecrawl's search endpoint.
    
    Args:
        app: Firecrawl client instance
        concept: The concept to search for
        limit: Maximum number of results
        
    Returns:
        List of search results
    """
    try:
        # Build academic-focused query
        query = f'"{concept}" definition explanation academic OR wikipedia OR edu OR arxiv'
        
        logger.info(f"Searching for: {concept}")
        
        # Use Firecrawl search
        results = app.search(
            query=query,
            limit=limit
        )
        
        # Extract the results
        if isinstance(results, dict) and 'data' in results:
            return results['data']
        elif isinstance(results, list):
            return results
        else:
            logger.warning(f"Unexpected search response format: {type(results)}")
            return []
            
    except Exception as e:
        logger.error(f"Search failed for '{concept}': {e}")
        return []

def extract_definitions_firecrawl(
    app: FirecrawlApp, 
    urls: List[str], 
    concept: str
) -> List[WebResource]:
    """
    Extract structured definitions from URLs using Firecrawl's extract endpoint.
    
    Args:
        app: Firecrawl client instance
        urls: List of URLs to extract from
        concept: The concept to extract definitions for
        
    Returns:
        List of WebResource objects with extracted definitions
    """
    if not urls:
        return []
    
    try:
        # Build extraction prompt
        prompt = f"""
        Extract comprehensive information about the academic concept "{concept}".
        
        For each occurrence, extract:
        1. A clear, authoritative definition
        2. The academic or technical context
        3. Key characteristics or properties (as a list)
        4. Related concepts mentioned
        5. Assess the source quality (authoritative/reliable/general)
        
        Focus on academic and technical definitions only.
        """
        
        # Define schema for extraction
        schema = {
            "type": "object",
            "properties": {
                "definitions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "concept": {"type": "string"},
                            "definition": {"type": "string"},
                            "context": {"type": "string"},
                            "key_points": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "related_concepts": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "source_quality": {
                                "type": "string",
                                "enum": ["authoritative", "reliable", "general"]
                            }
                        },
                        "required": ["concept", "definition", "context", "source_quality"]
                    }
                }
            },
            "required": ["definitions"]
        }
        
        logger.info(f"Extracting definitions from {len(urls)} URLs for '{concept}'")
        
        # Extract from URLs
        result = app.extract(
            urls=urls,
            prompt=prompt,
            schema=schema
        )
        
        # Process results
        resources = []
        
        # Handle different response formats
        extracted_data = {}
        if isinstance(result, dict):
            if 'data' in result:
                # New format: {'data': [...]}
                for item in result['data']:
                    if 'url' in item and 'extracted' in item:
                        extracted_data[item['url']] = item['extracted']
            elif 'results' in result:
                # Alternative format
                for item in result['results']:
                    if 'url' in item and 'extracted' in item:
                        extracted_data[item['url']] = item['extracted']
            else:
                # Direct URL mapping
                extracted_data = result
        
        # Convert to WebResource objects
        for url, data in extracted_data.items():
            if data and 'definitions' in data:
                definitions = []
                for def_data in data['definitions']:
                    definitions.append(ConceptDefinition(**def_data))
                
                resources.append(WebResource(
                    url=url,
                    title=f"Content from {url}",
                    definitions=definitions
                ))
        
        logger.info(f"Extracted {len(resources)} resources with definitions")
        return resources
        
    except Exception as e:
        logger.error(f"Extraction failed for concept '{concept}': {e}")
        return []

async def mine_concept_async(app: FirecrawlApp, concept: str) -> Dict[str, Any]:
    """
    Mine web content for a single concept asynchronously.
    
    Args:
        app: Firecrawl client instance
        concept: The concept to mine
        
    Returns:
        Dictionary with concept definition and sources
    """
    logger.info(f"Mining content for: {concept}")
    
    # Step 1: Search for relevant URLs
    search_results = await asyncio.to_thread(
        search_concept_firecrawl, app, concept, limit=5
    )
    
    if not search_results:
        logger.warning(f"No search results for: {concept}")
        return {
            "concept": concept,
            "resources": [],
            "summary": None,
            "error": "No search results found"
        }
    
    # Extract URLs from search results
    urls = []
    for result in search_results[:MAX_URLS_PER_CONCEPT]:
        if isinstance(result, dict) and 'url' in result:
            urls.append(result['url'])
        elif isinstance(result, str):
            urls.append(result)
    
    if not urls:
        return {
            "concept": concept,
            "resources": [],
            "summary": None,
            "error": "No valid URLs found"
        }
    
    logger.info(f"Found {len(urls)} URLs for {concept}")
    
    # Step 2: Extract definitions from URLs
    resources = await asyncio.to_thread(
        extract_definitions_firecrawl, app, urls, concept
    )
    
    # Step 3: Aggregate and score results
    result = {
        "concept": concept,
        "resources": [],
        "summary": None
    }
    
    # Process resources and create aggregated summary
    all_definitions = []
    for resource in resources:
        # Filter for quality
        quality_definitions = [
            d for d in resource.definitions 
            if d.source_quality in ["authoritative", "reliable"]
        ]
        
        if quality_definitions:
            result["resources"].append({
                "url": resource.url,
                "domain": resource.domain,
                "definitions": [d.model_dump() for d in quality_definitions]
            })
            all_definitions.extend(quality_definitions)
    
    # Create best summary from authoritative sources
    if all_definitions:
        # Prioritize authoritative sources
        auth_defs = [d for d in all_definitions if d.source_quality == "authoritative"]
        best_def = auth_defs[0] if auth_defs else all_definitions[0]
        
        # Aggregate related concepts
        all_related = set()
        for d in all_definitions:
            all_related.update(d.related_concepts)
        
        result["summary"] = {
            "definition": best_def.definition,
            "context": best_def.context,
            "key_points": best_def.key_points,
            "related_concepts": list(all_related)[:5],  # Top 5 related
            "source_count": len(result["resources"])
        }
    
    logger.info(f"Extracted {len(result['resources'])} quality resources for {concept}")
    return result

async def mine_concepts_batch_async(
    app: FirecrawlApp,
    concepts: List[str],
    max_concurrent: int = MAX_CONCURRENT_OPERATIONS
) -> Dict[str, Any]:
    """
    Mine web content for multiple concepts in parallel.
    
    Args:
        app: Firecrawl client instance
        concepts: List of concepts to mine
        max_concurrent: Maximum concurrent operations
        
    Returns:
        Dictionary with all results
    """
    results = {}
    
    # Process in batches
    for i in range(0, len(concepts), BATCH_SIZE):
        batch = concepts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(concepts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} concepts)")
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def mine_with_limit(concept):
            async with semaphore:
                return await mine_concept_async(app, concept)
        
        # Run batch in parallel
        tasks = [mine_with_limit(concept) for concept in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for concept, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to mine {concept}: {result}")
                results[concept] = {
                    "concept": concept,
                    "error": str(result),
                    "resources": [],
                    "summary": None
                }
            else:
                results[concept] = result
        
        # Small delay between batches
        if i + BATCH_SIZE < len(concepts):
            await asyncio.sleep(1)
    
    return results

def mine_concepts_with_firecrawl(
    concepts: List[str],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for mining concepts using Firecrawl SDK.
    
    Args:
        concepts: List of concepts to mine
        output_path: Optional path to save results
        
    Returns:
        Dictionary with all results and statistics
    """
    # Initialize Firecrawl
    app = initialize_firecrawl()
    if not app:
        logger.error("Cannot proceed without Firecrawl client")
        return {
            "error": "Firecrawl not configured",
            "results": {},
            "statistics": {"total": len(concepts), "successful": 0, "failed": len(concepts)}
        }
    
    logger.info(f"Starting web mining for {len(concepts)} concepts using Firecrawl SDK")
    
    start_time = time.time()
    
    # Run async mining
    results = asyncio.run(mine_concepts_batch_async(app, concepts))
    
    # Calculate statistics
    stats = {
        "total_concepts": len(concepts),
        "successful": sum(1 for r in results.values() if r.get("summary")),
        "failed": sum(1 for r in results.values() if "error" in r),
        "total_resources": sum(len(r.get("resources", [])) for r in results.values()),
        "concepts_with_content": sum(1 for r in results.values() if r.get("resources")),
        "processing_time": time.time() - start_time
    }
    
    logger.info(f"Mining complete in {stats['processing_time']:.1f}s")
    logger.info(f"Success rate: {stats['successful']}/{stats['total_concepts']} ({stats['successful']/stats['total_concepts']*100:.1f}%)")
    
    # Prepare output
    output_data = {
        "results": results,
        "statistics": stats,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    
    return output_data

# Backwards compatibility wrapper
def search_and_extract_batch(
    concepts: List[str],
    settings: Optional[Any] = None
) -> Dict[str, Any]:
    """Backwards compatible wrapper for batch mining."""
    return mine_concepts_with_firecrawl(concepts)

if __name__ == "__main__":
    # Test with sample concepts
    test_concepts = [
        "machine learning",
        "deep learning", 
        "neural networks",
        "natural language processing",
        "computer vision"
    ]
    
    print("\n" + "="*60)
    print("Testing Firecrawl SDK Web Mining")
    print("="*60)
    
    if not FIRECRAWL_API_KEY:
        print("\n⚠️  Warning: FIRECRAWL_API_KEY not set in environment")
        print("Set it with: export FIRECRAWL_API_KEY='your-api-key'")
    else:
        print(f"✅ Firecrawl API key configured")
    
    print(f"\nProcessing {len(test_concepts)} test concepts...")
    
    # Run the mining
    results = mine_concepts_with_firecrawl(
        test_concepts,
        output_path="firecrawl_test_results.json"
    )
    
    # Display results
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    stats = results.get("statistics", {})
    print(f"Total concepts: {stats.get('total_concepts', 0)}")
    print(f"Successful: {stats.get('successful', 0)}")
    print(f"Failed: {stats.get('failed', 0)}")
    print(f"Processing time: {stats.get('processing_time', 0):.1f}s")
    print(f"Total resources: {stats.get('total_resources', 0)}")
    
    # Show sample results
    print("\n" + "-"*60)
    print("Sample Results:")
    print("-"*60)
    
    for concept, data in list(results.get("results", {}).items())[:3]:
        print(f"\n📚 {concept}:")
        if data.get("summary"):
            definition = data["summary"]["definition"]
            print(f"  Definition: {definition[:150]}...")
            print(f"  Context: {data['summary']['context']}")
            print(f"  Sources: {data['summary']['source_count']}")
            print(f"  Key points: {len(data['summary'].get('key_points', []))}")
        else:
            print(f"  Status: {data.get('error', 'No content found')}")