"""
Shared web extraction functionality for levels 1-3 using Firecrawl v2.0.

This module leverages the new simplified mining API and Firecrawl v2.0 features
for 500% performance improvement with smart crawling, natural language prompts,
JSON extraction, and summary capabilities.
"""

import os
import json
import time
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import Counter

from generate_glossary.utils.error_handler import (
    ExternalServiceError, ValidationError, handle_error, processing_context
)
from generate_glossary.utils.logger import get_logger, log_processing_step
from generate_glossary.config import ensure_directories
from generate_glossary.mining import mine_concepts, ConceptDefinition, WebResource
from generate_glossary.config import get_level_config, get_web_extraction_config


def load_input_terms(input_file: str) -> List[str]:
    """Load terms from input file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def create_level_specific_smart_prompt(level: int) -> str:
    """
    Create level-specific smart prompts for Firecrawl v2.0 extraction.
    
    These prompts leverage v2.0's natural language understanding for
    better academic content extraction than manual parsing.
    """
    if level == 1:
        return """
        Extract academic departments and programs that are explicitly part of this institution.
        Focus on: 1) Official department names 2) Academic programs 3) Schools within the institution
        Exclude: Website navigation, general information, non-academic content
        Prioritize: .edu domains, official institutional pages, department directories
        """
    elif level == 2:
        return """
        Extract research areas, specializations, and academic focus areas.
        Focus on: 1) Research groups 2) Academic specializations 3) Laboratory areas 4) Research centers
        Exclude: General academic terms, administrative content, non-research information
        Prioritize: Research pages, faculty profiles, lab descriptions, academic publications
        """
    elif level == 3:
        return """
        Extract conference topics, research themes, and academic subjects.
        Focus on: 1) Conference tracks 2) Research themes 3) Academic topics 4) Special interest areas
        Exclude: Conference logistics, general information, administrative details
        Prioritize: Conference websites, call for papers, academic event descriptions
        """
    else:
        return """
        Extract relevant academic concepts and terminology.
        Focus on: 1) Academic terms 2) Educational concepts 3) Institutional information
        Prioritize: Educational and academic sources
        """


def as_dict(obj) -> Dict[str, Any]:
    """Convert Pydantic models to dicts with fallback for dict-like objects."""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif isinstance(obj, dict):
        return obj
    else:
        return {}


def extract_summary_content(summary) -> List[str]:
    """Extract content from summary with flexible shape handling."""
    if not summary:
        return []
    
    content = []
    
    # Handle different summary shapes
    if isinstance(summary, list):
        # Summary is a list - extract strings
        content.extend([str(item) for item in summary if item])
    elif isinstance(summary, dict):
        # Check for various key patterns
        for key in ['key_points', 'bullets', 'highlights', 'points']:
            if key in summary and isinstance(summary[key], list):
                content.extend(summary[key])
        
        # Check for nested content
        if 'content' in summary and isinstance(summary['content'], dict):
            nested = summary['content']
            for key in ['key_points', 'bullets', 'highlights']:
                if key in nested and isinstance(nested[key], list):
                    content.extend(nested[key])
        
        # If no specific keys found, try to extract any list values
        if not content:
            for value in summary.values():
                if isinstance(value, list) and all(isinstance(item, str) for item in value):
                    content.extend(value)
    elif isinstance(summary, str):
        # Summary is a string - split or use as-is
        content.append(summary)
    
    return [str(item).strip() for item in content if item and str(item).strip()]


def process_mining_results(
    mining_results: Dict[str, Any], 
    input_terms: List[str], 
    level: int, 
    max_results_per_term: int
) -> Dict[str, List[str]]:
    """
    Process results from the new mining API into the expected format.
    
    The new API returns structured ConceptDefinition and WebResource objects,
    which we convert to the simple string format expected by the existing pipeline.
    Supports both dict and Pydantic model inputs with flexible structure handling.
    """
    logger = get_logger(f"lv{level}.s0")
    all_results = {}
    
    # Harden against various mining result structures
    try:
        # Check for different result structures
        if 'results' in mining_results:
            api_results = mining_results.get('results', {})
        elif 'definitions' in mining_results:
            # Top-level definitions structure
            api_results = {term: {'definitions': mining_results['definitions']} for term in input_terms}
        else:
            # Use mining_results directly if it looks like results
            api_results = mining_results if isinstance(mining_results, dict) else {}
    except Exception as e:
        logger.warning(f"Error accessing mining results structure: {e}")
        api_results = {}
    
    for term in input_terms:
        term_results = []
        
        if term in api_results:
            try:
                term_data = api_results[term]
                
                # Handle case where term_data might be a Pydantic model
                term_dict = as_dict(term_data)
                
                # Extract from resources (WebResource objects or dicts)
                resources = term_dict.get('resources', [])
                for resource in resources:
                    try:
                        # Convert resource to dict if it's a Pydantic model
                        resource_dict = as_dict(resource)
                        
                        # Handle missing definitions gracefully
                        definitions = resource_dict.get('definitions', [])
                        if not definitions:
                            logger.debug(f"Resource missing definitions for term: {term}")
                            continue
                            
                        for definition in definitions:
                            # Convert definition to dict if it's a Pydantic model
                            def_dict = as_dict(definition)
                            
                            # Extract relevant academic items based on level
                            if level == 1:  # Departments
                                # Use the concept name if it looks like a department
                                concept = def_dict.get('concept', '')
                                context = def_dict.get('context', '').lower()
                                
                                # Include if it has department keywords OR context indicates department
                                has_dept_keywords = any(keyword in concept.lower() for keyword in ['department', 'school', 'college', 'program'])
                                is_dept_context = any(keyword in context for keyword in ['department', 'school', 'academic', 'faculty'])
                                
                                if has_dept_keywords or is_dept_context:
                                    term_results.append(concept)
                                
                                # Also extract related concepts that might be departments
                                related = def_dict.get('related_concepts', [])
                                for r in related:
                                    if any(keyword in r.lower() for keyword in ['department', 'school']) or is_dept_context:
                                        term_results.append(r)
                            
                            elif level == 2:  # Research areas
                                concept = def_dict.get('concept', '')
                                if any(keyword in concept.lower() for keyword in ['research', 'lab', 'group', 'center', 'area']):
                                    term_results.append(concept)
                                related = def_dict.get('related_concepts', [])
                                term_results.extend([r for r in related if any(keyword in r.lower() for keyword in ['research', 'lab', 'area'])])
                            
                            elif level == 3:  # Conference topics
                                concept = def_dict.get('concept', '')
                                term_results.append(concept)
                                # For conference topics, include all related concepts
                                related = def_dict.get('related_concepts', [])
                                term_results.extend(related)
                            
                            else:  # General extraction
                                concept = def_dict.get('concept', '')
                                if concept:
                                    term_results.append(concept)
                    except Exception as e:
                        logger.debug(f"Error processing resource for term {term}: {e}")
                        continue
                
                # Extract from summary with flexible shape handling
                summary = term_dict.get('summary')
                summary_content = extract_summary_content(summary)
                term_results.extend(summary_content)
                
            except Exception as e:
                logger.warning(f"Error processing term data for '{term}': {e}")
                continue
        
        # Clean and deduplicate results
        cleaned_results = []
        seen = set()
        for item in term_results:
            if isinstance(item, str) and item.strip() and item.strip().lower() not in seen:
                cleaned_item = item.strip()
                if 3 <= len(cleaned_item) <= 100:  # Reasonable length
                    cleaned_results.append(cleaned_item)
                    seen.add(cleaned_item.lower())
        
        # Limit results per term
        all_results[term] = cleaned_results[:max_results_per_term]
        
        if cleaned_results:
            logger.debug(f"Extracted {len(cleaned_results)} items for '{term}' (showing {len(all_results[term])})")
        else:
            logger.warning(f"No valid items extracted for term: {term}")
    
    return all_results


def save_results(results: Dict[str, List[str]], output_file: str, metadata_file: str, level: int, mining_results: Optional[Dict[str, Any]] = None):
    """Save extraction results to files."""
    with processing_context(f"save_web_extraction_results_lv{level}") as correlation_id:
        logger = get_logger(f"lv{level}.s0")
        
        log_processing_step(
            logger,
            f"save_web_extraction_results_lv{level}",
            "started",
            {
                "terms_processed": len(results),
                "total_items": sum(len(items) for items in results.values())
            }
        )
        
        try:
            web_config = get_web_extraction_config()
            # Prepare output data
            all_items = []
            for term, items in results.items():
                for item in items:
                    all_items.append(f"{term} - {item}")
            
            # Shuffle for variety
            random.shuffle(all_items)
            
            # Save main output file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in all_items:
                    f.write(item + '\n')
            
            # Save metadata
            metadata = {
                'level': level,
                'total_terms_processed': len(results),
                'total_items_extracted': len(all_items),
                'items_per_term': {term: len(items) for term, items in results.items()},
                'processing_timestamp': time.time(),
                'extractor': 'firecrawl_v2.0',
                'config_used': {
                    'batch_size': web_config.batch_size,
                    'max_results_per_term': web_config.max_results_per_term,
                    'llm_attempts': web_config.num_llm_attempts
                }
            }
            
            # Add features_used for parity with return metadata
            if mining_results:
                features_used = mining_results.get('statistics', {}).get('features_used', {})
                metadata['features_used'] = features_used
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            log_processing_step(
                logger,
                f"save_web_extraction_results_lv{level}",
                "completed",
                {
                    "items_saved": len(all_items),
                    "output_file": output_file,
                    "metadata_file": metadata_file
                }
            )
            
            logger.info(f"Saved {len(all_items)} items to {output_file}")
            if results:
                logger.info(f"Processed {len(results)} terms with average {len(all_items)/len(results):.1f} items per term")
                
        except Exception as e:
            handle_error(
                e,
                context={
                    "terms_count": len(results),
                    "total_items": sum(len(items) for items in results.values()),
                    "output_file": output_file,
                    "metadata_file": metadata_file,
                    "level": level,
                    "correlation_id": correlation_id
                },
                operation=f"save_web_extraction_results_lv{level}",
                reraise=True
            )


def extract_web_content(
    input_file: str,
    level: int,
    output_file: str,
    metadata_file: str
) -> Dict[str, Any]:
    """
    Generic web content extraction for any level using Firecrawl v2.0.
    
    This leverages ALL Firecrawl v2.0 features for 500% performance improvement
    and dramatically simplified implementation with better accuracy.
    """
    with processing_context(f"web_extraction_lv{level}") as correlation_id:
        logger = get_logger(f"lv{level}.s0")
        config = get_level_config(level)
        web_config = get_web_extraction_config()
        
        log_processing_step(
            logger,
            f"web_extraction_lv{level}",
            "started",
            {
                "input_file": input_file,
                "output_file": output_file,
                "level": level,
                "firecrawl_version": "v2.0"
            }
        )
        
        # Ensure directories exist
        ensure_directories(level)
        
        try:
            logger.info(f"Starting Level {level} web extraction with Firecrawl v2.0: {config.processing_description}")
            
            # Load input terms
            input_terms = load_input_terms(input_file)
            logger.info(f"Loaded {len(input_terms)} input terms")
            
            # Handle empty input terms gracefully
            if not input_terms:
                logger.info("No input terms provided, skipping mining and producing empty outputs")
                empty_results = {}
                save_results(empty_results, output_file, metadata_file, level)
                
                log_processing_step(
                    logger,
                    f"web_extraction_lv{level}",
                    "completed",
                    {
                        "terms_processed": 0,
                        "items_extracted": 0,
                        "average_items_per_term": 0,
                        "early_exit": "empty_input_terms"
                    }
                )
                
                return {
                    'level': level,
                    'input_terms_count': 0,
                    'processed_terms_count': 0,
                    'extracted_terms_count': 0,
                    'average_items_per_term': 0,
                    'processing_description': config.processing_description,
                    'extractor': 'firecrawl_v2.0',
                    'features_used': {}
                }
            
            # Create level-specific smart prompt for v2.0 extraction
            smart_prompt = create_level_specific_smart_prompt(level)
            
            # Use the new unified mining API with v2.0 features
            mining_results = mine_concepts(
                concepts=input_terms,
                max_concurrent=web_config.max_concurrent_mining,
                max_age=172800000,  # 2 days cache for academic content
                use_summary=True,   # Use v2.0 summary format for optimization
                use_batch_scrape=True,  # Enable 500% performance improvement
                summary_prompt=smart_prompt,  # Level-specific academic extraction
                use_hybrid=True     # Best of both batch + smart extraction
            )
            
            # Process results from the new API format
            all_results = process_mining_results(
                mining_results, 
                input_terms, 
                level, 
                web_config.max_results_per_term
            )
            
            # Save results in the expected format
            save_results(all_results, output_file, metadata_file, level, mining_results)
            
            total_extracted = sum(len(items) for items in all_results.values())
            
            log_processing_step(
                logger,
                f"web_extraction_lv{level}",
                "completed",
                {
                    "terms_processed": len(input_terms),
                    "items_extracted": total_extracted,
                    "average_items_per_term": total_extracted / len(input_terms) if input_terms else 0,
                    "firecrawl_features_used": mining_results.get("statistics", {}).get("features_used", {})
                }
            )
            
            return {
                'level': level,
                'input_terms_count': len(input_terms),
                'processed_terms_count': len(all_results),
                'extracted_terms_count': total_extracted,
                'average_items_per_term': total_extracted / len(input_terms) if input_terms else 0,
                'processing_description': config.processing_description,
                'extractor': 'firecrawl_v2.0',
                'features_used': mining_results.get("statistics", {}).get("features_used", {})
            }
            
        except Exception as e:
            handle_error(
                e,
                context={
                    "input_file": input_file,
                    "output_file": output_file,
                    "level": level,
                    "correlation_id": correlation_id
                },
                operation=f"web_extraction_lv{level}",
                reraise=True
            )


# Alias for backward compatibility
extract_web_content_simple = extract_web_content