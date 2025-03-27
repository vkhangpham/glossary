import re
from typing import List, Dict, Any, Optional, Union, Callable
from bs4 import BeautifulSoup
import logging

# Constants
MIN_LIST_ITEMS = 3  # Minimum number of items for a list to be considered
MAX_LIST_ITEMS = 50  # Maximum number of items for a list to be considered
MIN_ITEM_LENGTH = 5  # Minimum character length for a list item
MAX_ITEM_LENGTH = 100  # Maximum character length for a list item
LENGTH_VARIANCE_THRESHOLD = 0.6  # Maximum allowed variance in item lengths
NON_TERM_THRESHOLD = 0.25  # Maximum percentage of items that can be non-relevant terms

class ListExtractionConfig:
    """Configuration for list extraction"""
    def __init__(self,
                 min_items: int = MIN_LIST_ITEMS,
                 max_items: int = MAX_LIST_ITEMS,
                 min_item_length: int = MIN_ITEM_LENGTH,
                 max_item_length: int = MAX_ITEM_LENGTH,
                 length_variance_threshold: float = LENGTH_VARIANCE_THRESHOLD,
                 non_term_threshold: float = NON_TERM_THRESHOLD,
                 keywords: Optional[List[str]] = None,
                 anti_keywords: Optional[List[str]] = None,
                 patterns: Optional[List[str]] = None):
        self.min_items = min_items
        self.max_items = max_items
        self.min_item_length = min_item_length
        self.max_item_length = max_item_length
        self.length_variance_threshold = length_variance_threshold
        self.non_term_threshold = non_term_threshold
        self.keywords = keywords or []
        self.anti_keywords = anti_keywords or []
        self.patterns = patterns or []


def extract_lists_from_html(html_content: str, config: ListExtractionConfig) -> List[Dict[str, Any]]:
    """
    Extract list items from HTML content with enhanced filtering
    
    Args:
        html_content: HTML content to parse
        config: Configuration for list extraction
    
    Returns:
        List of dictionaries containing list items and metadata
    """
    if not html_content:
        return []
        
    extracted_lists = []
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Extract standard HTML lists (ul, ol)
    html_lists = soup.find_all(['ul', 'ol'])
    for list_elem in html_lists:
        items = list_elem.find_all('li')
        if config.min_items <= len(items) <= config.max_items:
            list_items = [item.get_text().strip() for item in items]
            # Filter out empty items and keep only those within length constraints
            filtered_items = [
                item for item in list_items 
                if item and config.min_item_length <= len(item) <= config.max_item_length
            ]
            
            if len(filtered_items) >= config.min_items:
                # Check HTML structure to filter out navigation/footer lists
                structure_analysis = analyze_html_structure(soup, list_elem)
                
                # If it's clearly navigation, skip this list
                if structure_analysis["nav_score"] > 0.7:
                    continue
                    
                # Check for non-relevant terms ratio
                non_term_ratio = calculate_non_term_ratio(filtered_items, config.anti_keywords)
                if non_term_ratio > config.non_term_threshold:
                    continue
                
                # Check for consistent lengths
                length_consistency = is_consistent_length(filtered_items, config.length_variance_threshold)
                
                # Calculate pattern matches
                pattern_matches = sum(1 for item in filtered_items if has_pattern(item, config.patterns))
                pattern_ratio = pattern_matches / len(filtered_items) if filtered_items else 0
                
                # Count keyword occurrences
                keyword_matches = sum(
                    1 for item in filtered_items 
                    if any(keyword.lower() in item.lower() for keyword in config.keywords)
                )
                keyword_ratio = keyword_matches / len(filtered_items) if filtered_items else 0
                
                extracted_lists.append({
                    "items": filtered_items,
                    "metadata": {
                        "html_type": list_elem.name,
                        "list_size": len(filtered_items),
                        "structure_analysis": structure_analysis,
                        "non_term_ratio": non_term_ratio,
                        "length_consistency": length_consistency,
                        "pattern_ratio": pattern_ratio,
                        "keyword_ratio": keyword_ratio
                    }
                })
    
    # 2. Extract heading-based lists (consecutive h2, h3, h4)
    for heading_tag in ['h2', 'h3', 'h4']:
        headings = soup.find_all(heading_tag)
        if config.min_items <= len(headings) <= config.max_items:
            heading_texts = [h.get_text().strip() for h in headings]
            # Filter out headings that are too long or short or contain typical non-relevant language
            filtered_headings = [
                h for h in heading_texts 
                if h and config.min_item_length <= len(h) <= config.max_item_length and 
                not any(w in h.lower() for w in ['page', 'home', 'about', 'contact'])
            ]
            
            if len(filtered_headings) >= config.min_items:
                # For headings, we can't do structure analysis on a single element
                # So we do it on the first heading's parent
                if headings and headings[0].parent:
                    structure_analysis = analyze_html_structure(soup, headings[0].parent)
                else:
                    structure_analysis = {"is_navigation": False, "is_footer": False, "is_sidebar": False, "nav_score": 0.0}
                
                # Check for non-relevant terms ratio
                non_term_ratio = calculate_non_term_ratio(filtered_headings, config.anti_keywords)
                if non_term_ratio > config.non_term_threshold:
                    continue
                
                # Check for consistent lengths
                length_consistency = is_consistent_length(filtered_headings, config.length_variance_threshold)
                
                # Calculate pattern matches
                pattern_matches = sum(1 for item in filtered_headings if has_pattern(item, config.patterns))
                pattern_ratio = pattern_matches / len(filtered_headings) if filtered_headings else 0
                
                # Count keyword occurrences
                keyword_matches = sum(
                    1 for item in filtered_headings 
                    if any(keyword.lower() in item.lower() for keyword in config.keywords)
                )
                keyword_ratio = keyword_matches / len(filtered_headings) if filtered_headings else 0
                
                extracted_lists.append({
                    "items": filtered_headings,
                    "metadata": {
                        "html_type": f"heading_{heading_tag}",
                        "list_size": len(filtered_headings),
                        "structure_analysis": structure_analysis,
                        "non_term_ratio": non_term_ratio,
                        "length_consistency": length_consistency,
                        "pattern_ratio": pattern_ratio,
                        "keyword_ratio": keyword_ratio
                    }
                })
    
    # 3. Look for div or span elements that might contain lists
    potential_list_containers = soup.find_all(['div', 'section', 'article'])
    for container in potential_list_containers:
        # Check if container has many similar child elements (potential list items)
        children = container.find_all(['p', 'div', 'span', 'a'], recursive=False)
        if config.min_items <= len(children) <= config.max_items:
            child_texts = [child.get_text().strip() for child in children]
            # Filter out empty or long/short texts
            filtered_texts = [
                text for text in child_texts 
                if text and config.min_item_length <= len(text) <= config.max_item_length
            ]
            
            if len(filtered_texts) >= config.min_items:
                # Check HTML structure
                structure_analysis = analyze_html_structure(soup, container)
                
                # If it's clearly navigation, skip this list
                if structure_analysis["nav_score"] > 0.7:
                    continue
                
                # Check for non-relevant terms ratio
                non_term_ratio = calculate_non_term_ratio(filtered_texts, config.anti_keywords)
                if non_term_ratio > config.non_term_threshold:
                    continue
                
                # Check for consistent lengths
                length_consistency = is_consistent_length(filtered_texts, config.length_variance_threshold)
                
                # Calculate pattern matches
                pattern_matches = sum(1 for item in filtered_texts if has_pattern(item, config.patterns))
                pattern_ratio = pattern_matches / len(filtered_texts) if filtered_texts else 0
                
                # Count keyword occurrences
                keyword_matches = sum(
                    1 for item in filtered_texts 
                    if any(keyword.lower() in item.lower() for keyword in config.keywords)
                )
                keyword_ratio = keyword_matches / len(filtered_texts) if filtered_texts else 0
                
                extracted_lists.append({
                    "items": filtered_texts,
                    "metadata": {
                        "html_type": f"container_{container.name}",
                        "list_size": len(filtered_texts),
                        "structure_analysis": structure_analysis,
                        "non_term_ratio": non_term_ratio,
                        "length_consistency": length_consistency,
                        "pattern_ratio": pattern_ratio,
                        "keyword_ratio": keyword_ratio
                    }
                })
                
    # 4. Look for tables that might contain lists
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        if config.min_items <= len(rows) <= config.max_items:
            # Extract text from first cell of each row
            cells = []
            for row in rows:
                row_cells = row.find_all(['td', 'th'])
                if row_cells:
                    cells.append(row_cells[0].get_text().strip())
            
            # Filter out empty or long/short texts
            filtered_cells = [
                text for text in cells 
                if text and config.min_item_length <= len(text) <= config.max_item_length
            ]
            
            if len(filtered_cells) >= config.min_items:
                # Check HTML structure
                structure_analysis = analyze_html_structure(soup, table)
                
                # If it's clearly navigation, skip this list
                if structure_analysis["nav_score"] > 0.7:
                    continue
                
                # Check for non-relevant terms ratio
                non_term_ratio = calculate_non_term_ratio(filtered_cells, config.anti_keywords)
                if non_term_ratio > config.non_term_threshold:
                    continue
                
                # Check for consistent lengths
                length_consistency = is_consistent_length(filtered_cells, config.length_variance_threshold)
                
                # Calculate pattern matches
                pattern_matches = sum(1 for item in filtered_cells if has_pattern(item, config.patterns))
                pattern_ratio = pattern_matches / len(filtered_cells) if filtered_cells else 0
                
                # Count keyword occurrences
                keyword_matches = sum(
                    1 for item in filtered_cells 
                    if any(keyword.lower() in item.lower() for keyword in config.keywords)
                )
                keyword_ratio = keyword_matches / len(filtered_cells) if filtered_cells else 0
                
                extracted_lists.append({
                    "items": filtered_cells,
                    "metadata": {
                        "html_type": "table",
                        "list_size": len(filtered_cells),
                        "structure_analysis": structure_analysis,
                        "non_term_ratio": non_term_ratio,
                        "length_consistency": length_consistency,
                        "pattern_ratio": pattern_ratio,
                        "keyword_ratio": keyword_ratio
                    }
                })
    
    # 5. Look for definition lists
    dl_lists = soup.find_all('dl')
    for dl in dl_lists:
        terms = dl.find_all('dt')
        if config.min_items <= len(terms) <= config.max_items:
            term_texts = [term.get_text().strip() for term in terms]
            # Filter out empty or long/short texts
            filtered_terms = [
                text for text in term_texts 
                if text and config.min_item_length <= len(text) <= config.max_item_length
            ]
            
            if len(filtered_terms) >= config.min_items:
                # Check HTML structure
                structure_analysis = analyze_html_structure(soup, dl)
                
                # If it's clearly navigation, skip this list
                if structure_analysis["nav_score"] > 0.7:
                    continue
                
                # Check for non-relevant terms ratio
                non_term_ratio = calculate_non_term_ratio(filtered_terms, config.anti_keywords)
                if non_term_ratio > config.non_term_threshold:
                    continue
                
                # Check for consistent lengths
                length_consistency = is_consistent_length(filtered_terms, config.length_variance_threshold)
                
                # Calculate pattern matches
                pattern_matches = sum(1 for item in filtered_terms if has_pattern(item, config.patterns))
                pattern_ratio = pattern_matches / len(filtered_terms) if filtered_terms else 0
                
                # Count keyword occurrences
                keyword_matches = sum(
                    1 for item in filtered_terms 
                    if any(keyword.lower() in item.lower() for keyword in config.keywords)
                )
                keyword_ratio = keyword_matches / len(filtered_terms) if filtered_terms else 0
                
                extracted_lists.append({
                    "items": filtered_terms,
                    "metadata": {
                        "html_type": "dl",
                        "list_size": len(filtered_terms),
                        "structure_analysis": structure_analysis,
                        "non_term_ratio": non_term_ratio,
                        "length_consistency": length_consistency,
                        "pattern_ratio": pattern_ratio,
                        "keyword_ratio": keyword_ratio
                    }
                })
    
    # 6. Look for bullet/numbered list patterns in text paragraphs
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        text = p.get_text()
        
        # Check for bullet patterns
        bullet_matches = re.findall(r'[•\-\*\−\–\—]\s*([^•\-\*\−\–\—\n\r]{3,100}?)(?=[•\-\*\−\–\—]|\n|\r|$)', text)
        if config.min_items <= len(bullet_matches) <= config.max_items:
            filtered_bullets = [
                match.strip() for match in bullet_matches 
                if match.strip() and config.min_item_length <= len(match.strip()) <= config.max_item_length
            ]
            
            if len(filtered_bullets) >= config.min_items:
                # Check HTML structure for paragraphs
                structure_analysis = analyze_html_structure(soup, p)
                
                # If it's clearly navigation, skip this list
                if structure_analysis["nav_score"] > 0.7:
                    continue
                
                # Check for non-relevant terms ratio
                non_term_ratio = calculate_non_term_ratio(filtered_bullets, config.anti_keywords)
                if non_term_ratio > config.non_term_threshold:
                    continue
                
                # Check for consistent lengths
                length_consistency = is_consistent_length(filtered_bullets, config.length_variance_threshold)
                
                # Calculate pattern matches
                pattern_matches = sum(1 for item in filtered_bullets if has_pattern(item, config.patterns))
                pattern_ratio = pattern_matches / len(filtered_bullets) if filtered_bullets else 0
                
                # Count keyword occurrences
                keyword_matches = sum(
                    1 for item in filtered_bullets 
                    if any(keyword.lower() in item.lower() for keyword in config.keywords)
                )
                keyword_ratio = keyword_matches / len(filtered_bullets) if filtered_bullets else 0
                
                extracted_lists.append({
                    "items": filtered_bullets,
                    "metadata": {
                        "html_type": "p_bullet",
                        "list_size": len(filtered_bullets),
                        "structure_analysis": structure_analysis,
                        "non_term_ratio": non_term_ratio,
                        "length_consistency": length_consistency,
                        "pattern_ratio": pattern_ratio,
                        "keyword_ratio": keyword_ratio
                    }
                })
            
        # Check for numbered patterns
        numbered_matches = re.findall(r'(\d+[\.\)]\s*[^0-9\.\)\n\r]{3,100}?)(?=\d+[\.\)]|\n|\r|$)', text)
        if config.min_items <= len(numbered_matches) <= config.max_items:
            filtered_numbered = [
                match.strip() for match in numbered_matches 
                if match.strip() and config.min_item_length <= len(match.strip()) <= config.max_item_length
            ]
            
            if len(filtered_numbered) >= config.min_items:
                # Check HTML structure for paragraphs
                structure_analysis = analyze_html_structure(soup, p)
                
                # If it's clearly navigation, skip this list
                if structure_analysis["nav_score"] > 0.7:
                    continue
                
                # Check for non-relevant terms ratio
                non_term_ratio = calculate_non_term_ratio(filtered_numbered, config.anti_keywords)
                if non_term_ratio > config.non_term_threshold:
                    continue
                
                # Check for consistent lengths
                length_consistency = is_consistent_length(filtered_numbered, config.length_variance_threshold)
                
                # Calculate pattern matches
                pattern_matches = sum(1 for item in filtered_numbered if has_pattern(item, config.patterns))
                pattern_ratio = pattern_matches / len(filtered_numbered) if filtered_numbered else 0
                
                # Count keyword occurrences
                keyword_matches = sum(
                    1 for item in filtered_numbered 
                    if any(keyword.lower() in item.lower() for keyword in config.keywords)
                )
                keyword_ratio = keyword_matches / len(filtered_numbered) if filtered_numbered else 0
                
                extracted_lists.append({
                    "items": filtered_numbered,
                    "metadata": {
                        "html_type": "p_numbered",
                        "list_size": len(filtered_numbered),
                        "structure_analysis": structure_analysis,
                        "non_term_ratio": non_term_ratio,
                        "length_consistency": length_consistency,
                        "pattern_ratio": pattern_ratio,
                        "keyword_ratio": keyword_ratio
                    }
                })
    
    return extracted_lists


def analyze_html_structure(html_soup: BeautifulSoup, list_element) -> Dict[str, Any]:
    """
    Analyze HTML structure to determine if a list element is likely part of navigation/footer
    
    Args:
        html_soup: BeautifulSoup object of the whole page
        list_element: The list element to analyze
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "is_navigation": False,
        "is_footer": False,
        "is_sidebar": False,
        "nav_score": 0.0
    }
    
    # Check ancestors for navigation indicators
    ancestors = list(list_element.parents)
    
    # Check for navigation elements
    for ancestor in ancestors:
        # Check tag name
        if ancestor.name in ['nav', 'header', 'footer', 'aside']:
            if ancestor.name == 'nav' or ancestor.name == 'header':
                results["is_navigation"] = True
                results["nav_score"] += 0.8
            elif ancestor.name == 'footer':
                results["is_footer"] = True
                results["nav_score"] += 0.9
            elif ancestor.name == 'aside':
                results["is_sidebar"] = True
                results["nav_score"] += 0.7
                
        # Check class and id attributes
        if ancestor.get('class'):
            ancestor_classes = ' '.join(ancestor.get('class', [])).lower()
            if any(nav_term in ancestor_classes for nav_term in ['nav', 'menu', 'header', 'footer', 'sidebar', 'topbar']):
                if 'nav' in ancestor_classes or 'menu' in ancestor_classes:
                    results["is_navigation"] = True
                    results["nav_score"] += 0.6
                elif 'footer' in ancestor_classes:
                    results["is_footer"] = True
                    results["nav_score"] += 0.7
                elif 'sidebar' in ancestor_classes:
                    results["is_sidebar"] = True
                    results["nav_score"] += 0.5
                
        if ancestor.get('id'):
            ancestor_id = ancestor.get('id', '').lower()
            if any(nav_term in ancestor_id for nav_term in ['nav', 'menu', 'header', 'footer', 'sidebar']):
                if 'nav' in ancestor_id or 'menu' in ancestor_id:
                    results["is_navigation"] = True
                    results["nav_score"] += 0.6
                elif 'footer' in ancestor_id:
                    results["is_footer"] = True
                    results["nav_score"] += 0.7
                elif 'sidebar' in ancestor_id:
                    results["is_sidebar"] = True
                    results["nav_score"] += 0.5
    
    # Normalize nav_score to 0-1
    results["nav_score"] = min(1.0, results["nav_score"])
    
    return results


def calculate_non_term_ratio(items: List[str], anti_keywords: List[str]) -> float:
    """
    Calculate the ratio of items containing non-relevant terms
    
    Args:
        items: List of items to check
        anti_keywords: List of keywords indicating non-relevant terms
        
    Returns:
        Ratio of items containing non-relevant terms (0-1)
    """
    non_term_count = 0
    
    for item in items:
        item_lower = item.lower()
        if any(term.lower() in item_lower for term in anti_keywords):
            non_term_count += 1
    
    # Avoid division by zero
    if not items:
        return 1.0
    
    return non_term_count / len(items)


def has_pattern(text: str, patterns: List[str]) -> bool:
    """
    Check if text matches patterns in the list
    
    Args:
        text: Text to check
        patterns: List of regex patterns to check against
        
    Returns:
        Boolean indicating if text contains one of the patterns
    """
    text = text.lower()
    
    # Check against regex patterns
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def is_consistent_length(items: List[str], threshold: float = LENGTH_VARIANCE_THRESHOLD) -> bool:
    """
    Check if items in a list have consistent lengths
    
    Args:
        items: List of items to check
        threshold: Maximum allowed variance in item lengths (normalized)
        
    Returns:
        Boolean indicating if items have consistent lengths
    """
    if not items:
        return False
    
    # Calculate length statistics
    lengths = [len(item) for item in items]
    avg_length = sum(lengths) / len(lengths)
    
    # Calculate variance
    variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
    
    # Normalize variance to a 0-1 scale
    max_possible_variance = avg_length ** 2  # Theoretical maximum variance
    normalized_variance = min(1.0, variance / max_possible_variance if max_possible_variance > 0 else 1.0)
    
    # Check if normalized variance is below threshold
    return normalized_variance <= threshold


def score_list(items: List[str],
             metadata: Dict[str, Any],
             context_term: str,
             keywords: List[str],
             scoring_weights: Optional[Dict[str, float]] = None,
             additional_scoring_fn: Optional[Callable[[List[str], str], float]] = None) -> float:
    """
    Score a list based on multiple heuristics and metadata
    
    Args:
        items: List of items
        metadata: Metadata about the list from extraction
        context_term: Related term for context
        keywords: List of relevant keywords to check items against
        scoring_weights: Optional custom weights for different scoring factors
        additional_scoring_fn: Optional additional scoring function
        
    Returns:
        Quality score between 0 and 1
    """
    if not items:
        return 0.0
    
    # Default weights
    weights = {
        "keyword": 0.25,      # Keyword occurrence weight
        "structure": 0.15,    # HTML structure weight
        "pattern": 0.20,      # Naming pattern weight
        "non_term": 0.15,     # Absence of non-relevant terms weight
        "consistency": 0.10,  # Consistency in formatting weight
        "size": 0.05,         # Appropriate list size weight
        "html_type": 0.10     # HTML element type appropriateness weight
    }
    
    # Override with custom weights if provided
    if scoring_weights:
        weights.update(scoring_weights)
    
    # Retrieve metadata values (with defaults in case something is missing)
    html_type = metadata.get("html_type", "unknown")
    structure_analysis = metadata.get("structure_analysis", {"nav_score": 0.0})
    non_term_ratio = metadata.get("non_term_ratio", 0.5)
    length_consistency = metadata.get("length_consistency", False)
    pattern_ratio = metadata.get("pattern_ratio", 0.0)
    keyword_ratio = metadata.get("keyword_ratio", 0.0)
    list_size = metadata.get("list_size", len(items))
    
    # 1. HTML structure score - lower score for navigation/footer elements
    structure_score = 1.0 - structure_analysis.get("nav_score", 0.0)
    
    # 2. Keyword score - check both predefined keywords and context term
    # Add context_term to the scoring as a special case
    context_matches = sum(1 for item in items if context_term.lower() in item.lower())
    context_score = min(1.0, context_matches / len(items) * 2)  # Scale to 0-1
    
    # Combined keyword score (with context_term given higher weight)
    keyword_score = (keyword_ratio * 0.7) + (context_score * 0.3)
    
    # 3. Pattern matching score
    pattern_score = pattern_ratio
    
    # 4. Non-term ratio score (inverse - lower ratio is better)
    non_term_score = 1.0 - non_term_ratio
    
    # 5. Length consistency score
    consistency_score = 1.0 if length_consistency else 0.5
    
    # 6. List size score (prefer medium-sized lists, not too small or large)
    if list_size < 8:
        size_score = 0.5
    elif list_size < 15:
        size_score = 0.8
    elif list_size < 30:
        size_score = 1.0
    elif list_size < 40:
        size_score = 0.7
    else:
        size_score = 0.5
    
    # 7. HTML type score (prioritize certain HTML structures)
    if html_type in ['ul', 'ol']:
        html_type_score = 1.0  # Standard lists are most reliable
    elif html_type.startswith('heading_'):
        html_type_score = 0.9  # Heading-based lists are quite reliable
    elif html_type == 'table':
        html_type_score = 0.8  # Tables are good but sometimes contain other data
    elif html_type == 'dl':
        html_type_score = 0.9  # Definition lists are often used for terms
    elif html_type.startswith('container_'):
        html_type_score = 0.7  # Container-based lists are less reliable
    elif html_type in ['p_bullet', 'p_numbered']:
        html_type_score = 0.8  # Text-based lists can be good but need more filtering
    else:
        html_type_score = 0.5  # Unknown types get average score
    
    # Apply additional scoring function if provided
    additional_score = 0.0
    if additional_scoring_fn:
        additional_score = additional_scoring_fn(items, context_term)
    
    # Weighted quality score
    quality_score = (
        keyword_score * weights["keyword"] +
        structure_score * weights["structure"] +
        pattern_score * weights["pattern"] +
        non_term_score * weights["non_term"] +
        consistency_score * weights["consistency"] +
        size_score * weights["size"] +
        html_type_score * weights["html_type"]
    )
    
    # Add additional score with equal weight as keyword score if it exists
    if additional_scoring_fn:
        # Adjust other weights to accommodate the additional score
        quality_score = quality_score * 0.8 + additional_score * 0.2
    
    return min(1.0, quality_score) 