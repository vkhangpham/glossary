#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import logging
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_data(level: int, base_dir: str = "data/final") -> Tuple[Dict, Dict]:
    """
    Load metadata and resources for a specific level.
    
    Args:
        level: The hierarchy level (0, 1, 2, 3)
        base_dir: Base directory for data
        
    Returns:
        Tuple of (metadata, resources)
    """
    level_dir = Path(base_dir) / f"lv{level}"
    
    # Load metadata
    metadata_file = level_dir / f"lv{level}_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load resources
    resources_file = level_dir / f"lv{level}_filtered_resources.json"
    resources = {}
    
    if resources_file.exists():
        with open(resources_file, 'r', encoding='utf-8') as f:
            resources = json.load(f)
    
    logger.info(f"Loaded metadata for {len(metadata)} terms from level {level}")
    logger.info(f"Loaded resources for {len(resources)} terms from level {level}")
    
    return metadata, resources

def extract_canonical_terms(metadata: Dict) -> Set[str]:
    """Extract canonical terms (terms that are not variations)."""
    return {term for term, data in metadata.items() if 'canonical_term' not in data}

def find_siblings(metadata: Dict) -> Dict[str, Dict[str, Set[str]]]:
    """
    Find siblings (terms with the same parent).
    
    Returns:
        Dict mapping parent terms to sets of sibling terms
    """
    parent_to_children = defaultdict(set)
    
    # Only consider canonical terms
    canonical_terms = extract_canonical_terms(metadata)
    
    for term in canonical_terms:
        term_data = metadata[term]
        parents = term_data.get('parents', [])
        
        # Add term to each parent's children
        for parent in parents:
            parent_to_children[parent].add(term)
    
    # Filter out parents with only one child (no siblings)
    sibling_groups = {
        parent: children 
        for parent, children in parent_to_children.items() 
        if len(children) > 1
    }
    
    logger.info(f"Found {len(sibling_groups)} parent terms with multiple children")
    
    return sibling_groups

def load_transformer_model():
    """Load a transformer model for text embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except ImportError:
        logger.error("SentenceTransformer not installed. Please install with: pip install sentence-transformers")
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info("Using Hugging Face transformers instead...")
            
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            def mean_pooling(model_output, attention_mask):
                # Mean pooling - take attention mask into account for correct averaging
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            def encode_texts(texts):
                # Tokenize sentences
                encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                
                # Compute token embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)
                
                # Perform pooling
                embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return embeddings.numpy()
            
            return encode_texts
        except ImportError:
            logger.error("Neither sentence-transformers nor transformers are installed.")
            logger.error("Please install one of them: pip install sentence-transformers or pip install transformers")
            logger.error("Falling back to term strings for embeddings (not recommended)")
            return None

def calculate_embeddings(terms: Set[str], resources: Dict) -> Dict[str, np.ndarray]:
    """
    Calculate transformer embeddings for terms based on their concatenated resource content.
    
    Args:
        terms: Set of terms to calculate embeddings for
        resources: Dictionary mapping terms to their resources
        
    Returns:
        Dictionary mapping terms to their embeddings
    """
    # Load the transformer model
    model = load_transformer_model()
    
    # Collect all processed content for each term
    term_to_content = {}
    for term in terms:
        if term in resources and resources[term]:
            # Concatenate all processed content for the term
            all_content = " ".join([
                resource.get('processed_content', '')
                for resource in resources[term]
                if 'processed_content' in resource
            ])
            
            if all_content:
                term_to_content[term] = all_content
            else:
                # Fallback to term itself if no content available
                term_to_content[term] = term
        else:
            # Fallback to term itself if no resources available
            term_to_content[term] = term
    
    # Get terms in a consistent order
    ordered_terms = list(term_to_content.keys())
    ordered_content = [term_to_content[term] for term in ordered_terms]
    
    # Calculate embeddings
    embeddings = {}
    
    if model is not None:
        try:
            # Check if model is a SentenceTransformer instance or a function
            if hasattr(model, 'encode'):
                # SentenceTransformer model
                logger.info("Generating embeddings with SentenceTransformer...")
                batch_size = 32  # Adjust based on your system's memory
                
                # Process in batches to avoid memory issues
                for i in range(0, len(ordered_content), batch_size):
                    batch = ordered_content[i:i+batch_size]
                    batch_terms = ordered_terms[i:i+batch_size]
                    batch_embeddings = model.encode(batch)
                    
                    for j, term in enumerate(batch_terms):
                        embeddings[term] = batch_embeddings[j]
                    
                    logger.info(f"Processed {i+len(batch)}/{len(ordered_content)} embeddings")
            else:
                # Custom function from transformers fallback
                logger.info("Generating embeddings with Hugging Face transformers...")
                batch_size = 32  # Adjust based on your system's memory
                
                for i in range(0, len(ordered_content), batch_size):
                    batch = ordered_content[i:i+batch_size]
                    batch_terms = ordered_terms[i:i+batch_size]
                    batch_embeddings = model(batch)
                    
                    for j, term in enumerate(batch_terms):
                        embeddings[term] = batch_embeddings[j]
                    
                    logger.info(f"Processed {i+len(batch)}/{len(ordered_content)} embeddings")
        
        except Exception as e:
            logger.error(f"Error generating transformer embeddings: {e}")
            logger.error("Falling back to direct term comparison")
            
            # Fallback: just use the terms themselves
            for term in ordered_terms:
                # Create a simple embedding (just to maintain the API)
                embeddings[term] = np.array([hash(term) % 100 for _ in range(10)], dtype=float)
    else:
        # Fallback: just use the terms themselves
        for term in ordered_terms:
            # Create a simple embedding (just to maintain the API)
            embeddings[term] = np.array([hash(term) % 100 for _ in range(10)], dtype=float)
    
    logger.info(f"Calculated embeddings for {len(embeddings)} terms")
    return embeddings

def calculate_similarity(term1: str, term2: str, embeddings: Dict[str, np.ndarray]) -> float:
    """Calculate cosine similarity between two terms based on their embeddings."""
    if term1 not in embeddings or term2 not in embeddings:
        return 0.0
    
    vec1 = embeddings[term1].reshape(1, -1)
    vec2 = embeddings[term2].reshape(1, -1)
    
    # Convert numpy float32 to Python float
    similarity = float(cosine_similarity(vec1, vec2)[0][0])
    
    return similarity

def calculate_co_occurrence(
    term1: str, 
    term2: str, 
    metadata: Dict, 
    resources: Dict,
    all_terms: Set[str] = None
) -> Tuple[float, float, int, Dict]:
    """
    Calculate co-occurrence statistics for two terms.
    
    Returns:
        Tuple of (conditional_prob_1_given_2, conditional_prob_2_given_1, co_occurrence_count, contingency_table)
    """
    if term1 not in metadata or term2 not in metadata:
        return 0.0, 0.0, 0, {}
    
    # Get sources for each term
    sources1 = set(metadata[term1].get('sources', []))
    sources2 = set(metadata[term2].get('sources', []))
    
    # Get resources for each term
    resource_urls1 = set()
    if term1 in resources:
        resource_urls1 = {resource.get('url', '') for resource in resources[term1]}
    
    resource_urls2 = set()
    if term2 in resources:
        resource_urls2 = {resource.get('url', '') for resource in resources[term2]}
    
    # Calculate co-occurrences
    source_co_occurrences = sources1.intersection(sources2)
    resource_co_occurrences = resource_urls1.intersection(resource_urls2)
    
    # Combine co-occurrences
    co_occurrences = len(source_co_occurrences) + len(resource_co_occurrences)
    
    # Calculate term counts
    term1_count = len(sources1) + len(resource_urls1)
    term2_count = len(sources2) + len(resource_urls2)
    
    # Get all sources and resources to determine the total document count
    total_sources = set()
    total_resources = set()
    
    # If all_terms is provided, use it to calculate totals across all terms
    if all_terms:
        for term in all_terms:
            if term in metadata:
                total_sources.update(metadata[term].get('sources', []))
            
            if term in resources:
                total_resources.update(resource.get('url', '') for resource in resources[term])
    else:
        # Otherwise, just use all values in the metadata and resources
        for term_data in metadata.values():
            total_sources.update(term_data.get('sources', []))
        
        for term_resources in resources.values():
            total_resources.update(resource.get('url', '') for resource in term_resources)
    
    # Total number of documents - real count from the corpus
    total_docs = len(total_sources) + len(total_resources)
    
    # Ensure we have a reasonable minimum
    if total_docs < 100:
        total_docs = 100
    
    # a: both terms present (co-occurrences)
    a = co_occurrences
    
    # b: term1 present, term2 absent
    b = term1_count - co_occurrences
    
    # c: term1 absent, term2 present 
    c = term2_count - co_occurrences
    
    # d: both terms absent - calculate from total real documents
    d = total_docs - a - b - c
    
    # Safety check: ensure d is not negative
    if d < 0:
        logger.warning(f"Negative value calculated for 'both absent' cell. Adjusting total document count.")
        total_docs = a + b + c + 1
        d = 1
    
    # Contingency table with real counts
    contingency_table = {
        "a": int(a),  # Both present
        "b": int(b),  # Term1 only
        "c": int(c),  # Term2 only
        "d": int(d),  # Both absent
        "term1_total": int(a + b),  # Term1 total
        "term2_total": int(a + c),  # Term2 total
        "term1_absent_total": int(c + d),  # Term1 absent total
        "term2_absent_total": int(b + d),  # Term2 absent total
        "total_docs": int(total_docs)  # Total document count (real)
    }
    
    # Calculate conditional probabilities using the real term counts
    # Avoid division by zero
    cond_prob_1_given_2 = co_occurrences / term2_count if term2_count > 0 else 0
    cond_prob_2_given_1 = co_occurrences / term1_count if term1_count > 0 else 0
    
    # Convert numpy types to Python types
    return float(cond_prob_1_given_2), float(cond_prob_2_given_1), int(co_occurrences), contingency_table

def calculate_mutual_information(
    term1: str, 
    term2: str,
    all_terms: Set[str],
    metadata: Dict, 
    resources: Dict
) -> float:
    """
    Calculate mutual information between two terms.
    
    I(X;Y) = sum_x sum_y p(x,y) * log(p(x,y)/(p(x)p(y)))
    """
    total_sources = set()
    total_resources = set()
    
    # Collect all sources and resources
    for term in all_terms:
        if term in metadata:
            total_sources.update(metadata[term].get('sources', []))
        
        if term in resources:
            total_resources.update(resource.get('url', '') for resource in resources[term])
    
    total_count = len(total_sources) + len(total_resources)
    
    # Get sources and resources for each term
    sources1 = set(metadata.get(term1, {}).get('sources', []))
    sources2 = set(metadata.get(term2, {}).get('sources', []))
    
    resource_urls1 = set()
    if term1 in resources:
        resource_urls1 = {resource.get('url', '') for resource in resources[term1]}
    
    resource_urls2 = set()
    if term2 in resources:
        resource_urls2 = {resource.get('url', '') for resource in resources[term2]}
    
    # Calculate occurrences
    count1 = len(sources1) + len(resource_urls1)
    count2 = len(sources2) + len(resource_urls2)
    
    # Calculate co-occurrences
    source_co_occurrences = sources1.intersection(sources2)
    resource_co_occurrences = resource_urls1.intersection(resource_urls2)
    co_occurrence_count = len(source_co_occurrences) + len(resource_co_occurrences)
    
    # Calculate probabilities
    p_x1 = count1 / total_count if total_count > 0 else 0
    p_x0 = 1 - p_x1
    
    p_y1 = count2 / total_count if total_count > 0 else 0
    p_y0 = 1 - p_y1
    
    p_x1_y1 = co_occurrence_count / total_count if total_count > 0 else 0
    p_x1_y0 = (count1 - co_occurrence_count) / total_count if total_count > 0 else 0
    p_x0_y1 = (count2 - co_occurrence_count) / total_count if total_count > 0 else 0
    p_x0_y0 = 1 - p_x1_y1 - p_x1_y0 - p_x0_y1
    
    # Calculate mutual information
    mi = 0
    
    # Only calculate for non-zero probabilities to avoid log(0)
    if p_x1_y1 > 0:
        mi += p_x1_y1 * np.log2(p_x1_y1 / (p_x1 * p_y1))
    if p_x1_y0 > 0:
        mi += p_x1_y0 * np.log2(p_x1_y0 / (p_x1 * p_y0))
    if p_x0_y1 > 0:
        mi += p_x0_y1 * np.log2(p_x0_y1 / (p_x0 * p_y1))
    if p_x0_y0 > 0:
        mi += p_x0_y0 * np.log2(p_x0_y0 / (p_x0 * p_y0))
    
    # Convert to regular Python float for JSON serialization
    return float(mi)

def analyze_duplicates(
    level: int, 
    base_dir: str = "data/final",
    similarity_threshold: float = 0.7,
    cond_prob_threshold: float = 0.3,
    mi_threshold: float = 0.1,
    output_dir: str = "data/analysis",
    visualize: bool = True
) -> Dict:
    """
    Analyze potential duplicates based on embedding similarity and co-occurrence statistics.
    
    Args:
        level: The hierarchy level (0, 1, 2, 3)
        base_dir: Base directory for data
        similarity_threshold: Threshold for embedding similarity
        cond_prob_threshold: Threshold for conditional probability
        mi_threshold: Threshold for mutual information
        output_dir: Directory for output files
        visualize: Whether to generate visualizations
    
    Returns:
        Dictionary of potential duplicates and their statistics
    """
    # Load data
    logger.info(f"Loading data for level {level}")
    metadata, resources = load_data(level, base_dir)
    
    # Find siblings
    logger.info("Finding siblings")
    sibling_groups = find_siblings(metadata)
    
    # Extract canonical terms
    canonical_terms = extract_canonical_terms(metadata)
    
    # Calculate embeddings
    logger.info("Calculating embeddings")
    embeddings = calculate_embeddings(canonical_terms, resources)
    
    # Analyze each sibling group
    potential_duplicates = {}
    
    logger.info("Analyzing sibling groups")
    for parent, siblings in sibling_groups.items():
        logger.info(f"Analyzing {len(siblings)} siblings of parent '{parent}'")
        
        for i, term1 in enumerate(siblings):
            for term2 in list(siblings)[i+1:]:  # Compare each pair once
                # Calculate similarity
                similarity = calculate_similarity(term1, term2, embeddings)
                
                # Calculate co-occurrence with real document counts
                cond_prob_1_given_2, cond_prob_2_given_1, co_occurrences, contingency_table = calculate_co_occurrence(
                    term1, term2, metadata, resources, canonical_terms
                )
                
                # Calculate mutual information
                mi = calculate_mutual_information(term1, term2, canonical_terms, metadata, resources)
                
                # Check for potential duplicates
                if (similarity >= similarity_threshold and 
                        (cond_prob_1_given_2 <= cond_prob_threshold or 
                         cond_prob_2_given_1 <= cond_prob_threshold) and
                        mi <= mi_threshold):
                    
                    key = f"{term1}|{term2}"
                    potential_duplicates[key] = {
                        "term1": term1,
                        "term2": term2,
                        "parent": parent,
                        "similarity": float(similarity),
                        "cond_prob_1_given_2": float(cond_prob_1_given_2),
                        "cond_prob_2_given_1": float(cond_prob_2_given_1),
                        "co_occurrences": int(co_occurrences),
                        "mutual_information": float(mi),
                        "contingency_table": contingency_table
                    }
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save potential duplicates to file
    output_file = os.path.join(output_dir, f"lv{level}_potential_duplicates.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(potential_duplicates, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Found {len(potential_duplicates)} potential duplicates")
    logger.info(f"Results saved to {output_file}")
    
    # Generate visualizations if requested
    if visualize and potential_duplicates:
        visualize_duplicates(potential_duplicates, level, output_dir)
    
    return potential_duplicates

def visualize_duplicates(duplicates: Dict, level: int, output_dir: str) -> None:
    """Generate visualizations for potential duplicates."""
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for scatter plot
    similarities = []
    cond_probs = []
    mutual_infos = []
    labels = []
    
    for key, data in duplicates.items():
        similarities.append(data["similarity"])
        # Use minimum of conditional probabilities
        cond_prob = min(data["cond_prob_1_given_2"], data["cond_prob_2_given_1"])
        cond_probs.append(cond_prob)
        mutual_infos.append(data["mutual_information"])
        labels.append(f"{data['term1']} | {data['term2']}")
    
    # Create scatter plot of similarity vs conditional probability
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(similarities, cond_probs, c=mutual_infos, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Mutual Information')
    plt.xlabel('Embedding Similarity')
    plt.ylabel('Conditional Probability (min)')
    plt.title(f'Level {level} - Potential Duplicates Analysis')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels for some points (not all to avoid clutter)
    for i, label in enumerate(labels):
        if similarities[i] > 0.8 or cond_probs[i] < 0.1:
            plt.annotate(label, (similarities[i], cond_probs[i]), 
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"lv{level}_duplicates_scatter.png"), dpi=300)
    plt.close()
    
    # Create network graph for duplicates
    G = nx.Graph()
    
    # Group duplicates by parent
    parent_groups = defaultdict(list)
    for key, data in duplicates.items():
        parent_groups[data["parent"]].append(data)
    
    # Create one subgraph per parent
    for parent, parent_duplicates in parent_groups.items():
        if len(parent_duplicates) > 1:  # Only if there are enough duplicates
            plt.figure(figsize=(12, 10))
            G = nx.Graph()
            
            # Add nodes
            terms = set()
            for data in parent_duplicates:
                terms.add(data["term1"])
                terms.add(data["term2"])
            
            for term in terms:
                G.add_node(term)
            
            # Add edges
            for data in parent_duplicates:
                G.add_edge(
                    data["term1"], 
                    data["term2"], 
                    weight=data["similarity"],
                    similarity=data["similarity"],
                    cond_prob=min(data["cond_prob_1_given_2"], data["cond_prob_2_given_1"]),
                    mi=data["mutual_information"]
                )
            
            # Draw graph
            pos = nx.spring_layout(G, seed=42)
            
            # Edge colors based on conditional probability
            edge_colors = [G[u][v]['cond_prob'] for u, v in G.edges()]
            
            # Node size based on degree
            node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
            edges = nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors, 
                                         edge_cmap=plt.cm.YlOrRd_r, edge_vmin=0, edge_vmax=0.5)
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            plt.colorbar(edges, label='Conditional Probability (lower is more likely a duplicate)')
            plt.title(f'Level {level} - Potential Duplicates - Parent: {parent}')
            plt.axis('off')
            plt.tight_layout()
            
            # Save figure
            parent_safe = parent.replace('/', '_').replace('\\', '_')
            plt.savefig(os.path.join(output_dir, f"lv{level}_duplicates_network_{parent_safe}.png"), dpi=300)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze potential duplicates in glossary terms')
    parser.add_argument('-l', '--level', type=int, required=True,
                        help='Level number (0, 1, 2, 3)')
    parser.add_argument('-d', '--data-dir', type=str, default="data/final",
                        help='Base directory for glossary data (default: data/final)')
    parser.add_argument('-o', '--output-dir', type=str, default="data/analysis",
                        help='Output directory for analysis results (default: data/analysis)')
    parser.add_argument('-s', '--similarity', type=float, default=0.7,
                        help='Similarity threshold (default: 0.7)')
    parser.add_argument('-c', '--conditional-prob', type=float, default=0.3,
                        help='Conditional probability threshold (default: 0.3)')
    parser.add_argument('-m', '--mutual-info', type=float, default=0.1,
                        help='Mutual information threshold (default: 0.1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization generation')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run analysis
    analyze_duplicates(
        args.level,
        base_dir=args.data_dir,
        similarity_threshold=args.similarity,
        cond_prob_threshold=args.conditional_prob,
        mi_threshold=args.mutual_info,
        output_dir=args.output_dir,
        visualize=not args.no_visualize
    )

if __name__ == "__main__":
    main() 