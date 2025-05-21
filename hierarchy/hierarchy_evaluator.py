#!/usr/bin/env python

import os
import json
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
import pandas as pd

DATA_DIR = 'data'
FINAL_DIR = os.path.join(DATA_DIR, 'final')
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')
EVALUATION_DIR = os.path.join(DATA_DIR, 'evaluation')

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

# Files for storing evaluation results
METRICS_FILE = os.path.join(EVALUATION_DIR, 'metrics.json')
ISSUES_FILE = os.path.join(EVALUATION_DIR, 'issues.json')
CONNECTIVITY_FILE = os.path.join(EVALUATION_DIR, 'connectivity.json')
SUMMARY_FILE = os.path.join(EVALUATION_DIR, 'summary.json')
VISUALIZATION_DIR = os.path.join(EVALUATION_DIR, 'visualizations')


def load_hierarchy(filepath: str = os.path.join(DATA_DIR, 'hierarchy.json')) -> Dict[str, Any]:
    """Load the hierarchy data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            hierarchy = json.load(f)
        return hierarchy
    except Exception as e:
        print(f"Error loading hierarchy data from {filepath}: {e}")
        return None


def create_hierarchy_graph(hierarchy: Dict[str, Any]) -> nx.DiGraph:
    """Create a directed graph from the hierarchy data."""
    G = nx.DiGraph()
    
    # Add nodes with level info
    for level, terms in hierarchy["levels"].items():
        level_int = int(level)
        for term in terms:
            G.add_node(term, level=level_int)
    
    # Add edges for parent-child relationships
    for parent, child, _ in hierarchy["relationships"]["parent_child"]:
        G.add_edge(parent, child)
    
    return G


def calculate_terms_per_level(hierarchy: Dict[str, Any]) -> Dict[str, int]:
    """Calculate the number of terms at each level of the hierarchy."""
    return {str(level): len(terms) for level, terms in hierarchy["levels"].items()}


def find_orphan_terms(hierarchy: Dict[str, Any]) -> Dict[str, List[str]]:
    """Find terms that have no parent in the hierarchy (excluding level 0)."""
    orphans = defaultdict(list)
    
    # For each level (excluding level 0, which doesn't need parents)
    for level in range(1, 4):  # Levels 1, 2, 3
        level_terms = hierarchy["levels"][str(level)]
        
        # Check each term in this level
        for term in level_terms:
            term_data = hierarchy["terms"][term]
            
            # If term has no parents, it's an orphan
            if not term_data["parents"]:
                orphans[str(level)].append(term)
    
    return orphans


def find_leaf_terms(hierarchy: Dict[str, Any]) -> Dict[int, List[str]]:
    """Find terms without children at each level."""
    leaves = defaultdict(list)
    
    for level_str in hierarchy["levels"]:
        level = int(level_str)
        for term in hierarchy["levels"][level_str]:
            # Check if term has any children
            if not hierarchy["terms"][term]["children"]:
                leaves[level].append(term)
    
    return dict(leaves)


def calculate_branching_factor(hierarchy: Dict[str, Any]) -> Dict[str, float]:
    """Calculate the average number of children per term at each level."""
    branching = {}
    
    # For each level (excluding the last level)
    for level in range(3):  # Levels 0, 1, 2 (level 3 has no children)
        level_terms = hierarchy["levels"][str(level)]
        
        # Skip if no terms at this level
        if not level_terms:
            branching[str(level)] = 0
            continue
        
        # Count total children for this level
        total_children = 0
        for term in level_terms:
            term_data = hierarchy["terms"][term]
            total_children += len(term_data["children"])
        
        # Calculate average
        branching[str(level)] = total_children / len(level_terms)
    
    return branching


def check_variation_consolidation(hierarchy: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Analyze variation consolidation metrics for terms at each level."""
    # Count terms with variations and average variations per term
    terms_with_variations = defaultdict(int)
    variation_counts = defaultdict(list)
    
    for term, term_data in hierarchy["terms"].items():
        level = str(term_data["level"])
        variations = term_data["variations"]
        
        if variations:
            terms_with_variations[level] += 1
            variation_counts[level].append(len(variations))
    
    # Calculate average variations per term
    avg_variations_per_term = {}
    for level, counts in variation_counts.items():
        if counts:
            avg_variations_per_term[level] = sum(counts) / len(counts)
        else:
            avg_variations_per_term[level] = 0
    
    return {
        "terms_with_variations": dict(terms_with_variations),
        "avg_variations_per_term": avg_variations_per_term
    }


def calculate_network_metrics(G: nx.DiGraph) -> Dict[str, Any]:
    """Calculate various network metrics for the hierarchy graph."""
    metrics = {
        "density": nx.density(G),
        "transitivity": nx.transitivity(G),
        "avg_shortest_path": None,
        "diameter": None,
        "degree_centrality": None,
        "centrality_by_level": defaultdict(list),
    }
    
    # Check if the graph is connected before calculating path metrics
    if nx.is_weakly_connected(G):
        try:
            metrics["avg_shortest_path"] = nx.average_shortest_path_length(G)
            metrics["diameter"] = nx.diameter(G)
        except nx.NetworkXError:
            # Handle disconnected graph
            print("Graph is not strongly connected, some path metrics unavailable")
    
    # Calculate degree centrality
    centrality = nx.degree_centrality(G)
    metrics["degree_centrality"] = centrality
    
    # Group centrality by level
    for node, cent in centrality.items():
        level = G.nodes[node].get("level", -1)
        metrics["centrality_by_level"][level].append((node, cent))
    
    # Sort centrality within each level
    for level in metrics["centrality_by_level"]:
        metrics["centrality_by_level"][level].sort(key=lambda x: x[1], reverse=True)
    
    return metrics


def analyze_level_connectivity(hierarchy: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Analyze connectivity between different levels of the hierarchy."""
    connectivity = {}
    
    # Build a graph representation
    G = nx.DiGraph()
    
    # Add all terms as nodes
    for level, terms in hierarchy["levels"].items():
        for term in terms:
            G.add_node(term, level=int(level))
    
    # Add parent-child relationships as edges
    for parent, child, _ in hierarchy["relationships"]["parent_child"]:
        G.add_edge(parent, child)
    
    # Analyze connectivity between adjacent levels
    for from_level in range(3):  # 0, 1, 2
        to_level = from_level + 1
        
        from_terms = set(hierarchy["levels"][str(from_level)])
        to_terms = set(hierarchy["levels"][str(to_level)])
        
        # Skip if either level is empty
        if not from_terms or not to_terms:
            continue
        
        # Count connections
        from_connected = set()
        to_connected = set()
        
        for parent, child, _ in hierarchy["relationships"]["parent_child"]:
            if parent in from_terms and child in to_terms:
                from_connected.add(parent)
                to_connected.add(child)
        
        # Calculate percentages
        perc_from_connected = (len(from_connected) / len(from_terms)) * 100
        perc_to_connected = (len(to_connected) / len(to_terms)) * 100
        
        key = f"{from_level}_to_{to_level}"
        connectivity[key] = {
            "from_level": from_level,
            "to_level": to_level,
            "from_terms_count": len(from_terms),
            "to_terms_count": len(to_terms),
            "from_connected_count": len(from_connected),
            "to_connected_count": len(to_connected),
            "perc_from_connected": perc_from_connected,
            "perc_to_connected": perc_to_connected
        }
    
    return connectivity


def calculate_path_distribution(G: nx.DiGraph, source_level: int = 0) -> Dict[str, Any]:
    """Calculate the distribution of path lengths from source level to all other levels."""
    results = {
        "max_path_lengths": defaultdict(int),
        "avg_path_lengths": defaultdict(float),
        "path_counts": defaultdict(list)
    }
    
    # Get all nodes at the source level
    source_nodes = [node for node, data in G.nodes(data=True) if data.get('level') == source_level]
    
    for source in source_nodes:
        # Get all reachable nodes from this source
        for target in G.nodes():
            if source == target:
                continue
                
            target_level = G.nodes[target].get('level', -1)
            if target_level <= source_level:
                continue  # Skip nodes at same or higher levels
                
            try:
                # Find all paths from source to target
                all_paths = list(nx.all_simple_paths(G, source, target))
                if all_paths:
                    path_lengths = [len(path) - 1 for path in all_paths]  # -1 because path length is #edges
                    max_length = max(path_lengths)
                    avg_length = sum(path_lengths) / len(path_lengths)
                    
                    # Update results
                    if max_length > results["max_path_lengths"][target_level]:
                        results["max_path_lengths"][target_level] = max_length
                    
                    results["path_counts"][target_level].append(len(all_paths))
                    results["avg_path_lengths"][target_level] += avg_length
            except nx.NetworkXNoPath:
                # No path exists, skip
                pass
    
    # Calculate averages
    for level in results["avg_path_lengths"]:
        count = len(results["path_counts"][level])
        if count > 0:
            results["avg_path_lengths"][level] /= count
    
    return dict(results)


def detect_hierarchy_issues(hierarchy: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Detect potential issues in the hierarchy structure."""
    issues = {
        "orphan_terms": [],
        "redundant_paths": [],
        "inconsistent_branching": [],
        "unbalanced_subtrees": []
    }
    
    # Find orphan terms (excluding level 0)
    orphans = find_orphan_terms(hierarchy)
    for level, terms in orphans.items():
        for term in terms:
            issues["orphan_terms"].append({
                "term": term,
                "level": level,
                "issue": "Term has no parent"
            })
    
    # Build a graph representation for further analysis
    G = nx.DiGraph()
    
    # Add all terms as nodes
    for level, terms in hierarchy["levels"].items():
        for term in terms:
            G.add_node(term, level=int(level))
    
    # Add parent-child relationships as edges
    for parent, child, _ in hierarchy["relationships"]["parent_child"]:
        G.add_edge(parent, child)
    
    # Check for redundant paths (when a term has a path to another term through multiple routes)
    for level in range(2, 4):  # Check levels 2 and 3
        for term in hierarchy["levels"][str(level)]:
            # Get all ancestors
            ancestors = set()
            for parent in hierarchy["terms"][term]["parents"]:
                ancestors.add(parent)
                # Add grandparents
                for grandparent in hierarchy["terms"].get(parent, {}).get("parents", []):
                    ancestors.add(grandparent)
            
            # Check if any ancestor is also a direct parent
            direct_parents = set(hierarchy["terms"][term]["parents"])
            redundant_ancestors = ancestors.intersection(direct_parents)
            
            if len(redundant_ancestors) > 0:
                for ancestor in redundant_ancestors:
                    issues["redundant_paths"].append({
                        "term": term,
                        "ancestor": ancestor,
                        "level": level,
                        "issue": "Term has both direct and indirect paths to the same ancestor"
                    })
    
    # Check for inconsistent branching (terms with very few or too many children)
    branching = calculate_branching_factor(hierarchy)
    
    for level in range(3):  # Levels 0, 1, 2
        avg_children = branching[str(level)]
        if avg_children == 0:
            continue
            
        for term in hierarchy["levels"][str(level)]:
            children_count = len(hierarchy["terms"][term]["children"])
            
            # If a term has 3x more or less than 1/3 of average children
            if children_count > 0 and (children_count > 3 * avg_children or children_count < avg_children / 3):
                issues["inconsistent_branching"].append({
                    "term": term,
                    "level": level,
                    "children_count": children_count,
                    "avg_level_children": avg_children,
                    "issue": "Term has significantly different number of children than average"
                })
    
    # Check for unbalanced subtrees (terms whose children are distributed unevenly across levels)
    for level in range(2):  # Levels 0, 1
        for term in hierarchy["levels"][str(level)]:
            children = hierarchy["terms"][term]["children"]
            children_by_level = defaultdict(int)
            
            for child in children:
                child_level = hierarchy["terms"][child]["level"]
                children_by_level[child_level] += 1
            
            # If children span across more than 2 levels, flag as potentially unbalanced
            if len(children_by_level) > 2:
                issues["unbalanced_subtrees"].append({
                    "term": term,
                    "level": level,
                    "children_by_level": dict(children_by_level),
                    "issue": "Term's children span across multiple levels, suggesting unbalanced categorization"
                })
    
    return issues


def generate_level_summary(hierarchy: Dict[str, Any]) -> pd.DataFrame:
    """Generate a summary dataframe with key statistics for each level."""
    # Calculate various metrics
    terms_per_level = calculate_terms_per_level(hierarchy)
    orphans = find_orphan_terms(hierarchy)
    branching = calculate_branching_factor(hierarchy)
    variation_stats = check_variation_consolidation(hierarchy)
    
    # Create summary dataframe
    summary = []
    
    for level in range(4):  # Levels 0, 1, 2, 3
        level_str = str(level)
        
        # Count parent-child relationships for this level as target
        parent_child_rels = 0
        for _, child, child_level in hierarchy["relationships"]["parent_child"]:
            if child_level == level:
                parent_child_rels += 1
        
        # Count variations for this level
        variations = 0
        for term in hierarchy["levels"].get(level_str, []):
            variations += len(hierarchy["terms"][term]["variations"])
        
        # Add row to summary
        summary.append({
            "Level": level,
            "Terms": terms_per_level.get(level_str, 0),
            "Orphans": len(orphans.get(level_str, [])),
            "Avg Children": round(branching.get(level_str, 0), 2),
            "Parent-Child Rels": parent_child_rels,
            "Terms with Variations": variation_stats["terms_with_variations"].get(level_str, 0),
            "Avg Variations per Term": round(variation_stats["avg_variations_per_term"].get(level_str, 0), 2),
            "Total Variations": variations
        })
    
    return pd.DataFrame(summary)


def visualize_hierarchy_metrics(hierarchy: Dict[str, Any], output_dir: str = "reports") -> Dict[str, str]:
    """Generate visualizations for hierarchy metrics and save to files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    visualizations = {}
    
    # 1. Terms per level visualization
    terms_per_level_file = os.path.join(output_dir, "terms_per_level.png")
    terms_per_level = calculate_terms_per_level(hierarchy)
    
    plt.figure(figsize=(10, 6))
    levels = [int(l) for l in terms_per_level.keys()]
    counts = list(terms_per_level.values())
    
    plt.bar(levels, counts, color='#4285F4')
    plt.title('Number of Terms per Level', fontsize=14)
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Number of Terms', fontsize=12)
    plt.xticks(levels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(terms_per_level_file, dpi=300, bbox_inches='tight')
    plt.close()
    visualizations["terms_per_level"] = terms_per_level_file
    
    # 2. Branching factor visualization
    branching_file = os.path.join(output_dir, "branching_factor.png")
    branching = calculate_branching_factor(hierarchy)
    
    plt.figure(figsize=(10, 6))
    levels = sorted([int(l) for l in branching.keys()])
    values = [branching.get(str(level), 0) for level in levels]
    
    plt.bar(levels, values, color='#34A853')
    plt.title('Average Number of Children per Term (Branching Factor)', fontsize=14)
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Average Children', fontsize=12)
    plt.xticks(levels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(branching_file, dpi=300, bbox_inches='tight')
    plt.close()
    visualizations["branching_factor"] = branching_file
    
    # 3. Connectivity heatmap
    connectivity_file = os.path.join(output_dir, "level_connectivity.png")
    connectivity = analyze_level_connectivity(hierarchy)
    
    plt.figure(figsize=(12, 6))
    
    connection_data = []
    labels = []
    
    for key, data in connectivity.items():
        from_level, to_level = key.split('_to_')
        connection_data.append([
            data['perc_from_connected'],
            data['perc_to_connected']
        ])
        labels.append(f"L{from_level}→L{to_level}")
    
    connection_data = np.array(connection_data).T
    
    plt.imshow(connection_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Percentage (%)')
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks([0, 1], ['% Source\nConnected', '% Target\nConnected'])
    plt.title('Level Connectivity', fontsize=14)
    
    # Add percentage values to cells
    for i in range(connection_data.shape[0]):
        for j in range(connection_data.shape[1]):
            plt.text(j, i, f"{connection_data[i, j]:.1f}%", 
                   ha="center", va="center", color="white", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(connectivity_file, dpi=300, bbox_inches='tight')
    plt.close()
    visualizations["connectivity"] = connectivity_file
    
    # 4. Variations visualization
    variations_file = os.path.join(output_dir, "term_variations.png")
    variation_stats = check_variation_consolidation(hierarchy)
    
    plt.figure(figsize=(12, 6))
    
    levels = sorted(set(list(variation_stats["terms_with_variations"].keys()) + 
                      list(variation_stats["avg_variations_per_term"].keys())))
    levels = [int(l) for l in levels]
    
    terms_with_var = [variation_stats["terms_with_variations"].get(str(level), 0) for level in levels]
    avg_variations = [variation_stats["avg_variations_per_term"].get(str(level), 0) for level in levels]
    
    x = np.arange(len(levels))
    width = 0.35
    
    plt.bar(x - width/2, terms_with_var, width, label='Terms with Variations', color='#FBBC05')
    plt.bar(x + width/2, avg_variations, width, label='Avg Variations per Term', color='#EA4335')
    
    plt.title('Term Variations by Level', fontsize=14)
    plt.xlabel('Level', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x, [f'Level {level}' for level in levels])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(variations_file, dpi=300, bbox_inches='tight')
    plt.close()
    visualizations["variations"] = variations_file
    
    return visualizations


def generate_html_report(hierarchy: Dict[str, Any], output_file: str = "hierarchy_evaluation.html") -> str:
    """Generate a comprehensive HTML report for hierarchy evaluation."""
    # Generate visualizations
    report_dir = os.path.dirname(output_file)
    os.makedirs(report_dir, exist_ok=True)
    
    vis_dir = os.path.join(report_dir, "visualizations")
    visualizations = visualize_hierarchy_metrics(hierarchy, vis_dir)
    
    # Calculate metrics
    terms_per_level = calculate_terms_per_level(hierarchy)
    orphans = find_orphan_terms(hierarchy)
    branching = calculate_branching_factor(hierarchy)
    variation_stats = check_variation_consolidation(hierarchy)
    connectivity = analyze_level_connectivity(hierarchy)
    issues = detect_hierarchy_issues(hierarchy)
    
    # Create summary dataframe
    summary_df = generate_level_summary(hierarchy)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Academic Hierarchy Evaluation</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .metric-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .metric-box {{ flex: 1; min-width: 200px; background-color: #f8f9fa; border-radius: 5px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; margin: 10px 0; }}
            .metric-name {{ font-size: 14px; color: #7f8c8d; }}
            .visualization {{ margin: 30px 0; text-align: center; }}
            .visualization img {{ max-width: 100%; height: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .issue {{ background-color: #fff8f8; border-left: 4px solid #e74c3c; padding: 10px; margin: 10px 0; }}
            .recommendations {{ background-color: #f0f9ff; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Academic Hierarchy Evaluation Report</h1>
        
        <div class="metric-container">
            <div class="metric-box">
                <div class="metric-name">Total Terms</div>
                <div class="metric-value">{hierarchy['stats']['total_terms']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-name">Total Relationships</div>
                <div class="metric-value">{hierarchy['stats']['total_relationships']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-name">Total Variations</div>
                <div class="metric-value">{hierarchy['stats']['total_variations']}</div>
            </div>
        </div>
        
        <h2>Level Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Level</th>
                    <th>Terms</th>
                    <th>Orphans</th>
                    <th>Avg Children</th>
                    <th>Parent-Child Rel.</th>
                    <th>Terms w/ Variations</th>
                    <th>Avg Variations/Term</th>
                    <th>Total Variations</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add rows for each level
    for _, row in summary_df.iterrows():
        html_content += f"""
                <tr>
                    <td>{int(row['Level'])}</td>
                    <td>{int(row['Terms'])}</td>
                    <td>{int(row['Orphans'])}</td>
                    <td>{row['Avg Children']:.2f}</td>
                    <td>{int(row['Parent-Child Rels'])}</td>
                    <td>{int(row['Terms with Variations'])}</td>
                    <td>{row['Avg Variations per Term']:.2f}</td>
                    <td>{int(row['Total Variations'])}</td>
                </tr>
        """
    
    html_content += """
            </tbody>
        </table>
        
        <h2>Visualizations</h2>
        
        <div class="visualization">
            <h3>Terms per Level</h3>
            <img src="visualizations/terms_per_level.png" alt="Terms per Level">
        </div>
        
        <div class="visualization">
            <h3>Branching Factor</h3>
            <img src="visualizations/branching_factor.png" alt="Branching Factor">
        </div>
        
        <div class="visualization">
            <h3>Level Connectivity</h3>
            <img src="visualizations/level_connectivity.png" alt="Level Connectivity">
        </div>
        
        <div class="visualization">
            <h3>Term Variations</h3>
            <img src="visualizations/term_variations.png" alt="Term Variations">
        </div>
        
        <h2>Issues Detected</h2>
    """
    
    # Add issues by category
    for issue_type, issue_list in issues.items():
        if not issue_list:
            continue
            
        html_content += f"""
        <h3>{issue_type.replace('_', ' ').title()} ({len(issue_list)})</h3>
        """
        
        # Show max 10 issues per category
        for issue in issue_list[:10]:
            html_content += f"""
            <div class="issue">
                <strong>{issue['term']}</strong> (Level {issue['level']}): {issue['issue']}
            </div>
            """
            
        if len(issue_list) > 10:
            html_content += f"""
            <p><em>{len(issue_list) - 10} more issues of this type not shown...</em></p>
            """
    
    # Add recommendations
    html_content += """
        <h2>Recommendations</h2>
        <div class="recommendations">
    """
    
    # Generate recommendations based on analysis
    recommendations = []
    
    # Check for orphans
    total_orphans = sum(len(terms) for terms in orphans.values())
    if total_orphans > 0:
        recommendations.append(
            f"Address the {total_orphans} orphaned terms by assigning appropriate parent terms."
        )
    
    # Check for very low connectivity
    low_connectivity = False
    for key, data in connectivity.items():
        if data["perc_to_connected"] < 50:
            low_connectivity = True
            from_level, to_level = key.split('_to_')
            recommendations.append(
                f"Improve connections between Level {from_level} and Level {to_level} - only {data['perc_to_connected']:.1f}% of Level {to_level} terms have parents."
            )
    
    # Check for variations
    if hierarchy['stats']['total_variations'] > 0:
        recommendations.append(
            f"Review the {hierarchy['stats']['total_variations']} term variations to ensure they're correctly consolidated."
        )
    
    # Check for branching inconsistency
    for level, avg in branching.items():
        if avg > 10:
            recommendations.append(
                f"Level {level} terms have a high branching factor ({avg:.2f} children on average). Consider reorganizing to create a more balanced hierarchy."
            )
    
    # Add recommendations to HTML
    if recommendations:
        for rec in recommendations:
            html_content += f"<p>• {rec}</p>\n"
    else:
        html_content += "<p>No specific recommendations - the hierarchy structure appears to be well-balanced.</p>\n"
    
    html_content += """
        </div>
        
        <h2>Methodology</h2>
        <p>This evaluation analyzes the academic hierarchy structure based on the following metrics:</p>
        <ul>
            <li><strong>Term Distribution:</strong> Analysis of term count and distribution across levels</li>
            <li><strong>Connectivity:</strong> Measurement of how well terms connect between hierarchy levels</li>
            <li><strong>Branching Factor:</strong> Analysis of the average number of children per term</li>
            <li><strong>Orphaned Terms:</strong> Identification of terms without proper parent relationships</li>
            <li><strong>Redundant Paths:</strong> Detection of terms with multiple paths to the same ancestor</li>
            <li><strong>Term Variations:</strong> Analysis of term variation patterns and consolidation</li>
        </ul>
        
        <footer>
            <p><em>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </footer>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file


def save_evaluation_results(hierarchy: Dict[str, Any], verbose: bool = False) -> None:
    """Calculate and save all evaluation results to files for the visualizer."""
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    if verbose:
        print("Calculating and saving evaluation metrics...")
        
    # Calculate metrics
    terms_per_level = calculate_terms_per_level(hierarchy)
    orphans = find_orphan_terms(hierarchy)
    branching = calculate_branching_factor(hierarchy)
    variation_stats = check_variation_consolidation(hierarchy)
    
    # Format metrics
    metrics = {
        "summary": {
            "total_terms": hierarchy['stats']['total_terms'],
            "total_relationships": hierarchy['stats']['total_relationships'],
            "total_variations": hierarchy['stats']['total_variations']
        },
        "terms_per_level": {k: v for k, v in terms_per_level.items()},
        "orphan_terms": {k: len(v) for k, v in orphans.items()},
        "branching_factor": {k: round(v, 2) for k, v in branching.items()},
        "variation_stats": {
            "terms_with_variations": {k: v for k, v in variation_stats["terms_with_variations"].items()},
            "avg_variations_per_term": {k: round(v, 2) for k, v in variation_stats["avg_variations_per_term"].items()}
        }
    }
    
    # Save metrics
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    if verbose:
        print(f"Metrics saved to {METRICS_FILE}")
    
    # Calculate and save issues
    issues = detect_hierarchy_issues(hierarchy)
    formatted_issues = {}
    for issue_type, issue_list in issues.items():
        formatted_issues[issue_type] = issue_list[:100]  # Limit to 100 issues per type
    
    with open(ISSUES_FILE, 'w', encoding='utf-8') as f:
        json.dump(formatted_issues, f, indent=2)
    
    if verbose:
        print(f"Issues saved to {ISSUES_FILE}")
    
    # Calculate and save connectivity
    connectivity = analyze_level_connectivity(hierarchy)
    with open(CONNECTIVITY_FILE, 'w', encoding='utf-8') as f:
        json.dump(connectivity, f, indent=2)
    
    if verbose:
        print(f"Connectivity data saved to {CONNECTIVITY_FILE}")
    
    # Generate and save summary
    summary_df = generate_level_summary(hierarchy)
    summary = summary_df.to_dict(orient='records')
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"Summary saved to {SUMMARY_FILE}")
    
    # Generate and save visualizations as base64 images
    if verbose:
        print("Generating visualizations...")
    
    # Terms per level visualization
    fig = create_terms_per_level_viz(hierarchy)
    img_data = figure_to_base64(fig)
    with open(os.path.join(VISUALIZATION_DIR, 'terms_per_level.json'), 'w', encoding='utf-8') as f:
        json.dump({"image": img_data}, f)
    
    # Connectivity visualization
    fig = create_connectivity_viz(hierarchy)
    img_data = figure_to_base64(fig)
    with open(os.path.join(VISUALIZATION_DIR, 'connectivity.json'), 'w', encoding='utf-8') as f:
        json.dump({"image": img_data}, f)
    
    # Branching visualization
    fig = create_branching_viz(hierarchy)
    img_data = figure_to_base64(fig)
    with open(os.path.join(VISUALIZATION_DIR, 'branching.json'), 'w', encoding='utf-8') as f:
        json.dump({"image": img_data}, f)
    
    # Variations visualization
    fig = create_variations_viz(hierarchy)
    img_data = figure_to_base64(fig)
    with open(os.path.join(VISUALIZATION_DIR, 'variations.json'), 'w', encoding='utf-8') as f:
        json.dump({"image": img_data}, f)
    
    if verbose:
        print(f"Visualizations saved to {VISUALIZATION_DIR}/")
        print("All evaluation results saved successfully")


def figure_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded string."""
    import io
    import base64
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_data


def create_terms_per_level_viz(hierarchy):
    """Create a bar chart of terms per level."""
    terms_per_level = calculate_terms_per_level(hierarchy)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = [int(l) for l in terms_per_level.keys()]
    counts = list(terms_per_level.values())
    
    bars = ax.bar(levels, counts, color='#4285F4')
    ax.set_title('Number of Terms per Level', fontsize=14)
    ax.set_xlabel('Level', fontsize=12)
    ax.set_ylabel('Number of Terms', fontsize=12)
    ax.set_xticks(levels)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels to bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom')
    
    return fig


def create_connectivity_viz(hierarchy):
    """Create a heatmap of level connectivity."""
    connectivity = analyze_level_connectivity(hierarchy)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    connection_data = []
    labels = []
    
    for key, data in connectivity.items():
        from_level, to_level = key.split('_to_')
        connection_data.append([
            data['perc_from_connected'],
            data['perc_to_connected']
        ])
        labels.append(f"L{from_level}→L{to_level}")
    
    connection_data = np.array(connection_data).T
    
    im = ax.imshow(connection_data, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Percentage (%)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['% Source\nConnected', '% Target\nConnected'])
    ax.set_title('Level Connectivity', fontsize=14)
    
    # Add percentage values to cells
    for i in range(connection_data.shape[0]):
        for j in range(connection_data.shape[1]):
            ax.text(j, i, f"{connection_data[i, j]:.1f}%", 
                   ha="center", va="center", color="white", fontweight="bold")
    
    plt.tight_layout()
    return fig


def create_branching_viz(hierarchy):
    """Create a visualization of branching factors."""
    branching = calculate_branching_factor(hierarchy)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = sorted([int(l) for l in branching.keys()])
    values = [branching.get(str(level), 0) for level in levels]
    
    bars = ax.bar(levels, values, color='#34A853')
    ax.set_title('Average Number of Children per Term (Branching Factor)', fontsize=14)
    ax.set_xlabel('Level', fontsize=12)
    ax.set_ylabel('Average Children', fontsize=12)
    ax.set_xticks(levels)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels to bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    return fig


def create_variations_viz(hierarchy):
    """Create a visualization of term variations."""
    variation_stats = check_variation_consolidation(hierarchy)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    levels = sorted(set(list(variation_stats["terms_with_variations"].keys()) + 
                      list(variation_stats["avg_variations_per_term"].keys())))
    
    terms_with_var = [variation_stats["terms_with_variations"].get(level, 0) for level in levels]
    avg_variations = [variation_stats["avg_variations_per_term"].get(level, 0) for level in levels]
    
    x = np.arange(len(levels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, terms_with_var, width, label='Terms with Variations', color='#FBBC05')
    bars2 = ax.bar(x + width/2, avg_variations, width, label='Avg Variations per Term', color='#EA4335')
    
    ax.set_title('Term Variations by Level', fontsize=14)
    ax.set_xlabel('Level', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Level {level}' for level in levels])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
    
    return fig


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate the quality of the academic hierarchy')
    parser.add_argument('-i', '--input', type=str, default='data/hierarchy.json',
                        help='Input hierarchy file (default: data/hierarchy.json)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output report file (default: data/reports/hierarchy_quality_report.html)')
    parser.add_argument('-s', '--save-all', action='store_true',
                        help='Save all evaluation metrics for offline visualization')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set default output if not specified
    if args.output is None:
        args.output = os.path.join(REPORTS_DIR, 'hierarchy_quality_report.html')
    
    if args.verbose:
        print(f"Loading hierarchy from {args.input}")
    
    # Load hierarchy
    hierarchy = load_hierarchy(args.input)
    
    if not hierarchy:
        print("Error: Failed to load hierarchy data")
        return
    
    # Generate HTML report
    if args.verbose:
        print("Calculating metrics and generating report...")
    
    report_file = generate_html_report(hierarchy, args.output)
    print(f"\nHierarchy quality report generated: {report_file}")
    
    # Save all evaluation results if requested
    if args.save_all:
        save_evaluation_results(hierarchy, args.verbose)
    
    # Print summary
    print("\nKey metrics summary:")
    summary_df = generate_level_summary(hierarchy)
    print(summary_df)


if __name__ == '__main__':
    main() 