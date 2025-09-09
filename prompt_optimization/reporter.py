"""
Functional optimization reporting utilities.

This module provides functions to generate comprehensive optimization reports
from GEPA runs, including TXT summaries and detailed JSON data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Note: evaluate_initial_performance removed - each optimizer should implement their own
# See template.py for implementation guidance


def analyze_detailed_results(detailed_results: Any) -> Dict[str, Any]:
    """
    Extract insights from GEPA detailed_results attribute.

    Args:
        detailed_results: GEPA detailed_results object with track_stats=True

    Returns:
        Dictionary containing analysis and insights
    """
    analysis = {
        "best_validation_scores": [],
        "top_improvements": [],
        "challenging_tasks": [],
        "insights": [],
    }

    if not detailed_results:
        analysis["insights"].append(
            "No detailed results available - ensure track_stats=True"
        )
        return analysis


    if hasattr(detailed_results, "highest_score_achieved_per_val_task"):
        highest_scores = detailed_results.highest_score_achieved_per_val_task
        analysis["best_validation_scores"] = highest_scores

        # Identify top improvements (assuming we have before/after comparison)
        # This would need initial scores to calculate actual improvements
        if highest_scores:
            sorted_scores = sorted(
                enumerate(highest_scores), key=lambda x: x[1], reverse=True
            )
            analysis["top_improvements"] = [
                f"Task {idx}: {score:.3f}" for idx, score in sorted_scores[:3]
            ]

            analysis["challenging_tasks"] = [
                f"Task {idx}: {score:.3f}" for idx, score in sorted_scores[-2:]
            ]


    if hasattr(detailed_results, "best_outputs_valset"):
        best_outputs = detailed_results.best_outputs_valset
        if best_outputs:
            analysis["insights"].append(
                f"Found {len(best_outputs)} best outputs during optimization"
            )


    if analysis["best_validation_scores"]:
        scores = analysis["best_validation_scores"]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)

        analysis["insights"].extend(
            [
                f"Average validation score: {avg_score:.3f}",
                f"Best validation score: {max_score:.3f}",
                f"Worst validation score: {min_score:.3f}",
                f"Score range: {max_score - min_score:.3f}",
            ]
        )

        if max_score - min_score > 0.3:
            analysis["insights"].append(
                "High score variance suggests some tasks are much harder"
            )
        if min_score < 0.5:
            analysis["insights"].append(
                "Some tasks still underperforming - may need more training data"
            )

    return analysis


def calculate_improvement(
    initial_scores: Dict[str, Any], optimized_scores: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate improvement metrics between initial and optimized performance.

    Args:
        initial_scores: Performance before optimization
        optimized_scores: Performance after optimization

    Returns:
        Dictionary containing improvement metrics
    """
    initial_avg = initial_scores.get("avg_score", 0.0)
    optimized_avg = optimized_scores.get("avg_score", 0.0)

    if initial_avg == 0:
        improvement_pct = 0.0
    else:
        improvement_pct = ((optimized_avg - initial_avg) / initial_avg) * 100

    return {
        "initial_avg": initial_avg,
        "optimized_avg": optimized_avg,
        "absolute_improvement": optimized_avg - initial_avg,
        "percentage_improvement": improvement_pct,
    }


def extract_optimized_scores(
    optimized_program: Any, detailed_results: Any
) -> Dict[str, Any]:
    """
    Extract performance scores from optimized program.

    Args:
        optimized_program: The optimized DSPy program
        detailed_results: GEPA detailed results

    Returns:
        Dictionary containing optimized performance metrics
    """
    scores = {"avg_score": 0.0, "individual_scores": [], "source": "optimized_program"}

    if detailed_results and hasattr(
        detailed_results, "highest_score_achieved_per_val_task"
    ):
        validation_scores = detailed_results.highest_score_achieved_per_val_task
        scores["individual_scores"] = validation_scores
        scores["avg_score"] = (
            sum(validation_scores) / len(validation_scores)
            if validation_scores
            else 0.0
        )

    return scores


def format_training_data_section(training_data: Dict[str, Any]) -> str:
    """Format training data information for TXT report."""
    source = training_data.get("source", "Unknown")
    examples = training_data.get("examples", 0)

    return f"""Source: {source}
Examples: {examples} total
Data loaded successfully"""


def format_performance_comparison(performance: Dict[str, Any]) -> str:
    """Format performance comparison table for TXT report."""
    initial = performance.get("initial", {})
    optimized = performance.get("optimized", {})
    improvement = performance.get("improvement", {})

    initial_avg = initial.get("avg_score", None)
    optimized_avg = optimized.get("avg_score", 0.0)


    if initial_avg is None:
        message = improvement.get("message", "No baseline available")
        return f"""                BEFORE    AFTER    IMPROVEMENT
Average Score:  N/A       {optimized_avg:.3f}    {message}"""

    pct_improvement = improvement.get("percentage_improvement", 0.0)
    direction = (
        "up" if pct_improvement > 0 else "down" if pct_improvement < 0 else "same"
    )
    arrow = "^" if direction == "up" else "v" if direction == "down" else "-"

    return f"""                BEFORE    AFTER    IMPROVEMENT
Average Score:  {initial_avg:.3f}     {optimized_avg:.3f}    {pct_improvement:+.1f}% {arrow}"""


def format_detailed_analysis(analysis: Dict[str, Any]) -> str:
    """Format detailed analysis section for TXT report."""
    best_scores = analysis.get("best_validation_scores", [])
    top_improvements = analysis.get("top_improvements", [])
    challenging_tasks = analysis.get("challenging_tasks", [])

    lines = []

    if best_scores:
        scores_str = ", ".join(f"{s:.3f}" for s in best_scores[:10])  # Show first 10
        if len(best_scores) > 10:
            scores_str += "..."
        lines.append(f"Best validation scores: [{scores_str}]")

    if top_improvements:
        lines.append("\nValidation tasks with highest scores:")
        for improvement in top_improvements:
            lines.append(f"  - {improvement}")

    if challenging_tasks:
        lines.append("\nMost challenging validation tasks:")
        for task in challenging_tasks:
            lines.append(f"  - {task}")

    return "\n".join(lines) if lines else "No detailed analysis available"


def format_insights(insights: List[str]) -> str:
    """Format insights section for TXT report."""
    if not insights:
        return "No insights generated"

    formatted_insights = []
    for insight in insights:
        formatted_insights.append(f"* {insight}")

    return "\n".join(formatted_insights)


def format_prompts_section(prompts: Dict[str, Any]) -> str:
    """Format prompts section for TXT report showing before/after comparison."""
    if not prompts:
        return "No prompt information available"
    
    initial_system = prompts.get("initial_system", "")
    initial_user = prompts.get("initial_user", "")
    optimized_system = prompts.get("optimized_system", "")
    optimized_user = prompts.get("optimized_user", "")
    
    lines = []
    
    if initial_system and optimized_system:
        lines.extend([
            "SYSTEM PROMPT COMPARISON:",
            "",
            "INITIAL SYSTEM PROMPT:",
            "─" * 50,
            initial_system,
            "",
            "OPTIMIZED SYSTEM PROMPT:",
            "─" * 50, 
            optimized_system,
            "",
        ])
    
    if initial_user and optimized_user:
        lines.extend([
            "USER PROMPT COMPARISON:",
            "",
            "INITIAL USER PROMPT:",
            "─" * 50,
            initial_user,
            "",
            "OPTIMIZED USER PROMPT:",
            "─" * 50,
            optimized_user,
            "",
        ])
    
    return "\n".join(lines) if lines else "No prompt comparison available"


def format_files_section(files: List[str]) -> str:
    """Format generated files section for TXT report."""
    if not files:
        return "No files generated"

    formatted_files = []
    for file_path in files:
        formatted_files.append(f"* {file_path}")

    return "\n".join(formatted_files)


def generate_txt_report(optimization_data: Dict[str, Any]) -> str:
    """
    Generate human-readable TXT report content.

    Args:
        optimization_data: Complete optimization data dictionary

    Returns:
        Formatted TXT report string
    """
    program_name = optimization_data.get("program_name", "Unknown")
    timestamp = optimization_data.get("timestamp", "Unknown")
    duration = optimization_data.get("duration", "Unknown")

    training_section = format_training_data_section(
        optimization_data.get("training_data", {})
    )

    performance_section = format_performance_comparison(
        optimization_data.get("performance", {})
    )

    analysis_section = format_detailed_analysis(optimization_data.get("analysis", {}))

    insights_section = format_insights(
        optimization_data.get("analysis", {}).get("insights", [])
    )

    prompts_section = format_prompts_section(optimization_data.get("prompts", {}))

    files_section = format_files_section(optimization_data.get("files", []))

    optimizer_config = optimization_data.get("optimizer_config", {})
    config_lines = []
    for key, value in optimizer_config.items():
        config_lines.append(f"{key}: {value}")
    config_section = (
        "\n".join(config_lines)
        if config_lines
        else "No configuration details available"
    )

    return f"""===============================================================================
OPTIMIZATION REPORT: {program_name}
===============================================================================
Date: {timestamp}    Duration: {duration}    Optimizer: GEPA

TRAINING DATA
━━━━━━━━━━━━━
{training_section}

PERFORMANCE IMPROVEMENT
━━━━━━━━━━━━━━━━━━━━━━━
{performance_section}

PROMPT OPTIMIZATION
━━━━━━━━━━━━━━━━━━━
{prompts_section}

DETAILED ANALYSIS (from GEPA detailed_results)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{analysis_section}

KEY INSIGHTS
━━━━━━━━━━━━
{insights_section}

OPTIMIZATION SETTINGS
━━━━━━━━━━━━━━━━━━━━━
{config_section}

FILES GENERATED
━━━━━━━━━━━━━━━
{files_section}

==============================================================================="""


def generate_json_report(optimization_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate detailed JSON report data.

    Args:
        optimization_data: Complete optimization data dictionary

    Returns:
        Dictionary suitable for JSON serialization
    """
    return optimization_data


def save_reports(
    txt_content: str,
    json_data: Dict[str, Any],
    report_name: str,
    output_dir: str = "data/optimization_reports",
) -> Dict[str, str]:
    """
    Save both TXT and JSON reports to files.
    
    Saves both timestamped versions and "latest" versions that get overwritten.

    Args:
        txt_content: Formatted TXT report content
        json_data: JSON report data
        report_name: Base name for report files
        output_dir: Directory to save reports

    Returns:
        Dictionary with paths to saved files

    Raises:
        IOError: If files cannot be written
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {output_path}")
    except Exception as e:
        raise IOError(f"Failed to create directory {output_path}: {e}")

    # Timestamped file paths
    txt_path = output_path / f"{report_name}_{timestamp}.txt"
    json_path = output_path / f"{report_name}_{timestamp}_detailed.json"
    
    # Latest file paths (will be overwritten)
    txt_latest_path = output_path / f"{report_name}_latest.txt"
    json_latest_path = output_path / f"{report_name}_latest_detailed.json"

    # Save timestamped TXT report
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        logger.info(f"Saved TXT report to: {txt_path}")
    except Exception as e:
        raise IOError(f"Failed to save TXT report to {txt_path}: {e}")

    # Save latest TXT report (overwrite)
    try:
        with open(txt_latest_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
        logger.info(f"Saved latest TXT report to: {txt_latest_path}")
    except Exception as e:
        logger.warning(f"Failed to save latest TXT report: {e}")

    # Save timestamped JSON report
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON report to: {json_path}")
    except Exception as e:
        raise IOError(f"Failed to save JSON report to {json_path}: {e}")

    # Save latest JSON report (overwrite)
    try:
        with open(json_latest_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved latest JSON report to: {json_latest_path}")
    except Exception as e:
        logger.warning(f"Failed to save latest JSON report: {e}")

    return {
        "txt_path": str(txt_path.resolve()),
        "json_path": str(json_path.resolve()),
        "txt_latest_path": str(txt_latest_path.resolve()),
        "json_latest_path": str(json_latest_path.resolve()),
    }


def create_optimization_report(
    initial_scores: Optional[Dict[str, Any]] = None,
    optimized_program: Any = None,
    detailed_results: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    prompts: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Main orchestration function - compose all the pieces to create complete report.

    Args:
        initial_scores: Performance scores before optimization (optional - can be None)
        optimized_program: The optimized DSPy program
        detailed_results: GEPA detailed_results with track_stats=True
        metadata: Additional metadata (program name, config, etc.)
        prompts: Dictionary containing initial and optimized prompts

    Returns:
        Dictionary with paths to saved TXT and JSON report files
    """

    optimized_scores = extract_optimized_scores(optimized_program, detailed_results)


    if initial_scores:
        improvement = calculate_improvement(initial_scores, optimized_scores)
    else:
        improvement = {
            "initial_avg": None,
            "optimized_avg": optimized_scores.get("avg_score", None),
            "improvement_pct": None,
            "message": "No baseline available - initial evaluation was skipped",
        }


    analysis = analyze_detailed_results(detailed_results)


    metadata = metadata or {}


    optimization_data = {
        "program_name": metadata.get("program_name", "unknown_optimization"),
        "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
        "duration": metadata.get("duration", "Unknown"),
        "training_data": metadata.get("training_data", {}),
        "performance": {
            "initial": initial_scores,
            "optimized": optimized_scores,
            "improvement": improvement,
        },
        "prompts": prompts or {},
        "analysis": analysis,
        "optimizer_config": metadata.get("optimizer_config", {}),
        "files": metadata.get("generated_files", []),
    }


    txt_content = generate_txt_report(optimization_data)
    json_data = generate_json_report(optimization_data)


    return save_reports(
        txt_content, json_data, metadata.get("program_name", "unknown_optimization")
    )


__all__ = [
    "analyze_detailed_results",
    "create_optimization_report",
    "generate_txt_report",
    "generate_json_report",
    "save_reports",
]
