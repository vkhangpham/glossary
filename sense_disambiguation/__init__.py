"""
Sense disambiguation package for academic glossary generation.

This package provides tools to identify and resolve ambiguous terms
in the generated academic glossary.
"""

from sense_disambiguation.detector import *
from .splitter import SenseSplitter

__all__ = ["ParentContextDetector", "ResourceClusterDetector", "SenseSplitter", "GlobalResourceClusterer"] 