"""
Level 2 Generation Scripts for Academic Glossary System

This module contains the 4-step generation pipeline for Level 2 (Research Areas):
- Step 0: Extract research areas from department websites
- Step 1: Extract academic concepts from research areas using LLM
- Step 2: Filter concepts by institutional frequency
- Step 3: Verify single-token academic terms using LLM

Each script can be run independently with test mode support.
"""