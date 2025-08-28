#!/usr/bin/env python3
"""
Fix Parent-Child Relationships After Sense Disambiguation

This script addresses the problem that occurs when parent terms are split into multiple senses
but their child terms still reference the original unsplit parent term, breaking the hierarchy.

The solution:
1. Identify all child terms that reference split parent terms
2. Determine which split sense each child should belong to using semantic similarity
3. Update parent references in child terms
4. Update children lists in split parent terms
5. Apply fixes to all relevant files
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParentChildRelationshipFixer:
    def __init__(self, 
                 hierarchy_file: str = "data/hierarchy_with_splits.json",
                 split_summary_file: str = "data/split_summary.json",
                 embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the relationship fixer.
        
        Args:
            hierarchy_file: Path to the hierarchy with splits
            split_summary_file: Path to the split summary file
            embedding_model_name: Name of the sentence transformer model to use
        """
        self.hierarchy_file = hierarchy_file
        self.split_summary_file = split_summary_file
        self.embedding_model_name = embedding_model_name
        
        # Data storage
        self.hierarchy = None
        self.split_summary = None
        self.split_terms_mapping = {}  # original_term -> [split_term1, split_term2, ...]
        self.affected_children = {}    # child_term -> original_parent_term
        self.affected_related = {}     # term -> original_related_concept
        
        # Embedding model (lazy loaded)
        self._embedding_model = None
        
    @property
    def embedding_model(self):
        """Lazy load the embedding model"""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def load_data(self):
        """Load hierarchy and split summary data"""
        logger.info("Loading hierarchy data...")
        with open(self.hierarchy_file, 'r') as f:
            self.hierarchy = json.load(f)
            
        logger.info("Loading split summary...")
        with open(self.split_summary_file, 'r') as f:
            self.split_summary = json.load(f)
            
        logger.info(f"Loaded hierarchy with {len(self.hierarchy['terms'])} terms")
        logger.info(f"Loaded {self.split_summary['total_terms_split']} split records")
        
    def build_split_mapping(self):
        """Build mapping from original terms to their split versions"""
        logger.info("Building split term mapping...")
        
        for split_record in self.split_summary['detailed_splits']:
            original_term = split_record['original_term']
            new_terms = [term_info['term'] for term_info in split_record['new_terms']]
            self.split_terms_mapping[original_term] = new_terms
            
        logger.info(f"Built mapping for {len(self.split_terms_mapping)} split terms")
        
    def identify_affected_children(self):
        """Identify child terms that reference split parent terms"""
        logger.info("Identifying affected child terms...")
        
        for term, term_data in self.hierarchy['terms'].items():
            parents = term_data.get('parents', [])
            
            for parent in parents:
                if parent in self.split_terms_mapping:
                    # This child references a split parent
                    if term not in self.affected_children or self.affected_children[term] != parent: # Store only once per unique child-parent pair
                        self.affected_children[term] = parent
                        logger.debug(f"Found affected child: '{term}' -> split parent '{parent}'")
                    
        logger.info(f"Found {len(self.affected_children)} children affected by parent splits")
        
        # Log statistics
        parent_children_count = defaultdict(list)
        for child, parent in self.affected_children.items():
            parent_children_count[parent].append(child)
            
        logger.info("Affected children by split parent:")
        for parent, children in parent_children_count.items():
            logger.info(f"  '{parent}' -> {len(children)} children: {children[:5]}{'...' if len(children) > 5 else ''}")
    
    def identify_affected_related_concepts(self):
        """Identify terms whose related_concepts reference split terms"""
        logger.info("Identifying affected related concepts...")
        
        for term, term_data in self.hierarchy['terms'].items():
            related_concepts = term_data.get('related_concepts', [])
            
            for related_concept in related_concepts:
                if related_concept in self.split_terms_mapping:
                    # This term references a split term in its related_concepts
                    if term not in self.affected_related:
                        self.affected_related[term] = []
                    if related_concept not in self.affected_related[term]: # Avoid duplicates
                        self.affected_related[term].append(related_concept)
                        logger.debug(f"Found term '{term}' with affected related_concept '{related_concept}'")
                        
        logger.info(f"Found {len(self.affected_related)} terms with related_concepts affected by splits")

        # Log statistics
        split_term_references_count = defaultdict(list)
        for term, original_related_list in self.affected_related.items():
            for original_related in original_related_list:
                 split_term_references_count[original_related].append(term)
        
        logger.info("Affected related concepts by split term:")
        for original_related, terms in split_term_references_count.items():
            logger.info(f"  Split term '{original_related}' is related to {len(terms)} terms: {terms[:5]}{'...' if len(terms) > 5 else ''}")
    
    def determine_best_split_parent(self, child_term: str, original_parent: str) -> str:
        """
        Determine which split version of the parent the child should belong to.
        
        Args:
            child_term: The child term that needs reassignment
            original_parent: The original parent term that was split
            
        Returns:
            The best split parent term for the child
        """
        if original_parent not in self.split_terms_mapping:
            logger.error(f"Original parent '{original_parent}' not found in split_terms_mapping. Cannot determine best split parent for '{child_term}'. Returning original parent.")
            return original_parent # Should not happen if called correctly

        split_parents = self.split_terms_mapping[original_parent]
        
        if not child_term in self.hierarchy['terms']:
            logger.warning(f"Child term '{child_term}' not found in main hierarchy. Cannot use its context to determine best split for parent '{original_parent}'. Assigning to first split parent: '{split_parents[0]}'")
            return split_parents[0]
            
        child_data = self.hierarchy['terms'][child_term]
        
        # Method 1: Check if child has resources and use semantic similarity
        child_resources = child_data.get('resources', [])
        if child_resources:
            return self._determine_by_resource_similarity(child_term, split_parents)
            
        # Method 2: Use term context (definition, sources) for semantic similarity
        child_context = self._get_term_context(child_term)
        if child_context:
            return self._determine_by_context_similarity(child_context, split_parents)
            
        # Method 3: Analyze child's own children for additional context
        child_children = child_data.get('children', [])
        if child_children:
            return self._determine_by_children_context(child_children, split_parents)
            
        # Fallback: Use the first split parent (with a warning)
        logger.warning(f"No semantic information available for '{child_term}'. Assigning to first split parent '{split_parents[0]}'")
        return split_parents[0]
    
    def _get_term_context(self, term: str) -> str:
        """Get contextual text for a term from various sources"""
        if term not in self.hierarchy['terms']:
            logger.warning(f"Term '{term}' not found in hierarchy while trying to get context. This might be an expected split term that failed to be created.")
            return "" # Return empty context if term doesn't exist
            
        term_data = self.hierarchy['terms'][term]
        context_parts = []
        
        # Add definition if available
        definition = term_data.get('definition', '')
        if definition:
            context_parts.append(definition)
            
        # Add sources
        sources = term_data.get('sources', [])
        if sources:
            # Take first few sources to avoid too much text
            context_parts.extend(sources[:3])
            
        # Add resource content if available
        resources = term_data.get('resources', [])
        for resource in resources[:2]:  # Limit to first 2 resources
            content = resource.get('processed_content', '')
            if content:
                # Take first 200 characters to avoid overwhelming the model
                context_parts.append(content[:200])
                
        return ' '.join(context_parts)
    
    def _determine_by_resource_similarity(self, child_term: str, split_parents: List[str]) -> str:
        """Determine best parent using resource content similarity"""
        child_data = self.hierarchy['terms'][child_term]
        child_resources = child_data.get('resources', [])
        
        # Collect child resource content
        child_texts = []
        for resource in child_resources[:5]:  # Limit to avoid too much computation
            content = resource.get('processed_content', '')
            if content:
                child_texts.append(content[:300])  # Truncate long content
                
        if not child_texts:
            return split_parents[0]  # Fallback
            
        # Collect split parent contexts
        parent_contexts = {}
        for parent in split_parents:
            parent_context = self._get_term_context(parent)
            if parent_context:
                parent_contexts[parent] = parent_context
                
        if not parent_contexts:
            return split_parents[0]  # Fallback
            
        # Calculate similarities using embeddings
        try:
            child_text = ' '.join(child_texts)
            parent_texts = list(parent_contexts.values())
            parent_names = list(parent_contexts.keys())
            
            # Get embeddings
            all_texts = [child_text] + parent_texts
            embeddings = self.embedding_model.encode(all_texts)
            
            child_embedding = embeddings[0]
            parent_embeddings = embeddings[1:]
            
            # Calculate cosine similarities
            similarities = []
            for parent_emb in parent_embeddings:
                sim = np.dot(child_embedding, parent_emb) / (
                    np.linalg.norm(child_embedding) * np.linalg.norm(parent_emb)
                )
                similarities.append(sim)
                
            # Return parent with highest similarity
            best_idx = np.argmax(similarities)
            best_parent = parent_names[best_idx]
            best_similarity = similarities[best_idx]
            
            logger.info(f"Assigned '{child_term}' to '{best_parent}' (similarity: {best_similarity:.3f})")
            return best_parent
            
        except Exception as e:
            logger.error(f"Error computing similarities for '{child_term}': {e}")
            return split_parents[0]  # Fallback
    
    def _determine_by_context_similarity(self, child_context: str, split_parents: List[str]) -> str:
        """Determine best parent using general context similarity"""
        if not child_context.strip():
            return split_parents[0]
            
        # Get parent contexts
        parent_contexts = {}
        for parent in split_parents:
            parent_context = self._get_term_context(parent)
            if parent_context:
                parent_contexts[parent] = parent_context
                
        if not parent_contexts:
            return split_parents[0]
            
        try:
            # Get embeddings
            all_texts = [child_context] + list(parent_contexts.values())
            embeddings = self.embedding_model.encode(all_texts)
            
            child_embedding = embeddings[0]
            parent_embeddings = embeddings[1:]
            parent_names = list(parent_contexts.keys())
            
            # Calculate similarities
            similarities = []
            for parent_emb in parent_embeddings:
                sim = np.dot(child_embedding, parent_emb) / (
                    np.linalg.norm(child_embedding) * np.linalg.norm(parent_emb)
                )
                similarities.append(sim)
                
            best_idx = np.argmax(similarities)
            best_parent = parent_names[best_idx]
            best_similarity = similarities[best_idx]
            
            logger.info(f"Context-based assignment: child to '{best_parent}' (similarity: {best_similarity:.3f})")
            return best_parent
            
        except Exception as e:
            logger.error(f"Error in context similarity: {e}")
            return split_parents[0]
    
    def _determine_by_children_context(self, child_children: List[str], split_parents: List[str]) -> str:
        """Determine best parent by analyzing the child's own children"""
        # Collect context from the child's children
        children_contexts = []
        for grandchild in child_children[:3]:  # Limit to avoid too much computation
            if grandchild in self.hierarchy['terms']:
                context = self._get_term_context(grandchild)
                if context:
                    children_contexts.append(context)
                    
        if not children_contexts:
            return split_parents[0]
            
        # Use the combined children context to determine best parent
        combined_children_context = ' '.join(children_contexts)
        return self._determine_by_context_similarity(combined_children_context, split_parents)
    
    def apply_fixes(self):
        """Apply the fixes to the hierarchy"""
        logger.info("Applying parent-child relationship fixes...")
        
        changes_made = 0
        parent_child_changes_made = 0
        related_concept_changes_made = 0
        
        for child_term, original_parent in self.affected_children.items():
            # Determine the best split parent
            best_split_parent = self.determine_best_split_parent(child_term, original_parent)
            
            # Update child's parent list
            child_data = self.hierarchy['terms'][child_term]
            old_parents = child_data['parents'].copy()
            
            # Replace the original parent with the best split parent
            new_parents = []
            for parent in old_parents:
                if parent == original_parent:
                    new_parents.append(best_split_parent)
                else:
                    new_parents.append(parent)
                    
            child_data['parents'] = new_parents
            
            # Update the split parent's children list
            if best_split_parent in self.hierarchy['terms']:
                split_parent_data = self.hierarchy['terms'][best_split_parent]
                if 'children' not in split_parent_data:
                    split_parent_data['children'] = []
                if child_term not in split_parent_data['children']:
                    split_parent_data['children'].append(child_term)
                    
            # Remove child from original parent's children (if it still exists)
            if original_parent in self.hierarchy['terms']:
                orig_parent_data = self.hierarchy['terms'][original_parent]
                children = orig_parent_data.get('children', [])
                if child_term in children:
                    children.remove(child_term)
                    
            logger.info(f"Fixed: '{child_term}' parent '{original_parent}' -> '{best_split_parent}'")
            changes_made += 1
            parent_child_changes_made += 1
            
        # Update parent-child relationships in the relationships list
        updated_relationships = []
        for parent, child, level in self.hierarchy['relationships']['parent_child']:
            # If this relationship involves a split parent, update it
            if child in self.affected_children and parent == self.affected_children[child]:
                # Find the new parent for this child
                new_parent = None
                child_data = self.hierarchy['terms'][child]
                for p in child_data['parents']:
                    if p in self.split_terms_mapping.get(parent, []):
                        new_parent = p
                        break
                        
                if new_parent:
                    updated_relationships.append((new_parent, child, level))
                    logger.debug(f"Updated relationship: {parent} -> {new_parent} for child {child}")
                else:
                    # Keep original if we couldn't find the new parent
                    updated_relationships.append((parent, child, level))
            else:
                updated_relationships.append((parent, child, level))
                
        self.hierarchy['relationships']['parent_child'] = updated_relationships
        
        logger.info(f"Applied {changes_made} parent-child relationship fixes")
        
    def apply_related_concept_fixes(self):
        """Apply fixes to related_concepts fields"""
        logger.info("Applying related concept fixes...")
        related_concept_changes_made = 0
        # A set to keep track of which (term, original_related_concept) pairs have been processed
        # to avoid redundant logging or processing if a related concept appears multiple times.
        processed_related_links = set()

        for term_referencing, original_split_related_concepts_list in self.affected_related.items():
            if term_referencing not in self.hierarchy['terms']:
                logger.warning(f"Term '{term_referencing}' found in affected_related but not in hierarchy. Skipping.")
                continue
            term_data = self.hierarchy['terms'][term_referencing]
            
            # Work on a copy of the current related concepts
            current_related_concepts = term_data.get('related_concepts', []).copy()
            new_related_concepts_for_term = [] # This will be the new list for term_data

            # Handle original split related concepts first
            for original_related_concept in original_split_related_concepts_list:
                if (term_referencing, original_related_concept) in processed_related_links:
                    continue # Already processed this specific link

                if original_related_concept in self.split_terms_mapping:
                    best_split_related = self.determine_best_split_parent(term_referencing, original_related_concept)
                    
                    if best_split_related not in new_related_concepts_for_term:
                        new_related_concepts_for_term.append(best_split_related)
                    
                    # Add reciprocal relationship
                    if best_split_related in self.hierarchy['terms']:
                        split_related_data = self.hierarchy['terms'][best_split_related]
                        if 'related_concepts' not in split_related_data:
                            split_related_data['related_concepts'] = []
                        if term_referencing not in split_related_data['related_concepts']:
                            split_related_data['related_concepts'].append(term_referencing)
                    
                    logger.info(f"Fixed related concept for '{term_referencing}': '{original_related_concept}' -> '{best_split_related}'")
                    related_concept_changes_made += 1
                    processed_related_links.add((term_referencing, original_related_concept))
            
            # Add back any non-split related concepts that were already there
            for rc in current_related_concepts:
                if rc not in self.split_terms_mapping: # if it's not a term that was split
                    if rc not in new_related_concepts_for_term: # and not already added
                        new_related_concepts_for_term.append(rc)
                # If it *was* a split term but *not* in original_split_related_concepts_list 
                # for this term_referencing (i.e., it was not identified by identify_affected_related_concepts as needing a fix for *this specific* term_referencing)
                # it implies it shouldn't be processed here or was already handled. 
                # The goal is to replace *only* the identified affected relations.

            term_data['related_concepts'] = new_related_concepts_for_term

        # Update the main relationships.related list
        if 'related' in self.hierarchy['relationships']:
            updated_main_related_list = []
            existing_relationships_tuples = set() # To handle duplicates in original list

            for r_term1, r_term2, r_level in self.hierarchy['relationships']['related']:
                # Skip if this exact tuple has already been processed from the original list
                if (r_term1, r_term2, r_level) in existing_relationships_tuples:
                    continue
                existing_relationships_tuples.add((r_term1, r_term2, r_level))

                new_r_term1, new_r_term2 = r_term1, r_term2

                # If r_term1 was split, find its new sense based on r_term2's updated related_concepts
                if r_term1 in self.split_terms_mapping and r_term2 in self.hierarchy['terms']:
                    r_term2_related = self.hierarchy['terms'][r_term2].get('related_concepts', [])
                    found_match = False
                    for split_sense in self.split_terms_mapping[r_term1]:
                        if split_sense in r_term2_related:
                            new_r_term1 = split_sense
                            found_match = True
                            break
                    if not found_match:
                        # Fallback: if r_term2 does not list any split sense of r_term1, 
                        # but r_term1 was split, we might need to pick one or log a warning.
                        # For now, let's use the semantic best fit if possible.
                        # This situation implies r_term2 might not have r_term1 as a related concept anymore, or it's an error.
                        if r_term1 in self.affected_related.get(r_term2, []): # check if this link was supposed to be fixed
                             best_fit = self.determine_best_split_parent(r_term2, r_term1)
                             new_r_term1 = best_fit
                             logger.debug(f"Fallback: Updating relationships.related for ({r_term1}, {r_term2}) to ({new_r_term1}, ...) based on best fit for r_term2")

                # If r_term2 was split, find its new sense based on r_term1's updated related_concepts
                if r_term2 in self.split_terms_mapping and r_term1 in self.hierarchy['terms']:
                    r_term1_related = self.hierarchy['terms'][r_term1].get('related_concepts', [])
                    found_match = False
                    for split_sense in self.split_terms_mapping[r_term2]:
                        if split_sense in r_term1_related:
                            new_r_term2 = split_sense
                            found_match = True
                            break
                    if not found_match:
                        if r_term2 in self.affected_related.get(r_term1, []): # check if this link was supposed to be fixed
                            best_fit = self.determine_best_split_parent(r_term1, r_term2)
                            new_r_term2 = best_fit
                            logger.debug(f"Fallback: Updating relationships.related for ({r_term1}, {r_term2}) to (..., {new_r_term2}) based on best fit for r_term1")
                
                # Add the potentially updated relationship, ensuring (a,b) is same as (b,a)
                # To avoid duplicates like (termA, termB) and (termB, termA) if levels are same,
                # we can normalize by sorting terms if they are at the same conceptual level of the relationship.
                # However, the original list might have specific directionality intended by `level`.
                # For now, let's just add if the (new_r_term1, new_r_term2, r_level) is not already present.
                # A better deduplication might sort (new_r_term1, new_r_term2) if rel_type is symmetric.
                current_rel_tuple = tuple(sorted((new_r_term1, new_r_term2)) + [r_level]) # Basic normalization for now
                
                is_present = False
                for existing_rel in updated_main_related_list:
                    if tuple(sorted((existing_rel[0], existing_rel[1])) + [existing_rel[2]]) == current_rel_tuple:
                        is_present = True
                        break
                if not is_present:
                    updated_main_related_list.append((new_r_term1, new_r_term2, r_level))

            self.hierarchy['relationships']['related'] = updated_main_related_list
            
        logger.info(f"Applied {related_concept_changes_made} related concept fixes.")
        return related_concept_changes_made
    
    def save_fixed_hierarchy(self, output_file: Optional[str] = None):
        """Save the fixed hierarchy"""
        if output_file is None:
            output_file = self.hierarchy_file.replace('.json', '_fixed.json')
            
        # Add metadata about the fixes
        fix_metadata = {
            'parent_child_fixes_applied': True,
            'fix_timestamp': datetime.now().isoformat(),
            'affected_children_count': len(self.affected_children),
            'affected_related_concepts_count': len(self.affected_related),
            'split_terms_count': len(self.split_terms_mapping)
        }
        
        if 'metadata' not in self.hierarchy:
            self.hierarchy['metadata'] = {}
        self.hierarchy['metadata'].update(fix_metadata)
        
        logger.info(f"Saving fixed hierarchy to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(self.hierarchy, f, indent=2)
            
        return output_file
    
    def update_level_files(self):
        """Update the level-specific files with the fixed parent-child relationships"""
        logger.info("Updating level-specific files...")
        
        level_files_updated = 0
        
        for level in range(4):  # Levels 0-3
            level_dir = f"data/final/lv{level}"
            metadata_file = os.path.join(level_dir, f"lv{level}_metadata.json")
            
            if os.path.exists(metadata_file):
                logger.info(f"Updating {metadata_file}")
                
                # Load level metadata
                with open(metadata_file, 'r') as f:
                    level_metadata = json.load(f)
                    
                # Update parents
                updates_made_in_file = 0 # Renamed from updates_made to avoid conflict
                for term_name_in_level_file, term_data_in_level_file in level_metadata.items():
                    # Update parents
                    current_parents_in_level_file = term_data_in_level_file.get('parents', [])
                    new_parents_for_level_file = []
                    parent_list_changed = False

                    if term_name_in_level_file == 'writing studies':
                        logger.info(f"DEBUG: Checking parents for 'writing studies'. Current: {current_parents_in_level_file}")

                    for p_idx, parent_name_in_level_file in enumerate(current_parents_in_level_file):
                        if term_name_in_level_file == 'writing studies' and parent_name_in_level_file == 'english':
                            logger.info(f"DEBUG: 'writing studies' has parent 'english'. Checking split_terms_mapping.")
                        
                        if parent_name_in_level_file in self.split_terms_mapping:
                            if term_name_in_level_file == 'writing studies' and parent_name_in_level_file == 'english':
                                logger.info(f"DEBUG: Parent 'english' for 'writing studies' IS in split_terms_mapping. Calling determine_best_split_parent.")

                            best_split_sense_for_parent = self.determine_best_split_parent(term_name_in_level_file, parent_name_in_level_file)
                            
                            if term_name_in_level_file == 'writing studies' and parent_name_in_level_file == 'english':
                                logger.info(f"DEBUG: Best split sense for parent 'english' of 'writing studies' is '{best_split_sense_for_parent}'")

                            new_parents_for_level_file.append(best_split_sense_for_parent)
                            if parent_name_in_level_file != best_split_sense_for_parent:
                                parent_list_changed = True
                        else:
                            new_parents_for_level_file.append(parent_name_in_level_file)
                    
                    if parent_list_changed:
                        term_data_in_level_file['parents'] = new_parents_for_level_file
                        updates_made_in_file += 1
                        logger.info(f"Updated parents for '{term_name_in_level_file}' in level {level} metadata from {current_parents_in_level_file} to {new_parents_for_level_file}")

                    # Update related_concepts (which is a dict in level files)
                    if 'related_concepts' in term_data_in_level_file and isinstance(term_data_in_level_file['related_concepts'], dict):
                        related_dict_changed = False
                        for rc_level_key, rc_list in term_data_in_level_file['related_concepts'].items(): # e.g. "lv0", ["termA", ...]
                            new_rc_list_for_level = []
                            list_changed_for_this_rc_level = False
                            for related_concept_name in rc_list:
                                if related_concept_name == 'english' and term_name_in_level_file == 'writing studies':
                                    logger.info(f"DEBUG: Processing 'writing studies' related concept 'english' in lv {rc_level_key}")

                                if related_concept_name in self.split_terms_mapping: # <--- POINT A
                                    # This related concept was split. Determine the best new sense.
                                    # The "child_term" for determine_best_split_parent is the term_name_in_level_file
                                    if term_name_in_level_file == 'writing studies' and related_concept_name == 'english':
                                        logger.info(f"DEBUG: 'english' is in split_terms_mapping. Calling determine_best_split_parent('{term_name_in_level_file}', '{related_concept_name}')")
                                    
                                    best_split_sense = self.determine_best_split_parent(term_name_in_level_file, related_concept_name)
                                    
                                    if term_name_in_level_file == 'writing studies' and related_concept_name == 'english':
                                        logger.info(f"DEBUG: best_split_sense for english (context: writing studies) = '{best_split_sense}'")

                                    if best_split_sense not in new_rc_list_for_level:
                                        new_rc_list_for_level.append(best_split_sense)
                                    if related_concept_name != best_split_sense:
                                        list_changed_for_this_rc_level = True
                                        related_dict_changed = True
                                else:
                                    if related_concept_name not in new_rc_list_for_level:
                                        new_rc_list_for_level.append(related_concept_name)
                            
                            if list_changed_for_this_rc_level:
                                term_data_in_level_file['related_concepts'][rc_level_key] = new_rc_list_for_level
                        
                        if related_dict_changed:
                            updates_made_in_file += 1 
                            logger.debug(f"Updated related_concepts for '{term_name_in_level_file}' in level {level} metadata")
                            
                if updates_made_in_file > 0:
                    # Save updated metadata
                    backup_file = metadata_file.replace('.json', '_backup_before_parent_fix.json')
                    # Ensure backup directory exists or handle error
                    if os.path.exists(metadata_file):
                        os.rename(metadata_file, backup_file)
                    else: # if metadata_file was already moved by a previous level run, we write to original name but log it
                        logger.warning(f"Original metadata file {metadata_file} not found for backup during level {level} update. Already backed up?")
                    
                    with open(metadata_file, 'w') as f:
                        json.dump(level_metadata, f, indent=2)
                        
                    logger.info(f"Updated {updates_made_in_file} entries in {metadata_file} (backup: {backup_file})")
                    level_files_updated += 1
                else:
                    logger.info(f"No updates needed for {metadata_file}")
            else:
                logger.warning(f"Level metadata file not found: {metadata_file}")
                
        logger.info(f"Updated {level_files_updated} level files")
    
    def generate_fix_report(self, output_file: str = "data/parent_child_fix_report.md"):
        """Generate a detailed report of the fixes applied"""
        logger.info(f"Generating fix report: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("# Parent-Child Relationship Fix Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total split terms**: {len(self.split_terms_mapping)}\n")
            f.write(f"- **Affected children**: {len(self.affected_children)}\n")
            f.write(f"- **Fixes applied (parent-child)**: {len(self.affected_children)}\n")
            f.write(f"- **Affected terms (related concepts)**: {len(self.affected_related)}\n")
            f.write("\n")
            
            f.write("## Split Terms\n\n")
            for original, splits in self.split_terms_mapping.items():
                f.write(f"- **{original}** → {', '.join(splits)}\n")
            f.write("\n")
            
            f.write("## Fixed Parent-Child Relationships\n\n")
            
            # Group by original parent
            parent_children = defaultdict(list)
            for child, parent in self.affected_children.items():
                parent_children[parent].append(child)
                
            for original_parent, children in parent_children.items():
                f.write(f"### {original_parent}\n\n")
                f.write(f"Split into: {', '.join(self.split_terms_mapping[original_parent])}\n\n")
                f.write("Child reassignments:\n")
                
                for child in children:
                    # Find which split parent this child was assigned to
                    if child in self.hierarchy['terms']:
                        child_parents = self.hierarchy['terms'][child]['parents']
                        new_parent = None
                        for p in child_parents:
                            if p in self.split_terms_mapping[original_parent]:
                                new_parent = p
                                break
                        f.write(f"- **{child}** → {new_parent or 'Unknown'}\n")
                f.write("\n")
                
            f.write("## Fixed Related Concepts\n\n")
            # Group by original split term that was a related_concept
            original_related_references = defaultdict(list)
            for term, original_rels_list in self.affected_related.items():
                for original_rel in original_rels_list:
                    original_related_references[original_rel].append(term)
            
            for original_related, terms_referencing_it in original_related_references.items():
                f.write(f"### Original Related Concept: {original_related}\n\n")
                if original_related in self.split_terms_mapping:
                    f.write(f"Split into: {', '.join(self.split_terms_mapping[original_related])}\n\n")
                f.write("Term reassignments (term → new related concept it points to):\n")
                
                for term in terms_referencing_it:
                    if term in self.hierarchy['terms']:
                        new_related_concepts_for_term = self.hierarchy['terms'][term].get('related_concepts', [])
                        assigned_new_related = "Unknown"
                        for new_rc in new_related_concepts_for_term:
                            if original_related in self.split_terms_mapping and new_rc in self.split_terms_mapping[original_related]:
                                assigned_new_related = new_rc
                                break
                            elif new_rc == original_related and original_related not in self.split_terms_mapping: # It wasn't split, but was re-added
                                assigned_new_related = new_rc
                                break
                        f.write(f"- **{term}** → {assigned_new_related}\n")
                f.write("\n")
                
        logger.info(f"Fix report saved to {output_file}")
    
    def run_complete_fix(self):
        """Run the complete fix process"""
        logger.info("Starting complete parent-child relationship fix process...")
        
        # Load data
        self.load_data()
        
        # Build mappings and identify problems
        self.build_split_mapping()
        self.identify_affected_children()
        self.identify_affected_related_concepts()
        
        if not self.affected_children and not self.affected_related:
            logger.info("No parent-child or related concept relationship issues found. Nothing to fix.")
            return
        
        # Apply fixes
        self.apply_fixes()
        self.apply_related_concept_fixes()
        
        # Save results - User requested to skip this for the main hierarchy file
        # fixed_hierarchy_file = self.save_fixed_hierarchy() 
        logger.info("Skipping save of the main fixed hierarchy file as per user request.")

        self.update_level_files()
        self.generate_fix_report()
        
        logger.info("Parent-child relationship fix process completed successfully!")
        # logger.info(f"Fixed hierarchy saved to: {fixed_hierarchy_file}")
        logger.info("Level-specific files have been updated.")
    
    def run_level_file_update_only(self):
        """Run only the parts necessary to update level files."""
        logger.info("Starting process to update level-specific files only...")

        # Load data - needed to understand splits and current hierarchy state
        self.load_data()

        # Build mappings and identify problems - needed to know what to fix
        self.build_split_mapping()
        self.identify_affected_children()
        self.identify_affected_related_concepts()

        if not self.affected_children and not self.affected_related:
            logger.info("No parent-child or related concept relationship issues identified based on the main hierarchy. Level files may not require extensive updates based on this script's primary logic, but will be checked.")
            # Still proceed to update_level_files as it might have its own checks or ensure consistency
            # based on the loaded (though perhaps not "fixed" if no affected items) hierarchy.
        else:
            # Apply fixes to self.hierarchy (in-memory) so update_level_files has correct data
            logger.info("Applying fixes to in-memory hierarchy to guide level file updates...")
            self.apply_fixes()
            self.apply_related_concept_fixes()
            logger.info("In-memory hierarchy updated.")

        # Update level files using the (potentially modified) in-memory hierarchy
        self.update_level_files()
        
        # Generate a report about the changes made to level files
        self.generate_fix_report(output_file="data/level_files_update_report_via_fix_script.md")
        
        logger.info("Level-specific file update process completed!")
        logger.info("Main hierarchy file was NOT modified on disk.")


def main():
    """Main function to run the fix process"""
    fixer = ParentChildRelationshipFixer()
    # For this request, we only want to update level files.
    # We can call run_complete_fix and it will skip saving the main hierarchy due to the above change.
    # Alternatively, a more targeted method could be used if it was designed for just this.
    # Let's assume the change in run_complete_fix is sufficient.
    # To be more explicit and potentially cleaner, let's add a specific method.
    fixer.run_level_file_update_only()


if __name__ == "__main__":
    main() 