"""
DSPy signature creation, parsing, and structured output handling.

This module focuses on the declarative signature approach that aligns with DSPy 2024-2025 
best practices, handling signature metadata parsing and dynamic class creation for both 
text and structured outputs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import dspy
from pydantic import BaseModel

# Try to import prompt optimization, but handle gracefully if not available
try:
    from prompt_optimization.core import save_prompt
    PROMPT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PROMPT_OPTIMIZATION_AVAILABLE = False
    save_prompt = None

logger = logging.getLogger(__name__)


def _parse_signature_from_prompt(
    system_prompt: str,
    user_prompt: str,
    signature_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Parse DSPy signature information from optimized prompts.
    
    This function analyzes optimized prompts to extract signature metadata
    including input fields, output fields, and their descriptions.
    
    Args:
        system_prompt: Optimized system prompt
        user_prompt: Optimized user prompt template
        signature_metadata: Optional metadata containing signature information
        
    Returns:
        Dictionary with signature metadata:
        {
            'input_fields': {'field_name': 'description', ...},
            'output_fields': {'field_name': 'description', ...},
            'instructions': 'combined prompt instructions',
            'signature_str': 'input_field1, input_field2 -> output_field1, output_field2'
        }
    """
    # If signature_metadata is provided, use it for deterministic parsing (preferred approach)
    if signature_metadata is not None:
        # Handle both new format (from common.py) and legacy formats
        input_fields = signature_metadata.get('input_fields', signature_metadata.get('inputs', {}))
        output_fields = signature_metadata.get('output_fields', signature_metadata.get('outputs', {}))
        instructions = signature_metadata.get('instructions', system_prompt or "")
        signature_str = signature_metadata.get('signature_str', "prompt -> response")
        predictor_type = signature_metadata.get('predictor_type', None)
        
        signature_info = {
            'input_fields': input_fields,
            'output_fields': output_fields,
            'instructions': instructions,
            'signature_str': signature_str,
            'predictor_type': predictor_type
        }
        
        # Build signature string from metadata if not provided or is default
        if signature_str == "prompt -> response" and input_fields and output_fields:
            input_field_names = list(input_fields.keys())
            output_field_names = list(output_fields.keys())
            signature_info['signature_str'] = f"{', '.join(input_field_names)} -> {', '.join(output_field_names)}"
        
        logger.debug(f"Using signature metadata: {signature_info['signature_str']}")
        return signature_info
    
    # Fallback to regex inference when metadata is not available
    signature_info = {
        'input_fields': {},
        'output_fields': {},
        'instructions': system_prompt or "",
        'signature_str': "prompt -> response"
    }
    
    # Safely handle None values and build instructions from available parts
    instructions_parts = []
    if system_prompt:
        instructions_parts.append(system_prompt)
    if user_prompt:
        instructions_parts.append(user_prompt)
    
    combined_text = " ".join(instructions_parts).lower()
    signature_info['instructions'] = " ".join(instructions_parts) or "Process the input"
    
    # Common input field patterns from optimized prompts
    input_patterns = [
        (r"field named ['\"]?text['\"]?", "text", "Input text to process"),
        (r"input field named ['\"]?text['\"]?", "text", "Input text to process"),
        (r"field named ['\"]?term['\"]?", "term", "Term to analyze"),
        (r"input field named ['\"]?term['\"]?", "term", "Term to analyze"),
        (r"field named ['\"]?content['\"]?", "content", "Content to process"),
        (r"input field named ['\"]?content['\"]?", "content", "Content to process"),
        (r"field named ['\"]?input['\"]?", "input", "User input to process"),
        (r"given.{0,20}text", "text", "Given text input"),
        (r"provided.{0,20}text", "text", "Provided text input"),
        (r"analyze.{0,20}text", "text", "Text to analyze"),
        (r"process.{0,20}text", "text", "Text to process")
    ]
    
    # Extract input fields - collect all matches instead of breaking on first
    for pattern, field_name, description in input_patterns:
        if re.search(pattern, combined_text):
            signature_info['input_fields'][field_name] = description
    
    # Default to 'text' if no specific field found
    if not signature_info['input_fields']:
        signature_info['input_fields']['text'] = "Input text to process"
    
    # Common output field patterns
    output_patterns = [
        (r"reasoning.{0,50}extraction", ["reasoning", "extraction"]),
        (r"reasoning.{0,50}output", ["reasoning", "output"]),
        (r"explanation.{0,50}result", ["explanation", "result"]),
        (r"analysis.{0,50}conclusion", ["analysis", "conclusion"]),
        (r"thinking.{0,50}response", ["thinking", "response"]),
        (r"chain.{0,20}of.{0,20}thought", ["reasoning", "response"]),
        (r"step.{0,20}by.{0,20}step", ["reasoning", "response"])
    ]
    
    # Extract output fields - collect all matches instead of breaking on first
    output_found = False
    for pattern, fields in output_patterns:
        if re.search(pattern, combined_text):
            for i, field in enumerate(fields):
                if field == "reasoning":
                    signature_info['output_fields'][field] = "Step-by-step reasoning"
                elif field == "extraction":
                    signature_info['output_fields'][field] = "Extracted information"
                elif field == "output":
                    signature_info['output_fields'][field] = "Final output"
                elif field == "result":
                    signature_info['output_fields'][field] = "Final result"
                elif field == "response":
                    signature_info['output_fields'][field] = "Response"
                elif field == "explanation":
                    signature_info['output_fields'][field] = "Explanation"
                elif field == "analysis":
                    signature_info['output_fields'][field] = "Analysis"
                elif field == "conclusion":
                    signature_info['output_fields'][field] = "Conclusion"
                else:
                    signature_info['output_fields'][field] = f"Output {field}"
            output_found = True
    
    # Default to single response field if no specific output structure found
    if not output_found:
        signature_info['output_fields']['response'] = "Response"
    
    # Build signature string
    input_fields = list(signature_info['input_fields'].keys())
    output_fields = list(signature_info['output_fields'].keys())
    signature_info['signature_str'] = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
    
    return signature_info


def _create_dynamic_signature(
    signature_info: Dict[str, Any],
    response_model: Optional[Type[BaseModel]] = None
) -> Type[dspy.Signature]:
    """
    Create dynamic DSPy signature from signature metadata or parsed prompt information.
    
    Prioritizes direct signature metadata over inferred information, supporting both
    optimization-time metadata and runtime text inference.
    """
    # Create signature class dynamically
    class_name = "DynamicSignature"
    
    # Build field definitions
    field_definitions = {}
    
    # Add input fields
    for field_name, description in signature_info['input_fields'].items():
        field_definitions[field_name] = dspy.InputField(desc=description)
    
    # Add output fields
    if response_model is not None:
        # Use structured output for TypedPredictor
        field_definitions['output'] = dspy.OutputField(desc="Structured output")
    else:
        # Use text-based output fields
        for field_name, description in signature_info['output_fields'].items():
            field_definitions[field_name] = dspy.OutputField(desc=description)
    
    # Create the signature class first
    signature_class = type(class_name, (dspy.Signature,), field_definitions)
    
    # Enhanced instruction handling for better DSPy compatibility
    instructions = signature_info.get('instructions', "")
    if instructions:
        # Set both __doc__ and dedicated instruction attribute for maximum compatibility
        signature_class.__doc__ = instructions
        
        # For DSPy 2024-2025, also set the instructions attribute directly
        if hasattr(dspy.Signature, 'instructions'):
            signature_class.instructions = instructions
        
        # For optimization metadata, also support predictor type hints
        predictor_type = signature_info.get('predictor_type', None)
        if predictor_type:
            signature_class._recommended_predictor_type = predictor_type
        
        # Only add instruction field if explicitly requested and not already present
        should_add_instruction = signature_info.get('add_instruction_field', False)
        legacy_add_instruction = signature_info.get('legacy_add_instruction_field', False)
        
        if should_add_instruction and 'instruction' not in field_definitions:
            logger.debug("Adding instruction field to signature for comprehensive DSPy integration")
            signature_class.instruction = dspy.InputField(desc="Detailed system instructions for processing")
            # Also update signature_info to keep input mapping consistent
            signature_info.setdefault('input_fields', {})['instruction'] = "Detailed system instructions for processing"
        elif should_add_instruction and 'instruction' in field_definitions:
            logger.debug("Skipping instruction field addition - already present in signature")
        elif legacy_add_instruction and 'instruction' not in field_definitions and len(instructions) > 50:
            # Legacy behavior: auto-add for long instructions only if explicitly enabled
            logger.debug("Auto-adding instruction field for long instructions (legacy behavior enabled)")
            signature_class.instruction = dspy.InputField(desc="Detailed system instructions for processing")
            # Also update signature_info to keep input mapping consistent
            signature_info.setdefault('input_fields', {})['instruction'] = "Detailed system instructions for processing"
        else:
            logger.debug("Skipping instruction field auto-add - not explicitly requested")
    
    return signature_class


def _extract_content_for_signature(
    messages: List[Dict[str, str]], 
    signature_info: Dict[str, Any]
) -> Dict[str, str]:
    """
    Extract content from messages for signature input fields.
    
    Returns dictionary mapping field names to content values.
    """
    field_mapping = {}
    input_fields = signature_info.get('input_fields', {})
    
    # Extract all message content
    all_content = []
    user_content = []
    system_content = []
    
    for message in messages:
        content = message.get("content", "")
        if content and isinstance(content, str):
            content = content.strip()
            if content:
                all_content.append(content)
                
                role = message.get("role", "").lower().strip()
                if role == "user":
                    user_content.append(content)
                elif role == "system":
                    system_content.append(content)
    
    combined_content = "\n\n".join(all_content)
    combined_user = "\n\n".join(user_content) if user_content else combined_content
    combined_system = "\n\n".join(system_content)
    
    # Map content to signature input fields
    for field_name in input_fields.keys():
        if field_name in ["text", "input", "content"]:
            # Primary content fields get user content or all content
            field_mapping[field_name] = combined_user or combined_content
        elif field_name == "term":
            # For term analysis, try to extract key terms or use full content
            field_mapping[field_name] = combined_user or combined_content
        elif field_name == "system":
            # System field gets system messages
            field_mapping[field_name] = combined_system or combined_content
        elif field_name == "instruction":
            # Instruction field gets instructions from signature_info
            field_mapping[field_name] = signature_info.get('instructions', '')
        else:
            # Unknown fields get full combined content
            field_mapping[field_name] = combined_content
    
    # Ensure we have at least some content for each field
    for field_name in input_fields.keys():
        if not field_mapping.get(field_name):
            field_mapping[field_name] = combined_content or "No content provided"
    
    return field_mapping


def _parse_structured_output(
    result: Any,
    response_model: Type[BaseModel]
) -> BaseModel:
    """
    Parse DSPy prediction result into structured Pydantic model.
    
    Compatible with both Pydantic v1 and v2. Handles JSON extraction from
    fenced code blocks and provides improved error messages.
    
    Args:
        result: DSPy prediction result
        response_model: Pydantic model class for structured output
        
    Returns:
        Parsed Pydantic model instance
        
    Raises:
        ValueError: If output cannot be parsed into the expected model
    """
    # Extract text content from DSPy result
    if hasattr(result, 'output'):
        text_output = result.output
    elif hasattr(result, 'response'):
        text_output = result.response
    else:
        text_output = str(result)
    
    # Try to parse as JSON with improved extraction
    try:
        # Look for fenced code blocks first - handle both object and array JSON
        code_block_pattern = r'```(?:json)?\s*([{\[].*?[}\]])\s*```'
        code_match = re.search(code_block_pattern, text_output, re.DOTALL)
        
        if code_match:
            json_str = code_match.group(1)
        else:
            # Try to balance braces/brackets for JSON extraction
            def find_balanced_json(text, start_char, end_char):
                count = 0
                start_idx = -1
                end_idx = -1
                
                for i, char in enumerate(text):
                    if char == start_char:
                        if start_idx == -1:
                            start_idx = i
                        count += 1
                    elif char == end_char:
                        count -= 1
                        if count == 0 and start_idx != -1:
                            end_idx = i + 1
                            break
                
                if start_idx != -1 and end_idx != -1:
                    return text[start_idx:end_idx]
                return None
            
            # Try object notation first
            json_str = find_balanced_json(text_output, '{', '}')
            
            # If no object, try array notation
            if json_str is None:
                json_str = find_balanced_json(text_output, '[', ']')
            
            if json_str is None:
                # Fallback to original regex patterns
                for pattern in [r'\{.*\}', r'\[.*\]']:
                    match = re.search(pattern, text_output, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        break
                
                if json_str is None:
                    raise ValueError("No JSON structure found")
        
        # Parse the JSON
        data = json.loads(json_str)
        
        # Handle list-shaped JSON: if parsed JSON is a list and model has a single list field
        if isinstance(data, list):
            # Check if the model has a single field that expects a list
            fields = getattr(response_model, 'model_fields', None) or response_model.__fields__
            
            if len(fields) == 1:
                field_name, field_info = next(iter(fields.items()))
                
                # Try to determine if the field expects a list
                field_type = None
                if hasattr(field_info, 'annotation'):  # Pydantic v2
                    field_type = field_info.annotation
                elif hasattr(field_info, 'type_'):  # Pydantic v1
                    field_type = field_info.type_
                
                # Check if it's a list type (simple heuristic)
                is_list_field = (
                    field_type and (
                        str(field_type).startswith('typing.List') or
                        str(field_type).startswith('list') or
                        (hasattr(field_type, '__origin__') and field_type.__origin__ is list)
                    )
                )
                
                if is_list_field:
                    logger.debug(f"Handling list-shaped JSON: populating single list field '{field_name}'")
                    return response_model(**{field_name: data})
        
        # Handle regular object-shaped JSON
        if isinstance(data, dict):
            return response_model(**data)
        else:
            raise ValueError(f"Parsed JSON is neither dict nor compatible list: {type(data)}")
        
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.debug(f"JSON parsing failed: {e}")
    
    # Fallback: try to create model with the text as the first field
    try:
        # Pydantic v2 compatibility - prefer model_fields over __fields__
        fields = getattr(response_model, 'model_fields', None) or response_model.__fields__
        first_field = next(iter(fields.keys()))
        return response_model(**{first_field: text_output})
    except Exception:
        # Last resort: create with generic 'response' field if it exists
        try:
            return response_model(response=text_output)
        except Exception:
            # If all else fails, raise an error with helpful message
            raise ValueError(
                f"Could not parse DSPy output into {response_model.__name__}. "
                f"Output: {text_output[:200]}... "
                f"Please ensure your prompts enforce JSON output format."
            )


def enhance_prompt_for_dspy(
    optimized_prompt: str,
    prompt_type: str = "user"
) -> Dict[str, Any]:
    """
    Enhance prompt optimization output for better DSPy integration.
    
    Args:
        optimized_prompt: The optimized prompt content
        prompt_type: Type of prompt ("system" or "user")
        
    Returns:
        Dictionary with enhanced format:
        {
            'original_prompt': str,
            'signature_metadata': {
                'input_fields': {...},
                'output_fields': {...},
                'instructions': str,
                'signature_str': str
            },
            'dspy_compatibility': {
                'chainofthought_ready': bool,
                'structured_output_ready': bool,
                'field_extraction_quality': str
            },
            'enhancement_suggestions': List[str]
        }
        
    Examples:
        enhanced = enhance_prompt_for_dspy(optimized_prompt)
        if enhanced['dspy_compatibility']['chainofthought_ready']:
            print("Prompt is ready for ChainOfThought integration")
    """
    enhancement_result = {
        'original_prompt': optimized_prompt,
        'signature_metadata': {},
        'dspy_compatibility': {
            'chainofthought_ready': False,
            'structured_output_ready': False,
            'field_extraction_quality': 'unknown'
        },
        'enhancement_suggestions': []
    }
    
    try:
        # Analyze prompt for DSPy signature compatibility
        if prompt_type == "user":
            # For user prompts, create a minimal system prompt for analysis
            system_prompt = "Process the input according to the instructions."
            user_prompt = optimized_prompt
        else:
            # For system prompts, create minimal user prompt
            system_prompt = optimized_prompt
            user_prompt = "Please process the provided input."
        
        # Extract signature metadata
        try:
            signature_info = _parse_signature_from_prompt(system_prompt, user_prompt, None)
            enhancement_result['signature_metadata'] = signature_info
            
            # Assess DSPy compatibility
            input_fields = signature_info.get('input_fields', {})
            output_fields = signature_info.get('output_fields', {})
            
            # Check ChainOfThought readiness
            if 'reasoning' in output_fields or 'thinking' in output_fields:
                enhancement_result['dspy_compatibility']['chainofthought_ready'] = True
            elif "step by step" in optimized_prompt.lower() or "reasoning" in optimized_prompt.lower():
                enhancement_result['dspy_compatibility']['chainofthought_ready'] = True
                enhancement_result['enhancement_suggestions'].append(
                    "Consider adding explicit 'reasoning' output field for better ChainOfThought integration"
                )
            
            # Check structured output readiness
            if len(output_fields) > 1:
                enhancement_result['dspy_compatibility']['structured_output_ready'] = True
            elif "json" in optimized_prompt.lower() or "format" in optimized_prompt.lower():
                enhancement_result['dspy_compatibility']['structured_output_ready'] = True
            
            # Assess field extraction quality
            if len(input_fields) > 0 and len(output_fields) > 0:
                if len(input_fields) == 1 and list(input_fields.keys())[0] in ['text', 'input', 'content']:
                    enhancement_result['dspy_compatibility']['field_extraction_quality'] = 'good'
                else:
                    enhancement_result['dspy_compatibility']['field_extraction_quality'] = 'excellent'
            else:
                enhancement_result['dspy_compatibility']['field_extraction_quality'] = 'poor'
                
        except Exception as e:
            enhancement_result['enhancement_suggestions'].append(f"Signature parsing failed: {e}")
        
        # Generate enhancement suggestions
        if not enhancement_result['dspy_compatibility']['chainofthought_ready']:
            enhancement_result['enhancement_suggestions'].append(
                "Consider adding reasoning steps or 'step by step' language for better ChainOfThought support"
            )
        
        if not enhancement_result['dspy_compatibility']['structured_output_ready']:
            enhancement_result['enhancement_suggestions'].append(
                "Consider defining clear output structure with multiple fields for better structured output"
            )
        
        if enhancement_result['dspy_compatibility']['field_extraction_quality'] == 'poor':
            enhancement_result['enhancement_suggestions'].append(
                "Consider explicitly mentioning input field names like 'text', 'term', or 'content'"
            )
        
        # Check for common DSPy optimization patterns
        prompt_lower = optimized_prompt.lower()
        if "analyze" in prompt_lower and "extract" in prompt_lower:
            enhancement_result['enhancement_suggestions'].append(
                "Good: Analysis and extraction pattern detected - works well with DSPy signatures"
            )
        
        if "given" in prompt_lower and "provide" in prompt_lower:
            enhancement_result['enhancement_suggestions'].append(
                "Good: Clear input-output pattern detected - suitable for DSPy predictors"
            )
        
    except Exception as e:
        enhancement_result['enhancement_suggestions'].append(f"Enhancement analysis failed: {e}")
    
    return enhancement_result


def save_enhanced_prompt(
    prompt_key: str,
    prompt_content: str,
    signature_metadata: Dict[str, Any],
    output_dir: str = "data/prompts"
) -> str:
    """
    Save prompts with enhanced DSPy metadata.
    
    Args:
        prompt_key: Unique key for the prompt (e.g., "lv0_s1_user_enhanced")
        prompt_content: The prompt content
        signature_metadata: DSPy signature metadata from enhance_prompt_for_dspy
        output_dir: Directory to save enhanced prompts
        
    Returns:
        Path to the saved enhanced prompt file
        
    Examples:
        signature_metadata = enhance_prompt_for_dspy(prompt)['signature_metadata']
        file_path = save_enhanced_prompt("lv0_s1_user", prompt, signature_metadata)
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create enhanced prompt structure
        enhanced_prompt = {
            'content': prompt_content,
            'signature_metadata': signature_metadata,
            'dspy_integration': {
                'signature_str': signature_metadata.get('signature_str', 'prompt -> response'),
                'input_fields': signature_metadata.get('input_fields', {}),
                'output_fields': signature_metadata.get('output_fields', {}),
                'instructions': signature_metadata.get('instructions', prompt_content)
            },
            'version': '1.0',
            'enhanced_by': 'llm_signatures.py'
        }
        
        # Save to JSON file
        file_path = output_path / f"{prompt_key}_enhanced.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_prompt, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced prompt saved to: {file_path}")
        
        # Try to integrate with existing prompt optimization system if available
        if PROMPT_OPTIMIZATION_AVAILABLE:
            try:
                # Import and use the save_prompt function from prompt optimization with signature metadata
                save_prompt(f"{prompt_key}_enhanced", prompt_content, signature_metadata=signature_metadata)
                logger.debug(f"Also saved to prompt optimization system with metadata: {prompt_key}_enhanced")
            except Exception as e:
                logger.debug(f"Could not save to prompt optimization system: {e}")
        
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Failed to save enhanced prompt: {e}")
        raise


def validate_dspy_compliance(
    use_case: str,
    prompt_content: Optional[str] = None,
    signature_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate prompt artifacts against DSPy 2024-2025 best practices.
    
    This function checks prompt optimization results for alignment with modern DSPy
    principles including declarative programming, signature-based approaches,
    and metric-driven development.
    
    Args:
        use_case: Use case identifier (e.g., "lv0_s1")
        prompt_content: Optional prompt content to validate
        signature_metadata: Optional signature metadata to validate
        
    Returns:
        Dictionary with compliance results:
        {
            'overall_score': float,  # 0.0 to 1.0
            'compliance_checks': {
                'has_signature_metadata': bool,
                'declarative_structure': bool,
                'modular_composition_ready': bool,
                'metric_driven_ready': bool,
                'separation_of_concerns': bool
            },
            'recommendations': List[str],
            'dspy_version_compatibility': str,
            'improvement_priority': List[str]
        }
        
    Examples:
        # Validate existing optimization artifacts
        result = validate_dspy_compliance("lv0_s1")
        print(f"DSPy compliance score: {result['overall_score']:.2f}")
        
        # Validate specific content
        result = validate_dspy_compliance("custom", prompt_content, signature_metadata)
    """
    validation_result = {
        'overall_score': 0.0,
        'compliance_checks': {
            'has_signature_metadata': False,
            'declarative_structure': False,
            'modular_composition_ready': False,
            'metric_driven_ready': False,
            'separation_of_concerns': False
        },
        'recommendations': [],
        'dspy_version_compatibility': '2024-2025',
        'improvement_priority': []
    }
    
    try:
        # Load prompt data if not provided
        if prompt_content is None or signature_metadata is None:
            try:
                from prompt_optimization.core import load_prompt
                
                # Try to load system and user prompts
                system_data = None
                user_data = None
                
                try:
                    system_data = load_prompt(f"{use_case}_system")
                    if not prompt_content:
                        prompt_content = system_data.get("content", "")
                    if not signature_metadata:
                        signature_metadata = system_data.get("signature_metadata")
                except (FileNotFoundError, KeyError):
                    pass
                
                try:
                    user_data = load_prompt(f"{use_case}_user")
                    if not prompt_content:
                        prompt_content = user_data.get("content", "")
                    if not signature_metadata:
                        signature_metadata = user_data.get("signature_metadata")
                except (FileNotFoundError, KeyError):
                    pass
                
            except ImportError:
                validation_result['recommendations'].append(
                    "Prompt optimization system not available for validation"
                )
        
        # Check 1: Signature metadata presence
        if signature_metadata is not None:
            validation_result['compliance_checks']['has_signature_metadata'] = True
            validation_result['recommendations'].append(
                "✓ Signature metadata available - enables deterministic DSPy integration"
            )
        else:
            validation_result['recommendations'].append(
                "⚠ Missing signature metadata - relying on text inference (less reliable)"
            )
            validation_result['improvement_priority'].append("Add signature metadata extraction")
        
        # Check 2: Declarative structure
        if signature_metadata:
            input_fields = signature_metadata.get('input_fields', {})
            output_fields = signature_metadata.get('output_fields', {})
            
            if input_fields and output_fields:
                validation_result['compliance_checks']['declarative_structure'] = True
                validation_result['recommendations'].append(
                    f"✓ Declarative structure: {len(input_fields)} inputs → {len(output_fields)} outputs"
                )
            else:
                validation_result['recommendations'].append(
                    "⚠ Incomplete declarative structure - missing field definitions"
                )
        elif prompt_content:
            # Fallback analysis for prompt content
            if "input" in prompt_content.lower() and "output" in prompt_content.lower():
                validation_result['compliance_checks']['declarative_structure'] = True
                validation_result['recommendations'].append(
                    "✓ Basic declarative structure detected in prompt text"
                )
        
        # Check 3: Modular composition readiness
        modular_indicators = []
        if signature_metadata:
            predictor_type = signature_metadata.get('predictor_type')
            if predictor_type in ['ChainOfThought', 'TypedPredictor']:
                modular_indicators.append(f"Structured predictor type: {predictor_type}")
                
            if len(signature_metadata.get('output_fields', {})) > 1:
                modular_indicators.append("Multiple output fields support composition")
                
        if prompt_content:
            if "reasoning" in prompt_content.lower():
                modular_indicators.append("Reasoning pattern detected")
            if "step by step" in prompt_content.lower():
                modular_indicators.append("Step-by-step pattern detected")
                
        if modular_indicators:
            validation_result['compliance_checks']['modular_composition_ready'] = True
            validation_result['recommendations'].extend([f"✓ {indicator}" for indicator in modular_indicators])
        else:
            validation_result['recommendations'].append(
                "⚠ Limited modular composition support - consider adding reasoning steps"
            )
            validation_result['improvement_priority'].append("Enhance modular composition patterns")
        
        # Check 4: Metric-driven development readiness
        if signature_metadata:
            # Check for clear output structure that enables metrics
            output_fields = signature_metadata.get('output_fields', {})
            if 'reasoning' in output_fields and len(output_fields) >= 2:
                validation_result['compliance_checks']['metric_driven_ready'] = True
                validation_result['recommendations'].append(
                    "✓ Multi-field output enables comprehensive metric evaluation"
                )
            elif len(output_fields) >= 1:
                validation_result['recommendations'].append(
                    "⚠ Basic metric support - consider adding reasoning field for richer evaluation"
                )
        
        # Check 5: Separation of concerns
        if signature_metadata and prompt_content:
            instructions = signature_metadata.get('instructions', '')
            if len(instructions) > 50:  # Substantial instructions
                validation_result['compliance_checks']['separation_of_concerns'] = True
                validation_result['recommendations'].append(
                    "✓ Clear separation: instructions in metadata, logic in signature"
                )
            else:
                validation_result['recommendations'].append(
                    "⚠ Consider moving more instructional content to signature metadata"
                )
                validation_result['improvement_priority'].append("Improve separation of concerns")
        
        # Calculate overall score
        total_checks = len(validation_result['compliance_checks'])
        passed_checks = sum(validation_result['compliance_checks'].values())
        validation_result['overall_score'] = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # Generate priority recommendations based on score
        if validation_result['overall_score'] >= 0.8:
            validation_result['recommendations'].append(
                "🎉 Excellent DSPy 2024-2025 compliance! Ready for production use."
            )
        elif validation_result['overall_score'] >= 0.6:
            validation_result['recommendations'].append(
                "👍 Good DSPy compliance with room for enhancement."
            )
        else:
            validation_result['recommendations'].append(
                "🔧 Significant improvements needed for DSPy 2024-2025 best practices."
            )
        
    except Exception as e:
        validation_result['recommendations'].append(f"Validation error: {e}")
        logger.error(f"DSPy compliance validation failed: {e}")
    
    return validation_result