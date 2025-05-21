"""
LLM (Language Model) module that provides a unified interface for different LLM providers.
This module includes both the configuration and implementation of LLM functionality.
"""

from typing import Optional, Union, Dict, Any, Type, Literal, List
from dataclasses import dataclass
from pydantic import BaseModel
from abc import ABC, abstractmethod
import time
import os
from openai import OpenAI
from google import genai
from google.genai.types import HttpOptions
from .exceptions import (
    LLMError,
    LLMConfigError,
    LLMAPIError,
    LLMValidationError,
    LLMRetryError,
    LLMProviderError
)
from .logger import setup_logger

# Provider constants
class Provider:
    """Constants for supported LLM providers"""
    OPENAI = "openai"
    GEMINI = "gemini"

# Model configurations
OPENAI_MODELS = {
    "default": "gpt-4.1",
    "pro": "gpt-4o",
    "mini": "gpt-4.1-mini",
    "nano": "gpt-4.1-nano"
}

GEMINI_MODELS = {
    "default": "gemini-2.5-flash-preview-04-17",
    "pro": "gemini-2.5-flash-preview-04-17",
    "mini": "gemini-2.5-flash-preview-04-17",
    "nano": "gemini-2.0-flash-lite"
}

@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider"""
    name: str
    models: Dict[str, str]
    default_model: str
    env_var: str
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: int = 5

# Provider configurations
PROVIDERS = {
    Provider.OPENAI: ProviderConfig(
        name=Provider.OPENAI,
        models=OPENAI_MODELS,
        default_model=OPENAI_MODELS["default"],
        env_var="OPENAI_API_KEY"
    ),
    Provider.GEMINI: ProviderConfig(
        name=Provider.GEMINI,
        models=GEMINI_MODELS,
        default_model=GEMINI_MODELS["default"],
        env_var="GOOGLE_API_KEY"
    )
}

def get_provider_config(provider: str) -> ProviderConfig:
    """
    Get configuration for a specific provider
    
    Args:
        provider: Provider name (e.g., "openai" or "gemini")
        
    Returns:
        ProviderConfig for the specified provider
        
    Raises:
        ValueError: If provider is not supported
    """
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDERS[provider]

def get_api_key(provider: str) -> str:
    """
    Get API key for a specific provider
    
    Args:
        provider: Provider name
        
    Returns:
        API key from environment variables
        
    Raises:
        ValueError: If API key is not set
    """
    config = get_provider_config(provider)
    api_key = os.environ.get(config.env_var)
    if not api_key:
        raise ValueError(f"{config.env_var} environment variable not set")
    return api_key

def get_available_models(provider: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get available models for specified provider or all providers
    
    Args:
        provider: Optional provider name to filter models
        
    Returns:
        Dictionary of provider -> list of model names
    """
    if provider:
        config = get_provider_config(provider)
        return {config.name: list(config.models.values())}
    
    return {name: list(config.models.values()) 
            for name, config in PROVIDERS.items()}

@dataclass
class InferenceResult:
    """Class to hold inference results"""
    text: Any
    raw: Dict[str, Any]

class ModelProvider:
    """Enum for supported model providers"""
    OPENAI = "openai"
    GEMINI = "gemini"

class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        # print(f"Initializing {self.__class__.__name__} with model: {model}")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = setup_logger(f"llm.{self.__class__.__name__}")
        self._setup_client()

    @abstractmethod
    def _setup_client(self):
        """Initialize the API client"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider"""
        pass

    def _handle_api_error(self, error: Exception, retries: int) -> None:
        """Handle API errors with proper logging and retries"""
        self.logger.warning(
            f"API call failed (attempt {retries + 1}/{self.max_retries}). Error: {str(error)}"
        )
        
        if retries >= self.max_retries:
            self.logger.error(
                f"Maximum retry attempts reached. Last error: {str(error)}"
            )
            raise LLMRetryError(
                message=f"Failed after {self.max_retries} retries",
                attempts=retries,
                last_error=error
            )
            
        self.logger.info(
            f"Retrying in {self.retry_delay} seconds"
        )
        time.sleep(self.retry_delay)

    @abstractmethod
    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> InferenceResult:
        """Run inference with structured output support"""
        pass

class ChatGPT(BaseLLM):
    """ChatGPT implementation using OpenAI's API with structured output support"""
    
    def get_provider_name(self) -> str:
        return Provider.OPENAI
    
    def _setup_client(self):
        """Initialize the OpenAI client"""
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise LLMConfigError(f"Failed to initialize OpenAI client: {str(e)}")

    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> InferenceResult:
        """
        Run inference with structured output support
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Override default temperature
            response_model: Pydantic model for structured output validation
            
        Returns:
            InferenceResult containing the parsed response and raw API response
            
        Raises:
            LLMValidationError: If input validation fails
            LLMAPIError: If API call fails
            LLMRetryError: If maximum retries are exhausted
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            temp = temperature if temperature is not None else self.temperature

            retries = 0
            while True:
                try:
                    if response_model:
                        response = self.client.beta.chat.completions.parse(
                            model=self.model,
                            messages=messages,
                            response_format=response_model,
                            temperature=temp,
                            max_tokens=max_tokens if max_tokens else None,
                        )
                        parsed_content = response.choices[0].message.parsed
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=temp,
                            max_tokens=max_tokens if max_tokens else None,
                        )
                        parsed_content = response.choices[0].message.content
                    
                    return InferenceResult(
                        text=parsed_content,
                        raw=response.model_dump()
                    )
                    
                except Exception as e:
                    self._handle_api_error(e, retries)
                    retries += 1
                    
        except LLMError:
            raise
        except Exception as e:
            self.logger.exception("Unexpected error during inference")
            raise LLMAPIError(
                message="Unexpected error during inference",
                details={"error": str(e)}
            )

class Gemini(BaseLLM):
    """Gemini implementation using Google's Gemini API with structured output support"""
    
    def get_provider_name(self) -> str:
        return Provider.GEMINI
    
    def _setup_client(self):
        """Initialize the Gemini client"""
        try:
            # verify environment variables
            # export GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT
            # export GOOGLE_CLOUD_LOCATION=us-central1
            # export GOOGLE_GENAI_USE_VERTEXAI=True
            assert os.environ.get("GOOGLE_CLOUD_PROJECT"), "GOOGLE_CLOUD_PROJECT environment variable not set"
            assert os.environ.get("GOOGLE_CLOUD_LOCATION"), "GOOGLE_CLOUD_LOCATION environment variable not set"
            assert os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"), "GOOGLE_GENAI_USE_VERTEXAI environment variable not set"
            self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
            self.model_name = self.model
        except Exception as e:
            raise LLMConfigError(f"Failed to initialize Gemini client: {str(e)}")

    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response by removing code block markers and extra whitespace"""
        # Remove code block markers if present
        response = response.replace('```json\n', '').replace('\n```', '')
        response = response.replace('```JSON\n', '').replace('\n```', '')
        response = response.replace('```\n', '').replace('\n```', '')
        # Remove any leading/trailing whitespace
        response = response.strip()
        
        # Attempt to fix common JSON errors
        return self._fix_json_errors(response)
    
    def _fix_json_errors(self, json_str: str) -> str:
        """
        Attempt to fix common JSON errors that might occur in LLM responses
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Fixed JSON string (best effort)
        """
        # Check for unclosed quotes in the JSON
        try:
            import json
            json.loads(json_str)
            return json_str  # If it parses correctly, return as is
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON decode error: {str(e)}")
            
            # Fix for unclosed quotes
            if "Unterminated string" in str(e) or "EOF while parsing a string" in str(e):
                # Enhanced fix for unterminated strings - more aggressive
                try:
                    # Get position of the error
                    pos = e.pos
                    if pos < len(json_str):
                        # First attempt: add a quote at the exact error position
                        fixed_json = json_str[:pos] + '"' + json_str[pos:]
                        try:
                            json.loads(fixed_json)
                            self.logger.info("Successfully fixed unclosed quote at exact position")
                            return fixed_json
                        except:
                            pass  # If still fails, try the next approach
                        
                        # Second attempt: Check if we can find the end of the unterminated string (next line or certain characters)
                        next_line_pos = json_str.find('\n', pos)
                        next_comma_pos = json_str.find(',', pos)
                        next_brace_pos = json_str.find('}', pos)
                        
                        # Find the closest terminator
                        terminator_positions = [p for p in [next_line_pos, next_comma_pos, next_brace_pos] if p > 0]
                        if terminator_positions:
                            fix_pos = min(terminator_positions)
                            fixed_json = json_str[:fix_pos] + '"' + json_str[fix_pos:]
                            try:
                                json.loads(fixed_json)
                                self.logger.info(f"Successfully fixed unclosed quote at terminator position {fix_pos}")
                                return fixed_json
                            except:
                                pass  # Continue with other fixes
                                
                        # Third attempt: for very large strings, truncate the problematic string where it occurs
                        # Find the last opening quote before the error position
                        last_quote_pos = json_str.rfind('"', 0, pos)
                        if last_quote_pos >= 0:
                            # Add closing quote right after the error position
                            # and check if we need to add a comma or other syntax
                            fixed_json = json_str[:pos] + '"'
                            
                            # Check what should come after the closed string
                            if next_comma_pos > 0 and (next_brace_pos < 0 or next_comma_pos < next_brace_pos):
                                # If a comma is closer than a closing brace, keep the comma
                                fixed_json += json_str[pos:next_comma_pos+1]
                                if next_brace_pos > 0:
                                    fixed_json += json_str[next_comma_pos+1:]
                            elif next_brace_pos > 0:
                                # If a closing brace is next, keep it
                                fixed_json += json_str[pos:next_brace_pos+1]
                                if json_str[pos:].find('}', next_brace_pos+1) > 0:
                                    # Try to preserve the rest of the JSON
                                    fixed_json += json_str[next_brace_pos+1:]
                            
                            try:
                                json.loads(fixed_json)
                                self.logger.info("Successfully fixed unterminated string with structural correction")
                                return fixed_json
                            except:
                                pass  # Continue with other fixes
                except Exception as fix_error:
                    self.logger.warning(f"Advanced quote fixing failed: {str(fix_error)}")
                
                # Fourth attempt: complete reconstruction for heavily malformed JSON
                import re
                
                # Extract the first part of the JSON which might be mostly valid
                try:
                    # Define a comprehensive pattern to match the correct structure
                    pattern = r'^\s*\{\s*"extractions"\s*:\s*\[\s*\{.*?'
                    
                    # Try to find a valid beginning
                    match = re.search(pattern, json_str, re.DOTALL)
                    
                    if match:
                        # Get just the valid beginning and reconstruct minimal valid JSON
                        valid_start = match.group(0)
                        
                        # Extract department from the original string if available
                        dept_match = re.search(r'"department"\s*:\s*"([^"]*)"', json_str)
                        department = dept_match.group(1) if dept_match else ""
                        
                        # Extract research_area or source from the original string if available
                        area_match = re.search(r'"research_area"\s*:\s*"([^"]*)"', json_str) or re.search(r'"source"\s*:\s*"([^"]*)"', json_str)
                        area = area_match.group(1) if area_match else ""
                        
                        # Extract any concepts that are properly formatted
                        concepts = []
                        concept_matches = re.findall(r'"([^"]+)"', json_str[json_str.find('"concepts"'):json_str.find(']', json_str.find('"concepts"'))])
                        if concept_matches:
                            concepts = [c for c in concept_matches if c != "concepts"]
                        
                        # Create a new, clean JSON structure
                        min_json = {
                            "extractions": [
                                {
                                    "source": area,
                                    "concepts": concepts,
                                    "department": department if "department" in json_str else "",
                                    "research_area": area if "research_area" in json_str else ""
                                }
                            ]
                        }
                        
                        # Clean up the structure by removing empty fields
                        if not min_json["extractions"][0]["department"]:
                            del min_json["extractions"][0]["department"]
                        if not min_json["extractions"][0]["research_area"]:
                            del min_json["extractions"][0]["research_area"]
                        
                        fixed_json = json.dumps(min_json)
                        try:
                            json.loads(fixed_json)  # Test if valid
                            self.logger.info("Successfully reconstructed JSON from extracted data")
                            return fixed_json
                        except Exception as e:
                            self.logger.warning(f"Reconstruction validation failed: {str(e)}")
                except Exception as extract_error:
                    self.logger.warning(f"JSON extraction reconstruction failed: {str(extract_error)}")
            
            # Fix for trailing commas in arrays/objects
            if "Expecting ',' delimiter" in str(e) or "Expecting property name" in str(e):
                # Try removing trailing commas
                fixed_json = json_str.replace(",]", "]").replace(",}", "}")
                try:
                    json.loads(fixed_json)
                    self.logger.info("Successfully fixed trailing comma in JSON")
                    return fixed_json
                except:
                    pass
            
            # Fix for missing commas between properties
            if "Expecting ',' delimiter" in str(e):
                try:
                    pos = e.pos
                    # Check if we need to add a comma (if pos is at the start of a new property)
                    # Look backward for a closing quote or brace, and forward for an opening quote
                    before_pos = json_str[:pos].rstrip()
                    after_pos = json_str[pos:].lstrip()
                    
                    if (before_pos.endswith('"') or before_pos.endswith('}') or before_pos.endswith(']')) and \
                       (after_pos.startswith('"') or after_pos.startswith('{')):
                        # Add missing comma
                        fixed_json = json_str[:pos] + "," + json_str[pos:]
                        try:
                            json.loads(fixed_json)
                            self.logger.info("Successfully added missing comma in JSON")
                            return fixed_json
                        except:
                            pass
                except Exception as comma_fix_error:
                    self.logger.warning(f"Comma fixing failed: {str(comma_fix_error)}")
            
            # If we have a severely damaged JSON, try to build a minimal valid structure
            try:
                # Look for any concepts that might be valid
                import re
                concepts = re.findall(r'"([^"]+)"', json_str)
                
                # Remove common field names and keep only what might be concepts
                common_fields = ["source", "concepts", "department", "research_area", "extractions"]
                potential_concepts = [c for c in concepts if c not in common_fields]
                
                if potential_concepts:
                    # Build minimal valid JSON
                    min_json = {
                        "extractions": [
                            {
                                "source": "recovered_source",
                                "concepts": potential_concepts[:5],  # Take first 5 potential concepts
                                "department": "" if "department" in json_str else None,
                                "research_area": "" if "research_area" in json_str else None
                            }
                        ]
                    }
                    
                    # Clean up null fields
                    extraction = min_json["extractions"][0]
                    if extraction["department"] is None:
                        del extraction["department"]
                    if extraction["research_area"] is None:
                        del extraction["research_area"]
                    
                    fixed_json = json.dumps(min_json)
                    self.logger.info("Created minimal valid JSON structure from fragments")
                    return fixed_json
            except Exception as min_json_error:
                self.logger.warning(f"Minimal JSON creation failed: {str(min_json_error)}")
            
            # If all fixes fail, log the issue and return the original
            self.logger.error(f"Failed to fix JSON: {json_str[:100]}...")
            return json_str

    def infer(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> InferenceResult:
        """
        Run inference with structured output support
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Override default temperature
            response_model: Pydantic model for structured output validation
            
        Returns:
            InferenceResult containing the parsed response and raw API response
            
        Raises:
            LLMValidationError: If input validation fails
            LLMAPIError: If API call fails
            LLMRetryError: If maximum retries are exhausted
        """
        try:
            # Build the content with system prompt if provided
            content = prompt
            if system_prompt:
                content = f"{system_prompt}\n\n{prompt}"

            temp = temperature if temperature is not None else self.temperature
            
            retries = 0
            while True:
                try:
                    generation_config = {
                        'temperature': temp,
                        'max_output_tokens': max_tokens if max_tokens else None,
                    }

                    if response_model:
                        # Add JSON schema configuration for structured output
                        generation_config.update({
                            'response_mime_type': 'application/json',
                            'response_schema': response_model,
                        })

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=content,
                        config=generation_config
                    )

                    # Handle structured output
                    if response_model:
                        # For structured output, clean and parse JSON
                        text_response = response.text
                        cleaned_json = self._clean_json_response(text_response)
                        
                        try:
                            parsed_content = response_model.model_validate_json(cleaned_json)
                        except Exception as json_error:
                            self.logger.error(f"Failed to validate JSON against model: {str(json_error)}")
                            self.logger.debug(f"Problematic JSON: {cleaned_json[:500]}...")
                            
                            # Try a more aggressive JSON fix approach
                            import json
                            import re
                            
                            # Try to extract valid JSON using regex pattern matching
                            # Look for the expected structure based on the model
                            if hasattr(response_model, 'model_json_schema'):
                                schema = response_model.model_json_schema()
                                if 'properties' in schema:
                                    # Get the top-level properties to look for
                                    properties = list(schema['properties'].keys())
                                    self.logger.info(f"Looking for properties: {properties}")
                                    
                                    # Try to find a JSON object with these properties
                                    # Use a non-recursive approach to find balanced braces
                                    def find_balanced_json(text):
                                        """Find balanced JSON objects in text"""
                                        results = []
                                        stack = []
                                        start_indices = []
                                        
                                        for i, char in enumerate(text):
                                            if char == '{':
                                                if not stack:  # Start of a new object
                                                    start_indices.append(i)
                                                stack.append('{')
                                            elif char == '}':
                                                if stack and stack[-1] == '{':
                                                    stack.pop()
                                                    if not stack:  # End of an object
                                                        start = start_indices.pop()
                                                        results.append(text[start:i+1])
                                        
                                        return results
                                    
                                    matches = find_balanced_json(cleaned_json)
                                    
                                    for match in matches:
                                        try:
                                            obj = json.loads(match)
                                            # Check if it has the expected properties
                                            if all(prop in obj for prop in properties):
                                                self.logger.info("Found valid JSON object with expected properties")
                                                parsed_content = response_model.model_validate(obj)
                                                break
                                        except:
                                            continue
                            
                            # If we still don't have valid content, try to reconstruct it
                            if 'parsed_content' not in locals():
                                self.logger.warning("Attempting to reconstruct JSON from LLM response")
                                # Make another call to the LLM to fix the JSON
                                fix_prompt = f"""The following JSON is malformed. Please fix it and return ONLY the corrected JSON:

{cleaned_json}"""
                                try:
                                    fix_response = self.client.models.generate_content(
                                        model=self.model_name,
                                        contents=fix_prompt,
                                        config={'temperature': 0.1}  # Low temperature for deterministic response
                                    )
                                    fixed_json = self._clean_json_response(fix_response.text)
                                    parsed_content = response_model.model_validate_json(fixed_json)
                                    self.logger.info("Successfully reconstructed JSON using LLM")
                                except Exception as fix_error:
                                    # If all attempts fail, raise the original error
                                    self.logger.error(f"Failed to reconstruct JSON: {str(fix_error)}")
                                    raise LLMValidationError(
                                        message="Failed to validate response against model",
                                        details={"error": str(json_error), "response": cleaned_json[:500]}
                                    )
                    else:
                        # For regular output, just use the text response
                        parsed_content = response.text
                    
                    return InferenceResult(
                        text=parsed_content,
                        raw={"response": response.text}
                    )
                    
                except Exception as e:
                    self._handle_api_error(e, retries)
                    retries += 1
                    
        except LLMError:
            raise
        except Exception as e:
            self.logger.exception("Unexpected error during inference")
            raise LLMAPIError(
                message="Unexpected error during inference",
                details={"error": str(e)}
            )

    async def infer_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> InferenceResult:
        """
        Async version of infer method. For Gemini, we'll run the synchronous version
        in an executor since the Gemini API doesn't have native async support.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.infer(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                response_model=response_model
            )
        )

class LLMFactory:
    """Factory class for creating LLM instances"""
    
    @staticmethod
    def create_llm(
        provider: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None
    ) -> BaseLLM:
        """
        Create an LLM instance based on provider and model
        
        Args:
            provider: The LLM provider (openai or gemini)
            model: Optional model name, will use default if not specified
            temperature: Model temperature, uses provider default if not specified
            max_retries: Maximum number of retries, uses provider default if not specified
            retry_delay: Delay between retries in seconds, uses provider default if not specified
            
        Returns:
            An instance of BaseLLM implementation
            
        Raises:
            LLMConfigError: If configuration is invalid
            LLMProviderError: If provider is not supported
        """
        try:
            if not isinstance(provider, str):
                raise LLMProviderError(
                    message=f"Unsupported provider: {provider}",
                    provider=str(provider)
                )
            
            provider = provider.lower()
            if provider not in [Provider.OPENAI, Provider.GEMINI]:
                raise LLMProviderError(
                    message=f"Unsupported provider: {provider}",
                    provider=provider
                )
            
            config = get_provider_config(provider)
            api_key = get_api_key(provider)
            
            # Use provider defaults if not specified
            model_name = model or config.default_model
            temp = temperature if temperature is not None else config.temperature
            retries = max_retries if max_retries is not None else config.max_retries
            delay = retry_delay if retry_delay is not None else config.retry_delay
            
            if provider == Provider.OPENAI:
                return ChatGPT(
                    api_key=api_key,
                    model=model_name,
                    temperature=temp,
                    max_retries=retries,
                    retry_delay=delay
                )
                
            elif provider == Provider.GEMINI:
                return Gemini(
                    api_key=api_key,
                    model=model_name,
                    temperature=temp,
                    max_retries=retries,
                    retry_delay=delay
                )
                
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMConfigError(
                message=f"Failed to create LLM instance: {str(e)}",
                details={"provider": provider, "model": model}
            )

    @staticmethod
    def get_available_models(provider: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get available models for specified provider or all providers
        
        Args:
            provider: Optional provider name to filter models
            
        Returns:
            Dictionary of provider -> list of model names
            
        Raises:
            LLMProviderError: If provider is not supported
        """
        try:
            return get_available_models(provider)
        except ValueError as e:
            raise LLMProviderError(
                message=str(e),
                provider=provider if provider else "all"
            )

# Module-level function to get an LLM instance
def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseLLM:
    """
    Create an LLM instance with the specified provider and model
    
    Args:
        provider: Optional provider name (defaults to "gemini")
        model: Optional model name or tier ("default", "pro", "mini", etc.)
        
    Returns:
        LLM instance ready for inference
        
    Raises:
        LLMConfigError: If configuration is invalid
        LLMProviderError: If provider is not supported
    """
    if not provider:
        provider = Provider.GEMINI  # Default to Gemini
    
    # If model is a tier name (default, pro, mini), get the actual model name
    model_dict = GEMINI_MODELS if provider == Provider.GEMINI else OPENAI_MODELS
    selected_model = model_dict.get(model, None) if model else None
    
    return LLMFactory.create_llm(
        provider=provider,
        model=selected_model,
        temperature=0.7  # Default temperature
    )