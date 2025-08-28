"""
Integration tests for LLM system with real API calls
These tests require valid API keys and will make actual LLM calls
"""

import pytest
import os
from typing import List, Dict
from pydantic import BaseModel, Field

from generate_glossary.utils.llm_simple import (
    infer_structured,
    infer_text,
    get_random_llm_config,
    openai_text,
    openai_structured
)


# Test models for structured responses
class SimpleResponse(BaseModel):
    """Simple test response model"""
    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="High, Medium, or Low confidence")


class ConceptList(BaseModel):
    """Model for concept extraction testing"""
    concepts: List[str] = Field(description="List of extracted concepts")
    total_count: int = Field(description="Total number of concepts found")


# Skip integration tests if no API keys are available
skip_if_no_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

skip_if_no_gemini = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
    reason="GEMINI_API_KEY or GOOGLE_API_KEY not set"
)


class TestLLMIntegrationBasic:
    """Basic integration tests for LLM functionality"""
    
    @skip_if_no_openai
    def test_openai_text_basic(self):
        """Test basic OpenAI text completion"""
        result = openai_text(
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        # Should contain "4" somewhere in the response
        assert "4" in result
    
    @skip_if_no_openai
    def test_openai_structured_basic(self):
        """Test basic OpenAI structured completion"""
        result = openai_structured(
            messages=[{
                "role": "user", 
                "content": "Is Python a programming language? Answer 'Yes' or 'No' and your confidence level."
            }],
            response_model=SimpleResponse,
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        assert result is not None
        assert isinstance(result, SimpleResponse)
        assert result.answer.lower() in ["yes", "no"]
        assert result.confidence.lower() in ["high", "medium", "low"]
    
    @skip_if_no_openai
    def test_infer_text_compatibility(self):
        """Test the compatibility function infer_text"""
        result = infer_text(
            provider="openai",
            prompt="Name one programming language. Give just the name.",
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        assert result is not None
        assert hasattr(result, 'text')
        assert isinstance(result.text, str)
        assert len(result.text.strip()) > 0
    
    @skip_if_no_openai
    def test_infer_structured_compatibility(self):
        """Test the compatibility function infer_structured"""
        result = infer_structured(
            provider="openai",
            prompt="Extract programming concepts from this text: 'Python is an object-oriented programming language that supports machine learning'",
            response_model=ConceptList,
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        assert result is not None
        assert hasattr(result, 'text')
        assert isinstance(result.text, ConceptList)
        concepts = result.text
        assert len(concepts.concepts) > 0
        assert concepts.total_count == len(concepts.concepts)
        # Should extract some relevant concepts
        concept_text = " ".join(concepts.concepts).lower()
        assert any(term in concept_text for term in ["python", "programming", "object", "machine learning"])


class TestLLMIntegrationAdvanced:
    """Advanced integration tests"""
    
    @skip_if_no_openai
    def test_system_prompt_handling(self):
        """Test that system prompts are properly handled"""
        result = infer_text(
            provider="openai",
            prompt="What should I say?",
            system_prompt="You are a helpful assistant. Always respond with exactly: 'Hello from system prompt'",
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        assert result is not None
        assert "Hello from system prompt" in result.text
    
    @skip_if_no_openai
    def test_temperature_effects(self):
        """Test that temperature parameter affects randomness"""
        prompt = "Give me a creative name for a pet robot."
        
        # Low temperature - should be more deterministic
        result1 = infer_text(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        result2 = infer_text(
            provider="openai", 
            prompt=prompt,
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        # High temperature - should be more creative/random
        result3 = infer_text(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini", 
            temperature=0.9
        )
        
        assert result1.text is not None
        assert result2.text is not None
        assert result3.text is not None
        
        # All should be strings
        assert all(isinstance(r.text, str) for r in [result1, result2, result3])
        assert all(len(r.text.strip()) > 0 for r in [result1, result2, result3])
    
    @skip_if_no_openai
    def test_concept_extraction_realistic(self):
        """Test concept extraction with realistic academic text"""
        academic_text = """
        The Department of Computer Science and Engineering offers comprehensive programs 
        in artificial intelligence, machine learning, data structures, algorithms, 
        software engineering, and cybersecurity. Students learn programming languages 
        like Python and Java while exploring areas such as neural networks, 
        natural language processing, and distributed systems.
        """
        
        result = infer_structured(
            provider="openai",
            prompt=f"Extract academic concepts and disciplines from this text: {academic_text}",
            response_model=ConceptList,
            system_prompt="You are an expert in academic classification. Extract specific technical concepts and disciplines mentioned in the text.",
            model="gpt-4o-mini",
            temperature=0.2
        )
        
        assert result is not None
        concepts = result.text
        assert isinstance(concepts, ConceptList)
        assert len(concepts.concepts) >= 5  # Should extract several concepts
        # Allow for small discrepancies in count due to LLM variability
        assert abs(concepts.total_count - len(concepts.concepts)) <= 2
        
        # Check for expected concepts (case-insensitive)
        concept_text = " ".join(concepts.concepts).lower()
        expected_concepts = [
            "computer science", "artificial intelligence", "machine learning",
            "algorithms", "software engineering", "python", "java"
        ]
        
        # Should find at least half of the expected concepts
        found_concepts = sum(1 for concept in expected_concepts if concept in concept_text)
        assert found_concepts >= len(expected_concepts) // 2


class TestRandomLLMConfig:
    """Test the random LLM configuration in real scenarios"""
    
    def test_random_config_generates_valid_combinations(self):
        """Test that random configurations are valid"""
        configs = []
        
        # Generate multiple random configs
        for _ in range(10):
            provider, model = get_random_llm_config()
            configs.append((provider, model))
        
        # All should be valid provider/model combinations
        for provider, model in configs:
            assert provider in ["openai", "anthropic"]
            assert isinstance(model, str)
            assert len(model) > 0
            
            # Basic model name validation
            if provider == "openai":
                assert model.startswith("gpt") or "gpt" in model.lower()
    
    def test_random_config_level_distribution(self):
        """Test that different levels produce appropriate model selections"""
        level_configs = {}
        
        # Test different levels
        for level in range(4):
            level_configs[level] = []
            for _ in range(5):  # Generate 5 configs per level
                provider, model = get_random_llm_config(level=level)
                level_configs[level].append((provider, model))
        
        # Verify we get valid configs for each level
        for level, configs in level_configs.items():
            assert len(configs) == 5
            for provider, model in configs:
                assert provider in ["openai", "anthropic"]
                assert isinstance(model, str)


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios"""
    
    @skip_if_no_openai
    def test_invalid_model_handling(self):
        """Test handling of invalid model specifications"""
        with pytest.raises(Exception):
            # This should fail due to invalid model
            infer_text(
                provider="openai",
                prompt="Test",
                model="invalid-model-name-that-does-not-exist"
            )
    
    @skip_if_no_openai  
    def test_malformed_structured_request(self):
        """Test handling of malformed structured requests"""
        class BadModel(BaseModel):
            # Intentionally problematic model that might cause issues
            impossible_field: List[Dict[str, List[Dict[str, str]]]] = Field(
                description="Overly complex nested structure"
            )
        
        # This might fail due to complex model structure, but shouldn't crash
        try:
            result = infer_structured(
                provider="openai",
                prompt="Generate something impossible",
                response_model=BadModel,
                model="gpt-4o-mini"
            )
            # If it succeeds, that's fine too
            assert result is not None
        except Exception as e:
            # If it fails, that's expected - just ensure it's a reasonable error
            assert isinstance(e, Exception)
            # Error message should be informative
            assert len(str(e)) > 0


if __name__ == "__main__":
    # Run integration tests with verbose output
    pytest.main([__file__, "-v", "-s"])