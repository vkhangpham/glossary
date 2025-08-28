"""
Unit tests for the simplified LLM system using LiteLLM + Instructor
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from generate_glossary.utils.llm_simple import (
    get_llm_client,
    structured_completion,
    text_completion, 
    openai_structured,
    openai_text,
    anthropic_structured,
    anthropic_text,
    infer_structured,
    infer_text,
    get_random_llm_config,
    LLMResult
)


# Test Pydantic models
class MockResponse(BaseModel):
    """Test model for structured responses"""
    content: str = Field(description="Test content")
    confidence: float = Field(description="Confidence score")


class MockConceptList(BaseModel):
    """Test model for concept extraction"""
    concepts: List[str] = Field(description="List of concepts")
    count: int = Field(description="Number of concepts")


class TestLLMResult:
    """Test the LLMResult wrapper class"""
    
    def test_llm_result_creation(self):
        """Test creating an LLMResult object"""
        content = "Test content"
        result = LLMResult(content)
        assert result.text == content
    
    def test_llm_result_with_structured_content(self):
        """Test LLMResult with structured content"""
        structured_content = MockResponse(content="test", confidence=0.9)
        result = LLMResult(structured_content)
        assert result.text == structured_content


class TestLLMClient:
    """Test the LLM client functionality"""
    
    def test_get_llm_client(self):
        """Test that we can get an LLM client"""
        client = get_llm_client()
        assert client is not None
        # Should be an instructor client
        assert hasattr(client, 'chat')


class TestRandomLLMConfig:
    """Test the random LLM configuration function"""
    
    def test_get_random_config_returns_tuple(self):
        """Test that get_random_llm_config returns a tuple"""
        provider, model = get_random_llm_config()
        assert isinstance(provider, str)
        assert isinstance(model, str)
        assert provider in ["openai", "anthropic"]
    
    def test_get_random_config_level_0(self):
        """Test configuration for level 0 (college level)"""
        provider, model = get_random_llm_config(level=0)
        assert provider in ["openai", "anthropic"]
        if provider == "openai":
            assert model in ["gpt-4o-mini", "gpt-4"]
    
    def test_get_random_config_level_1(self):
        """Test configuration for level 1 (department level)"""
        provider, model = get_random_llm_config(level=1)
        assert provider in ["openai", "anthropic"]
        if provider == "openai":
            assert model in ["gpt-4", "gpt-4o-mini"]
    
    def test_get_random_config_level_2(self):
        """Test configuration for level 2 (research areas)"""
        provider, model = get_random_llm_config(level=2)
        assert provider in ["openai", "anthropic"]
        if provider == "openai":
            assert model in ["gpt-4", "gpt-4o"]
    
    def test_get_random_config_level_3(self):
        """Test configuration for level 3 (conference topics)"""
        provider, model = get_random_llm_config(level=3)
        assert provider in ["openai", "anthropic"]
        if provider == "openai":
            assert model in ["gpt-4o-mini", "gpt-4o"]
    
    def test_get_random_config_invalid_level(self):
        """Test configuration for invalid level defaults to gpt-4"""
        provider, model = get_random_llm_config(level=999)
        assert provider in ["openai", "anthropic"]
        if provider == "openai":
            assert model == "gpt-4"


class TestStructuredCompletion:
    """Test structured completion functionality"""
    
    @patch('generate_glossary.utils.llm_simple.client.chat.completions.create')
    def test_structured_completion_success(self, mock_create):
        """Test successful structured completion"""
        # Mock response
        mock_response = MockResponse(content="test response", confidence=0.95)
        mock_create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test prompt"}]
        result = structured_completion(
            model="openai/gpt-4",
            messages=messages,
            response_model=MockResponse
        )
        
        assert result == mock_response
        mock_create.assert_called_once_with(
            model="openai/gpt-4",
            messages=messages,
            response_model=MockResponse,
            temperature=0.7,
            max_tokens=None,
            max_retries=3
        )
    
    @patch('generate_glossary.utils.llm_simple.client.chat.completions.create')
    def test_structured_completion_with_params(self, mock_create):
        """Test structured completion with custom parameters"""
        mock_response = MockResponse(content="test", confidence=0.8)
        mock_create.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test"}]
        result = structured_completion(
            model="anthropic/claude-3-sonnet-20240229",
            messages=messages,
            response_model=MockResponse,
            temperature=0.3,
            max_tokens=1000,
            max_retries=5
        )
        
        assert result == mock_response
        mock_create.assert_called_once_with(
            model="anthropic/claude-3-sonnet-20240229",
            messages=messages,
            response_model=MockResponse,
            temperature=0.3,
            max_tokens=1000,
            max_retries=5
        )


class TestTextCompletion:
    """Test text completion functionality"""
    
    @patch('generate_glossary.utils.llm_simple.completion')
    def test_text_completion_success(self, mock_completion):
        """Test successful text completion"""
        # Mock response structure
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response content"
        mock_completion.return_value = mock_response
        
        messages = [{"role": "user", "content": "Test prompt"}]
        result = text_completion(
            model="openai/gpt-4",
            messages=messages
        )
        
        assert result == "Test response content"
        mock_completion.assert_called_once_with(
            model="openai/gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=None
        )
    
    @patch('generate_glossary.utils.llm_simple.completion')
    def test_text_completion_with_params(self, mock_completion):
        """Test text completion with custom parameters"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Custom response"
        mock_completion.return_value = mock_response
        
        messages = [{"role": "system", "content": "You are helpful"}, 
                   {"role": "user", "content": "Test"}]
        result = text_completion(
            model="anthropic/claude-3-sonnet-20240229",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        
        assert result == "Custom response"
        mock_completion.assert_called_once_with(
            model="anthropic/claude-3-sonnet-20240229",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )


class TestConvenienceFunctions:
    """Test provider-specific convenience functions"""
    
    @patch('generate_glossary.utils.llm_simple.structured_completion')
    def test_openai_structured(self, mock_structured):
        """Test OpenAI structured completion convenience function"""
        mock_response = MockResponse(content="openai test", confidence=0.9)
        mock_structured.return_value = mock_response
        
        messages = [{"role": "user", "content": "test"}]
        result = openai_structured(messages, MockResponse, model="gpt-4o")
        
        assert result == mock_response
        mock_structured.assert_called_once_with(
            "openai/gpt-4o", messages, MockResponse
        )
    
    @patch('generate_glossary.utils.llm_simple.text_completion')
    def test_openai_text(self, mock_text):
        """Test OpenAI text completion convenience function"""
        mock_text.return_value = "OpenAI response"
        
        messages = [{"role": "user", "content": "test"}]
        result = openai_text(messages, model="gpt-4o")
        
        assert result == "OpenAI response"
        mock_text.assert_called_once_with("openai/gpt-4o", messages)
    
    @patch('generate_glossary.utils.llm_simple.structured_completion')
    def test_anthropic_structured(self, mock_structured):
        """Test Anthropic structured completion convenience function"""
        mock_response = MockResponse(content="anthropic test", confidence=0.85)
        mock_structured.return_value = mock_response
        
        messages = [{"role": "user", "content": "test"}]
        result = anthropic_structured(messages, MockResponse)
        
        assert result == mock_response
        mock_structured.assert_called_once_with(
            "anthropic/claude-3-sonnet-20240229", messages, MockResponse
        )
    
    @patch('generate_glossary.utils.llm_simple.text_completion')
    def test_anthropic_text(self, mock_text):
        """Test Anthropic text completion convenience function"""
        mock_text.return_value = "Anthropic response"
        
        messages = [{"role": "user", "content": "test"}]
        result = anthropic_text(messages)
        
        assert result == "Anthropic response"
        mock_text.assert_called_once_with(
            "anthropic/claude-3-sonnet-20240229", messages
        )


class TestCompatibilityFunctions:
    """Test migration compatibility functions"""
    
    @patch('generate_glossary.utils.llm_simple.structured_completion')
    def test_infer_structured_openai(self, mock_structured):
        """Test infer_structured compatibility function with OpenAI"""
        mock_response = MockResponse(content="structured test", confidence=0.9)
        mock_structured.return_value = mock_response
        
        result = infer_structured(
            provider="openai",
            prompt="Extract concepts from this text",
            response_model=MockResponse,
            system_prompt="You are a concept extractor",
            model="gpt-4o",
            temperature=0.3
        )
        
        assert isinstance(result, LLMResult)
        assert result.text == mock_response
        
        expected_messages = [
            {"role": "system", "content": "You are a concept extractor"},
            {"role": "user", "content": "Extract concepts from this text"}
        ]
        mock_structured.assert_called_once_with(
            model="openai/gpt-4o",
            messages=expected_messages,
            response_model=MockResponse,
            temperature=0.3
        )
    
    @patch('generate_glossary.utils.llm_simple.structured_completion')
    def test_infer_structured_anthropic_default_model(self, mock_structured):
        """Test infer_structured with Anthropic and default model"""
        mock_response = MockConceptList(concepts=["AI", "ML"], count=2)
        mock_structured.return_value = mock_response
        
        result = infer_structured(
            provider="anthropic",
            prompt="List AI concepts",
            response_model=MockConceptList
        )
        
        assert isinstance(result, LLMResult)
        assert result.text == mock_response
        
        expected_messages = [
            {"role": "user", "content": "List AI concepts"}
        ]
        mock_structured.assert_called_once_with(
            model="anthropic/claude-3-sonnet-20240229",
            messages=expected_messages,
            response_model=MockConceptList,
            temperature=0.7
        )
    
    @patch('generate_glossary.utils.llm_simple.text_completion')
    def test_infer_text_with_system_prompt(self, mock_text):
        """Test infer_text compatibility function with system prompt"""
        mock_text.return_value = "Generated definition"
        
        result = infer_text(
            provider="openai",
            prompt="Define machine learning",
            system_prompt="You are an academic expert",
            model="gpt-4",
            temperature=0.5
        )
        
        assert isinstance(result, LLMResult)
        assert result.text == "Generated definition"
        
        expected_messages = [
            {"role": "system", "content": "You are an academic expert"},
            {"role": "user", "content": "Define machine learning"}
        ]
        mock_text.assert_called_once_with(
            model="openai/gpt-4",
            messages=expected_messages,
            temperature=0.5
        )
    
    @patch('generate_glossary.utils.llm_simple.text_completion')
    def test_infer_text_unknown_provider_fallback(self, mock_text):
        """Test infer_text falls back to gpt-4 for unknown providers"""
        mock_text.return_value = "Fallback response"
        
        result = infer_text(
            provider="unknown_provider",
            prompt="Test prompt"
        )
        
        assert isinstance(result, LLMResult)
        assert result.text == "Fallback response"
        
        expected_messages = [
            {"role": "user", "content": "Test prompt"}
        ]
        mock_text.assert_called_once_with(
            model="unknown_provider/gpt-4",  # Falls back to gpt-4
            messages=expected_messages,
            temperature=0.7
        )


class TestErrorHandling:
    """Test error handling in LLM functions"""
    
    @patch('generate_glossary.utils.llm_simple.completion')
    def test_text_completion_exception_propagation(self, mock_completion):
        """Test that exceptions from LiteLLM are properly propagated"""
        mock_completion.side_effect = Exception("API Error")
        
        messages = [{"role": "user", "content": "test"}]
        
        with pytest.raises(Exception) as exc_info:
            text_completion("openai/gpt-4", messages)
        
        assert "API Error" in str(exc_info.value)
    
    @patch('generate_glossary.utils.llm_simple.client.chat.completions.create')
    def test_structured_completion_exception_propagation(self, mock_create):
        """Test that exceptions from Instructor are properly propagated"""
        mock_create.side_effect = Exception("Validation Error")
        
        messages = [{"role": "user", "content": "test"}]
        
        with pytest.raises(Exception) as exc_info:
            structured_completion(
                "openai/gpt-4", 
                messages, 
                MockResponse
            )
        
        assert "Validation Error" in str(exc_info.value)


class TestMessageFormatting:
    """Test message formatting in compatibility functions"""
    
    @patch('generate_glossary.utils.llm_simple.text_completion')
    def test_infer_text_no_system_prompt(self, mock_text):
        """Test infer_text without system prompt"""
        mock_text.return_value = "Response without system"
        
        result = infer_text(
            provider="openai",
            prompt="Just a user prompt"
        )
        
        expected_messages = [
            {"role": "user", "content": "Just a user prompt"}
        ]
        mock_text.assert_called_once_with(
            model="openai/gpt-4",
            messages=expected_messages,
            temperature=0.7
        )
    
    @patch('generate_glossary.utils.llm_simple.structured_completion')
    def test_infer_structured_message_order(self, mock_structured):
        """Test that system prompt comes before user prompt"""
        mock_response = MockResponse(content="ordered", confidence=1.0)
        mock_structured.return_value = mock_response
        
        result = infer_structured(
            provider="anthropic",
            prompt="User message",
            response_model=MockResponse,
            system_prompt="System message"
        )
        
        expected_messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"}
        ]
        mock_structured.assert_called_once_with(
            model="anthropic/claude-3-sonnet-20240229",
            messages=expected_messages,
            response_model=MockResponse,
            temperature=0.7
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])