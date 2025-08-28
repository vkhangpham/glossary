"""
Test migration compatibility - ensure old LLM usage patterns work with new system
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List
from pydantic import BaseModel, Field

from generate_glossary.utils.llm_simple import (
    infer_structured,
    infer_text,
    get_random_llm_config,
    LLMResult
)


# Test models matching those used in the actual codebase
class ConceptExtraction(BaseModel):
    """Model matching the one used in concept extraction"""
    source: str = Field(description="Source text being processed")
    concepts: List[str] = Field(description="List of extracted concepts")


class ConceptExtractionList(BaseModel):
    """Model matching the batch processing results"""
    extractions: List[ConceptExtraction] = Field(description="List of concept extractions")


class TestMigrationCompatibility:
    """Test that migrated code patterns work correctly"""
    
    @patch('generate_glossary.utils.llm_simple.structured_completion')
    def test_concept_extraction_pattern(self, mock_structured):
        """Test the concept extraction pattern used in lv0_s1_extract_concepts.py"""
        # Mock the response structure that would come from the real system
        mock_extraction = ConceptExtraction(
            source="College of Engineering and Applied Sciences",
            concepts=["engineering", "applied sciences"]
        )
        mock_response = ConceptExtractionList(extractions=[mock_extraction])
        mock_structured.return_value = mock_response
        
        # Simulate the exact pattern used in the migrated code
        provider, model = get_random_llm_config(level=0)
        system_prompt = """You are an expert in academic research classification..."""
        prompt = """Extract broad academic disciplines from these college names..."""
        
        response = infer_structured(
            provider=provider,
            prompt=prompt,
            response_model=ConceptExtractionList,
            system_prompt=system_prompt,
            model=model
        )
        
        # Verify the response structure matches what the code expects
        assert response is not None
        assert hasattr(response, 'text')
        extractions = response.text
        assert isinstance(extractions, ConceptExtractionList)
        assert len(extractions.extractions) > 0
        assert all(isinstance(e, ConceptExtraction) for e in extractions.extractions)
    
    @patch('generate_glossary.utils.llm_simple.text_completion')  
    def test_definition_generation_pattern(self, mock_text):
        """Test the definition generation pattern used in generate_definitions.py"""
        mock_text.return_value = "Machine learning is a subset of artificial intelligence..."
        
        # Simulate the pattern used in definition generation
        provider, model = get_random_llm_config()
        prompt = """Generate a formal, academic definition for the research topic: 'machine learning'..."""
        
        result = infer_text(
            provider=provider,
            prompt=prompt,
            model=model
        )
        
        # Verify the response structure matches expectations
        assert result is not None
        assert hasattr(result, 'text')
        assert isinstance(result.text, str)
        assert "machine learning" in result.text.lower()
    
    @patch('generate_glossary.utils.llm_simple.text_completion')
    def test_validator_pattern(self, mock_text):
        """Test the pattern used in validator/cli.py"""
        mock_text.return_value = "Yes, computer science is a valid academic discipline focused on computational systems..."
        
        # Simulate the validator pattern
        term = "computer science"
        prompt = f"Is '{term}' a valid academic discipline or field of study? Answer with 'yes' or 'no' followed by a brief explanation."
        
        response = infer_text(
            provider="openai",
            prompt=prompt
        )
        
        assert response is not None
        assert hasattr(response, 'text')
        assert isinstance(response.text, str)
        assert "yes" in response.text.lower()
    
    def test_llm_result_backward_compatibility(self):
        """Test that LLMResult maintains backward compatibility"""
        # Test with string content
        text_result = LLMResult("Simple text response")
        assert text_result.text == "Simple text response"
        
        # Test with structured content
        structured_content = ConceptExtraction(
            source="test source",
            concepts=["concept1", "concept2"]
        )
        structured_result = LLMResult(structured_content)
        assert structured_result.text == structured_content
        assert structured_result.text.source == "test source"
        assert len(structured_result.text.concepts) == 2
    
    def test_provider_model_selection_levels(self):
        """Test that provider/model selection works for all processing levels"""
        # Test all levels used in the system
        for level in range(4):
            provider, model = get_random_llm_config(level=level)
            
            assert provider in ["openai", "anthropic"]
            assert isinstance(model, str)
            assert len(model) > 0
            
            # Verify OpenAI models are valid
            if provider == "openai":
                valid_models = ["gpt-4", "gpt-4o", "gpt-4o-mini"]
                assert model in valid_models
    
    @patch('generate_glossary.utils.llm_simple.text_completion')
    def test_error_handling_migration(self, mock_text):
        """Test that error handling works as expected in migrated code"""
        # Test that exceptions are properly propagated
        mock_text.side_effect = Exception("API rate limit exceeded")
        
        with pytest.raises(Exception) as exc_info:
            infer_text(
                provider="openai",
                prompt="test prompt"
            )
        
        assert "API rate limit exceeded" in str(exc_info.value)
    
    @patch('generate_glossary.utils.llm_simple.structured_completion')
    def test_batch_processing_pattern(self, mock_structured):
        """Test the batch processing pattern used in concept extraction"""
        # Mock multiple extractions like in real usage
        mock_extractions = [
            ConceptExtraction(source="College of Engineering", concepts=["engineering"]),
            ConceptExtraction(source="School of Medicine", concepts=["medicine"]),
            ConceptExtraction(source="Department of Computer Science", concepts=["computer science"])
        ]
        mock_response = ConceptExtractionList(extractions=mock_extractions)
        mock_structured.return_value = mock_response
        
        # Simulate processing a batch of sources
        sources = [
            "College of Engineering",
            "School of Medicine", 
            "Department of Computer Science"
        ]
        
        sources_str = "\n".join(f"- {source}" for source in sources)
        prompt = f"""Extract concepts from these sources:\n{sources_str}"""
        
        response = infer_structured(
            provider="openai",
            prompt=prompt,
            response_model=ConceptExtractionList,
            system_prompt="Extract academic concepts...",
            model="gpt-4"
        )
        
        # Verify batch processing works correctly
        assert response is not None
        extractions = response.text
        assert len(extractions.extractions) == 3
        
        # Verify each extraction is properly structured
        for i, extraction in enumerate(extractions.extractions):
            assert extraction.source == sources[i]
            assert len(extraction.concepts) > 0
    
    def test_temperature_and_model_parameter_handling(self):
        """Test that temperature and model parameters are handled correctly"""
        with patch('generate_glossary.utils.llm_simple.text_completion') as mock_text:
            mock_text.return_value = "test response"
            
            # Test with explicit temperature
            result = infer_text(
                provider="openai",
                prompt="test",
                temperature=0.3,
                model="gpt-4"
            )
            
            # Verify the call was made with correct parameters
            mock_text.assert_called_once()
            call_kwargs = mock_text.call_args.kwargs
            
            # Check that text_completion was called with expected parameters
            # The actual call signature uses keyword arguments
            expected_kwargs = {
                'model': 'openai/gpt-4',
                'messages': [{'role': 'user', 'content': 'test'}],
                'temperature': 0.3
            }
            
            for key, value in expected_kwargs.items():
                assert key in call_kwargs
                assert call_kwargs[key] == value
    
    @patch('generate_glossary.utils.llm_simple.structured_completion')
    def test_system_prompt_integration(self, mock_structured):
        """Test that system prompts are correctly integrated in migrated code"""
        mock_response = ConceptExtraction(source="test", concepts=["test_concept"])
        mock_structured.return_value = mock_response
        
        system_prompt = "You are an expert in academic classification"
        user_prompt = "Extract concepts from this text"
        
        result = infer_structured(
            provider="openai",
            prompt=user_prompt,
            response_model=ConceptExtraction,
            system_prompt=system_prompt
        )
        
        # Verify the structured completion was called with correct message structure
        mock_structured.assert_called_once()
        call_kwargs = mock_structured.call_args.kwargs
        
        # Check that structured_completion was called with expected parameters
        assert 'messages' in call_kwargs
        messages = call_kwargs['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == system_prompt
        assert messages[1]['role'] == 'user'  
        assert messages[1]['content'] == user_prompt
        
        # Verify other expected parameters
        assert call_kwargs['model'] == 'openai/gpt-4'
        assert call_kwargs['response_model'] == ConceptExtraction


class TestSpecificMigrationScenarios:
    """Test specific scenarios from the migrated codebase"""
    
    @patch('generate_glossary.utils.llm_simple.text_completion')
    def test_sense_disambiguation_pattern(self, mock_text):
        """Test the pattern used in sense_disambiguation/splitter.py"""
        mock_text.return_value = "academic_computing"
        
        # Simulate the sense disambiguation pattern
        provider, model = get_random_llm_config()
        prompt = "Generate a domain tag for this cluster..."
        
        response = infer_text(
            provider=provider,
            prompt=prompt
        )
        
        assert response is not None
        assert hasattr(response, 'text')
        tag = response.text.strip()
        assert len(tag) > 0
    
    @patch('generate_glossary.utils.llm_simple.text_completion')
    def test_security_cli_pattern(self, mock_text):
        """Test the pattern used in utils/security_cli.py"""
        mock_text.return_value = "Connection test successful"
        
        # Simulate the security CLI test pattern
        test_prompt = "Respond with exactly: 'Connection test successful'"
        response = infer_text(
            provider="openai",
            prompt=test_prompt
        )
        
        assert response is not None
        assert "Connection test successful" in response.text
    
    def test_config_compatibility(self):
        """Test that configuration patterns work with new system"""
        # Test different level configurations
        configs = {}
        for level in range(4):
            provider, model = get_random_llm_config(level=level)
            configs[level] = (provider, model)
        
        # All configs should be valid and potentially different
        assert len(configs) == 4
        for level, (provider, model) in configs.items():
            assert provider in ["openai", "anthropic"]
            assert isinstance(model, str)
            assert len(model) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])