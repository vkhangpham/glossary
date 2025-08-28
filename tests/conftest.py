"""
Pytest configuration and fixtures for glossary tests
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

from pydantic import BaseModel, Field


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_concepts():
    """Sample concepts for testing"""
    return [
        "machine learning",
        "artificial intelligence", 
        "data structures",
        "algorithms",
        "software engineering",
        "computer networks",
        "database systems",
        "cybersecurity"
    ]


@pytest.fixture
def sample_academic_text():
    """Sample academic text for testing concept extraction"""
    return """
    The Computer Science Department offers courses in machine learning, 
    artificial intelligence, and data structures. Students learn algorithms 
    and software engineering principles while working on projects involving 
    neural networks, natural language processing, and computer vision.
    The curriculum covers both theoretical foundations and practical applications
    in areas such as distributed systems, cybersecurity, and human-computer interaction.
    """


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    mock_response = Mock()
    mock_response.text = "This is a mocked LLM response for testing purposes."
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mocked response content"
    return mock_response


@pytest.fixture
def mock_structured_response():
    """Mock structured LLM response for testing"""
    class TestModel(BaseModel):
        content: str = Field(description="Test content")
        score: float = Field(description="Test score")
    
    return TestModel(content="test response", score=0.95)


@pytest.fixture
def sample_glossary_metadata():
    """Sample glossary metadata structure for testing"""
    return {
        "machine learning": {
            "definition": "A field of artificial intelligence that focuses on algorithms that can learn from data",
            "level": 2,
            "parent_terms": ["artificial intelligence", "computer science"],
            "frequency": 15,
            "confidence": 0.92
        },
        "neural networks": {
            "definition": "Computing systems inspired by biological neural networks",
            "level": 3,
            "parent_terms": ["machine learning", "artificial intelligence"],
            "frequency": 8,
            "confidence": 0.88
        },
        "algorithms": {
            "definition": "Step-by-step procedures for solving computational problems",
            "level": 1,
            "parent_terms": ["computer science"],
            "frequency": 25,
            "confidence": 0.95
        }
    }


@pytest.fixture
def sample_web_resources():
    """Sample web resources for testing"""
    return {
        "machine learning": [
            {
                "url": "https://example.com/ml-intro",
                "title": "Introduction to Machine Learning",
                "processed_content": "Machine learning is a subset of AI that enables computers to learn from data...",
                "relevance_score": 0.9
            },
            {
                "url": "https://example.com/ml-algorithms", 
                "title": "ML Algorithms Overview",
                "processed_content": "Common machine learning algorithms include linear regression, decision trees...",
                "relevance_score": 0.85
            }
        ],
        "neural networks": [
            {
                "url": "https://example.com/nn-basics",
                "title": "Neural Network Fundamentals", 
                "processed_content": "Neural networks are computing systems inspired by biological neural networks...",
                "relevance_score": 0.88
            }
        ]
    }


@pytest.fixture
def env_with_api_keys(monkeypatch):
    """Set up environment with mock API keys for testing"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")


@pytest.fixture
def env_without_api_keys(monkeypatch):
    """Set up environment without API keys for testing error handling"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False) 
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


@pytest.fixture
def sample_config_data():
    """Sample configuration data for testing"""
    return {
        "levels": {
            0: {
                "name": "colleges",
                "batch_size": 10,
                "chunk_size": 100,
                "max_workers": 4,
                "llm_attempts": 3,
                "concept_agreement_threshold": 2,
                "keyword_appearance_threshold": 5
            },
            1: {
                "name": "departments", 
                "batch_size": 8,
                "chunk_size": 80,
                "max_workers": 4,
                "llm_attempts": 3,
                "concept_agreement_threshold": 2,
                "keyword_appearance_threshold": 3
            }
        },
        "providers": {
            "openai": {
                "models": {
                    "default": "gpt-4",
                    "mini": "gpt-4o-mini",
                    "large": "gpt-4o"
                }
            },
            "gemini": {
                "models": {
                    "default": "gemini-1.5-flash",
                    "pro": "gemini-1.5-pro"
                }
            }
        }
    }


@pytest.fixture
def temp_data_files(temp_dir, sample_concepts, sample_glossary_metadata):
    """Create temporary data files for testing"""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    
    # Create level directories and files
    for level in range(4):
        level_dir = data_dir / f"lv{level}"
        level_dir.mkdir()
        
        # Create concepts file
        concepts_file = level_dir / f"lv{level}_final.txt"
        with open(concepts_file, 'w') as f:
            for concept in sample_concepts[:5]:  # Different concepts per level
                f.write(f"{concept}\n")
        
        # Create metadata file  
        metadata_file = level_dir / f"lv{level}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(sample_glossary_metadata, f, indent=2)
    
    return data_dir


@pytest.fixture
def mock_litellm_completion():
    """Mock LiteLLM completion function"""
    mock = MagicMock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mocked LiteLLM response"
    mock.return_value = mock_response
    return mock


@pytest.fixture 
def mock_instructor_client():
    """Mock Instructor client"""
    mock_client = MagicMock()
    mock_response = Mock()
    mock_response.content = "test content"
    mock_response.score = 0.95
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture(scope="session")
def integration_test_marker():
    """Marker to identify integration tests that require API keys"""
    return pytest.mark.integration


def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests requiring API keys"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"  
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items to add markers automatically"""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to unit tests
        elif "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that make real API calls
        if any(keyword in item.name.lower() for keyword in ["real", "api", "actual"]):
            item.add_marker(pytest.mark.slow)