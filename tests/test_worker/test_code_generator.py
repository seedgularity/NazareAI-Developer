# tests/test_worker/test_code_generator.py

import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from typing import Dict, Any, Optional

from src.worker.code_generator import CodeGenerator, GenerationContext
from src.llm.providers.base_provider import BaseProvider


class MockLLMProvider(BaseProvider):
    """Mock provider for testing"""

    async def generate_code(
            self,
            prompt: str,
            language: str,
            context: Optional[Dict[str, Any]] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000,
    ) -> str:
        return f"Mock generated {language} code for: {prompt}"

    async def analyze_code(
            self,
            code: str,
            language: str,
            context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "raw_analysis": f"Mock analysis of {language} code",
            "language": language,
            "timestamp": "2024-01-06T12:00:00",
            "has_security_issues": False,
            "has_performance_issues": False,
            "has_best_practice_issues": False
        }

    async def generate(self, prompt: str) -> str:
        return "Mock generated content for: " + prompt

    async def validate_connection(self) -> bool:
        return True


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Fixture to provide temporary output directory"""
    return tmp_path / "test_project"


@pytest.fixture
def mock_provider():
    """Fixture to provide mock LLM provider"""
    return MockLLMProvider()


@pytest.fixture
def code_generator(mock_provider):
    """Fixture to provide CodeGenerator instance"""
    return CodeGenerator(mock_provider)


@pytest.fixture
def sample_context(tmp_output_dir):
    """Fixture to provide sample GenerationContext"""
    return GenerationContext(
        project_type="nextjs",
        language="typescript",
        features=["auth", "api", "database"],
        requirements={"database": "postgresql", "auth": "nextauth"},
        output_dir=tmp_output_dir
    )


@pytest.mark.asyncio
async def test_generate_project_success(code_generator, sample_context):
    """Test successful project generation"""
    result = await code_generator.generate_project(sample_context)

    assert result is True
    assert sample_context.output_dir.exists()


@pytest.mark.asyncio
async def test_create_project_structure(code_generator, sample_context):
    """Test project structure creation"""
    await code_generator._create_project_structure(sample_context)
    assert sample_context.output_dir.exists()


@pytest.mark.asyncio
async def test_generate_project_handles_error(code_generator, sample_context):
    """Test error handling during project generation"""
    # Mock the LLM provider to raise an exception
    code_generator.llm.generate = AsyncMock(side_effect=Exception("Test error"))

    result = await code_generator.generate_project(sample_context)
    assert result is False


def test_build_structure_prompt(code_generator, sample_context):
    """Test structure prompt building"""
    prompt = code_generator._build_structure_prompt(sample_context)

    # Verify all required elements are in the prompt
    assert sample_context.project_type in prompt
    assert all(feature in prompt for feature in sample_context.features)
    # Check that each requirement key and value is in the prompt
    for key, value in sample_context.requirements.items():
        assert str(key) in prompt
        assert str(value) in prompt


def test_build_core_files_prompt(code_generator, sample_context):
    """Test core files prompt building"""
    prompt = code_generator._build_core_files_prompt(sample_context)

    # Verify prompt contains necessary project information
    assert sample_context.project_type in prompt
    assert all(feature in prompt for feature in sample_context.features)


def test_format_requirements(code_generator, sample_context):
    """Test requirements formatting"""
    formatted = code_generator._format_requirements(sample_context.requirements)

    # Verify format and content
    assert isinstance(formatted, str)
    assert "database: postgresql" in formatted
    assert "auth: nextauth" in formatted


@pytest.mark.asyncio
async def test_full_generation_pipeline(code_generator, sample_context):
    """Test the full generation pipeline with all steps"""
    result = await code_generator.generate_project(sample_context)

    # Verify overall success
    assert result is True
    assert sample_context.output_dir.exists()