# tests/test_llm/test_router.py
import pytest
import os
import httpx
from src.llm.providers.openrouter import OpenRouterProvider
from src.llm.providers.base_provider import BaseProvider


@pytest.fixture
def provider():
    """Fixture for OpenRouter provider."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
    return OpenRouterProvider()


def test_api_key_loading():
    """Test that the API key is loaded correctly from .env"""
    provider = OpenRouterProvider()
    assert provider.api_key is not None
    assert len(provider.api_key) > 0
    assert provider.api_key != "your-openrouter-api-key-here"


def test_inheritance():
    """Test that OpenRouterProvider properly inherits from BaseProvider"""
    provider = OpenRouterProvider()
    assert isinstance(provider, BaseProvider)
    assert isinstance(provider, OpenRouterProvider)


def test_language_validation():
    """Test language validation logic"""
    provider = OpenRouterProvider()

    # Test valid languages
    for lang in ['python', 'javascript', 'typescript', 'solidity']:
        provider._validate_language(lang)  # Should not raise

    # Test invalid language
    with pytest.raises(ValueError) as exc_info:
        provider._validate_language('invalid_lang')
    assert "not supported" in str(exc_info.value)


def test_language_prompt():
    """Test language-specific prompt generation"""
    provider = OpenRouterProvider()

    # Test Python prompt
    python_prompt = provider._get_language_prompt('python')
    assert 'PEP 8' in python_prompt
    assert 'type hints' in python_prompt

    # Test JavaScript prompt
    js_prompt = provider._get_language_prompt('javascript')
    assert 'ESLint' in js_prompt
    assert 'ES6+' in js_prompt


@pytest.mark.asyncio
async def test_validate_connection(provider):
    """Test connection validation"""
    is_valid = await provider.validate_connection()
    assert is_valid is True


@pytest.mark.asyncio
async def test_generate_code(provider):
    """Test code generation"""
    prompt = "Write a Python function that implements binary search"
    result = await provider.generate_code(
        prompt=prompt,
        language="python",
        temperature=0.7
    )

    assert result is not None
    assert "def" in result
    assert "binary" in result.lower()
    assert "search" in result.lower()

    # Test with context
    context = {
        "return_type": "Optional[int]",
        "input_type": "List[int]",
    }

    result_with_context = await provider.generate_code(
        prompt=prompt,
        language="python",
        context=context,
        temperature=0.7
    )

    assert "Optional[int]" in result_with_context
    assert "List[int]" in result_with_context


@pytest.mark.asyncio
async def test_analyze_code(provider):
    """Test code analysis"""
    code = """
def binary_search(arr, target):
    # No type hints
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1  # No docstring
    """

    result = await provider.analyze_code(
        code=code,
        language="python"
    )

    assert result is not None
    assert "raw_analysis" in result
    assert "language" in result
    assert "timestamp" in result
    assert "has_security_issues" in result
    assert "has_performance_issues" in result
    assert "has_best_practice_issues" in result
    assert result["language"] == "python"


@pytest.mark.asyncio
async def test_error_handling(provider):
    """Test error handling"""
    # Test with invalid language
    with pytest.raises(ValueError) as exc_info:
        await provider.generate_code(
            prompt="Write a hello world program",
            language="invalid_language"
        )
    assert "not supported" in str(exc_info.value)

    # Test with empty prompt
    with pytest.raises(ValueError) as exc_info:
        await provider.generate_code(
            prompt="",
            language="python"
        )
    assert "cannot be empty" in str(exc_info.value)

    # Test with whitespace prompt
    with pytest.raises(ValueError) as exc_info:
        await provider.generate_code(
            prompt="    ",
            language="python"
        )
    assert "cannot be empty" in str(exc_info.value)

    # Test with invalid API key
    with pytest.raises(httpx.HTTPError) as exc_info:
        invalid_provider = OpenRouterProvider()
        invalid_provider.api_key = "invalid"  # Use a clearly invalid key
    assert "Invalid API key format" in str(exc_info.value)