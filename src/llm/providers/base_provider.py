# src/llm/providers/base_provider.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from loguru import logger


class BaseProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self):
        self.name = self.__class__.__name__
        logger.debug(f"Initializing {self.name} provider")

    @abstractmethod
    async def generate_code(
            self,
            prompt: str,
            language: str,
            context: Optional[Dict[str, Any]] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000,
    ) -> str:
        """
        Generate code using the LLM provider.

        Args:
            prompt: The main prompt for code generation
            language: Programming language for the code
            context: Additional context for code generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens in the response

        Returns:
            Generated code as string
        """
        pass

    @abstractmethod
    async def analyze_code(
            self,
            code: str,
            language: str,
            context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze code for improvements and potential issues.

        Args:
            code: The code to analyze
            language: Programming language of the code
            context: Additional context for analysis

        Returns:
            Dictionary containing analysis results
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate that the provider is properly configured and accessible.

        Returns:
            True if connection is valid, False otherwise
        """
        pass

    def _validate_language(self, language: str) -> None:
        """
        Validate that the specified programming language is supported.

        Args:
            language: Programming language to validate

        Raises:
            ValueError: If the language is not supported
        """
        supported_languages = {
            'python', 'javascript', 'typescript', 'solidity',
            'rust', 'go', 'java', 'kotlin', 'swift'
        }

        if language.lower() not in supported_languages:
            raise ValueError(
                f"Language '{language}' is not supported. "
                f"Supported languages: {sorted(supported_languages)}"
            )

    def _get_language_prompt(self, language: str) -> str:
        """
        Get language-specific prompt additions.

        Args:
            language: Programming language

        Returns:
            Language-specific prompt string
        """
        language_prompts = {
            'python': (
                "Follow PEP 8 style guide. Include type hints. "
                "Add docstrings for functions and classes."
            ),
            'javascript': (
                "Follow ESLint standards. Use modern ES6+ features. "
                "Add JSDoc comments for functions and classes."
            ),
            'typescript': (
                "Use strict typing. Follow TSLint standards. "
                "Implement interfaces where appropriate."
            ),
            'solidity': (
                "Follow Solidity style guide. Include NatSpec comments. "
                "Implement security best practices."
            ),
        }

        return language_prompts.get(
            language.lower(),
            "Follow standard best practices and conventions for the language."
        )

    async def analyze_request(self, prompt: str) -> str:
        """Generate text response for analysis and recommendations."""
        raise NotImplementedError