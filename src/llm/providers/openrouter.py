# src/llm/providers/openrouter.py
import os
from typing import Dict, Optional, Any
import httpx
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import json
import asyncio

from .base_provider import BaseProvider


class OpenRouterProvider(BaseProvider):
    """OpenRouter implementation of the LLM provider."""

    def __init__(self):
        super().__init__()
        load_dotenv()

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.base_url = "https://openrouter.ai/api/v1"
        #self.default_model = "openai/gpt-4o-mini"
        self.default_model = "anthropic/claude-3.5-sonnet:beta"
        self._update_headers()

    def _update_headers(self):
        """Update headers with current API key."""
        if not self.api_key or len(self.api_key) < 32:  # Basic validation for API key format
            raise httpx.HTTPError("Invalid API key format")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-username/ai-developer-worker",
            "X-Title": "AI Developer Worker"
        }

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value
        if value:  # Only update headers if value is set
            self._update_headers()

    async def validate_connection(self) -> bool:
        """Validate OpenRouter connection and API key."""
        try:
            self._update_headers()  # Ensure headers are up to date
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self.headers,
                    timeout=10.0
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"OpenRouter connection validation failed: {e}")
            return False

    def _validate_prompt(self, prompt: str) -> None:
        """Validate that the prompt is not empty or just whitespace."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty or just whitespace")

    async def generate_code(
            self,
            prompt: str,
            language: str,
            context: Optional[Dict[str, Any]] = None,
            temperature: float = 0.7,
            max_tokens: int = 4000,
    ) -> str:
        """Generate code using OpenRouter."""
        self._validate_prompt(prompt)
        self._validate_language(language)

        try:
            # Get language-specific prompt additions
            language_prompt = self._get_language_prompt(language)

            # Break down the generation into smaller, focused tasks
            tasks = [
                "1. Create the project structure and essential configuration files",
                "2. Implement the core components and layouts",
                "3. Create the main pages and routing",
                "4. Add utility functions and helpers",
                "5. Implement styling and assets"
            ]

            full_response = []
            
            for task in tasks:
                # Construct task-specific prompt
                task_prompt = (
                    f"You are an expert {language} developer. Generate production-ready code "
                    f"for the following task: {task}\n\n"
                    f"Original request: {prompt}\n\n"
                    f"Follow these rules:\n"
                    f"1. Use actual module names instead of placeholders\n"
                    f"2. Generate complete file content - no truncation\n"
                    f"3. Use proper imports between files\n"
                    f"4. Include full implementations\n"
                    f"5. Follow {language} best practices\n"
                    f"{language_prompt}\n\n"
                    f"For each file, use exactly this format:\n"
                    f"FILE: filename.ext\n###CONTENT_START###\nComplete file content here\n###CONTENT_END###"
                )

                messages = [
                    {"role": "system", "content": task_prompt},
                    {"role": "user", "content": f"Generate code for: {task}"}
                ]

                # Add context if provided
                if context:
                    context_str = "\nContext:\n" + "\n".join(
                        f"{k}: {v}" for k, v in context.items()
                    )
                    messages.append({
                        "role": "system",
                        "content": context_str
                    })

                # Make API request
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json={
                            "model": self.default_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        timeout=30.0
                    )
                    response.raise_for_status()
                    result = response.json()

                    if "choices" not in result or not result["choices"]:
                        raise httpx.HTTPError("Invalid response from OpenRouter API")

                    current_response = result["choices"][0]["message"]["content"]
                    full_response.append(current_response)

            # Combine all responses
            combined_response = "\n".join(full_response)

            # Log the complete response
            log_dir = Path("./logs")
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            (log_dir / f"complete_response_{timestamp}.txt").write_text(combined_response)

            return combined_response

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during code generation: {e}")
            raise
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise httpx.HTTPError(f"Error during code generation: {str(e)}")

    async def analyze_code(self, content: str, language: str, context: Dict = None) -> Dict:
        """Analyze code using OpenRouter."""
        self._validate_prompt(content)
        self._validate_language(language)

        try:
            # Construct the system message
            system_message = (
                f"You are a code review expert for {language}. "
                f"Analyze the code for: security issues, performance optimizations, "
                f"best practices, and potential bugs. "
                f"Provide specific, actionable feedback in a structured format."
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Please analyze this code:\n\n{content}"}
            ]

            if context:
                messages.append({
                    "role": "system",
                    "content": f"Additional context:\n{json.dumps(context)}"
                })

            payload = {
                "model": self.default_model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1500,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()

            result = response.json()
            analysis = result["choices"][0]["message"]["content"]

            return {
                "raw_analysis": analysis,
                "language": language,
                "timestamp": result["created"],
                "has_security_issues": "security" in analysis.lower(),
                "has_performance_issues": "performance" in analysis.lower(),
                "has_best_practice_issues": "practice" in analysis.lower()
            }

        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            # Return empty dict instead of raising to allow analysis to continue
            return {}

    async def analyze_request(self, prompt: str) -> str:
        """Generate text response for analysis and recommendations."""
        try:
            # Construct messages array
            messages = [
                {"role": "system", "content": "You are an experienced software architect providing project analysis and recommendations."},
                {"role": "user", "content": prompt}
            ]

            # Prepare request payload
            payload = {
                "model": self.default_model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 4000,  # Increased for longer responses
                "stop": None  # Remove any stop sequences
            }

            # Make API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0  # Increased timeout
                )
                response.raise_for_status()

                result = response.json()
                if "choices" not in result or not result["choices"]:
                    raise httpx.HTTPError("Invalid response from OpenRouter API")

                # Get the complete response content
                content = result["choices"][0]["message"]["content"].strip()
                
                # Log the raw response
                log_dir = Path("./logs")
                log_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                (log_dir / f"llm_response_{timestamp}.txt").write_text(content)

                return content

        except Exception as e:
            logger.error(f"Analysis request failed: {e}")
            raise