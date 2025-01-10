# src/llm/router.py
from typing import Dict, List, Optional, Union
import httpx
from loguru import logger
import os
from dotenv import load_dotenv


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = "openai/gpt-4-turbo-preview"  # Default to GPT-4
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/your-username/ai-developer-worker",  # Update this
            "X-Title": "AI Developer Worker"
        }

    async def generate_code(
            self,
            prompt: str,
            language: str,
            context: Optional[Dict] = None,
            model: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 2000,
    ) -> str:
        """
        Generate code using OpenRouter API.

        Args:
            prompt: The main prompt for code generation
            language: Programming language for the code
            context: Additional context for code generation
            model: Specific model to use (defaults to GPT-4)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response

        Returns:
            Generated code as string
        """
        try:
            # Construct the system message with language context
            system_message = (
                f"You are an expert {language} developer. Generate clean, "
                f"well-documented, and efficient code based on the requirements. "
                f"Follow best practices and include error handling."
            )

            # Construct the messages array
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
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

            # Prepare the request payload
            payload = {
                "model": model or self.default_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Make the API request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()

            result = response.json()
            generated_code = result["choices"][0]["message"]["content"]

            return generated_code

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during code generation: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during code generation: {e}")
            raise

    async def analyze_code(
            self,
            code: str,
            language: str,
            model: Optional[str] = None,
    ) -> Dict:
        """
        Analyze code for improvements and potential issues.

        Args:
            code: The code to analyze
            language: Programming language of the code
            model: Specific model to use (defaults to GPT-4)

        Returns:
            Dictionary containing analysis results
        """
        try:
            system_message = (
                f"You are a code review expert for {language}. "
                f"Analyze the code for: security issues, performance optimizations, "
                f"best practices, and potential bugs. Provide specific, actionable feedback."
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Please analyze this code:\n\n{code}"}
            ]

            payload = {
                "model": model or self.default_model,
                "messages": messages,
                "temperature": 0.3,  # Lower temperature for more focused analysis
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

            # Parse the analysis into structured format
            # TODO: Implement more structured parsing of the analysis
            return {
                "raw_analysis": analysis,
                "language": language,
                "timestamp": result["created"]
            }

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during code analysis: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during code analysis: {e}")
            raise