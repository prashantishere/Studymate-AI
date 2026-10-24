"""AI Client for hosted Gemma model inference.

Supports:
1. Google's official `google-genai` SDK for Gemma models (including Gemma 4:
   e.g. `gemma-4-26b-a4b-it`, `gemma-4-31b-it`) hosted via Google AI Studio / Gemini API.
2. Standard OpenAI-compatible endpoints (Groq, OpenRouter, DeepInfra, vLLM).
3. Deterministic offline mock mode for headless unit testing without API keys.
"""

import json
import os
import re
from typing import Any
from dotenv import load_dotenv

from studymate.schemas import (
    ConfigurationError,
    InvalidModelResponseError,
    ModelRequestError,
)

# Load local .env if available
load_dotenv()


def clean_json_response(raw_text: str) -> str:
    """Extract raw JSON text from a model response, stripping markdown code fences."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


class GemmaAIClient:
    """Client for interacting with hosted Gemma models."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        mock_mode: bool | None = None,
    ):
        self.api_key = api_key or os.getenv("GEMMA_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.base_url = os.getenv("GEMMA_BASE_URL") or base_url
        self.model = model or os.getenv("GEMMA_MODEL") or "gemma-4-26b-a4b-it"
        self.timeout = timeout

        # Detect offline mock mode
        if mock_mode is not None:
            self.mock_mode = mock_mode
        else:
            env_mock = os.getenv("STUDYMATE_MOCK_AI", "0").lower()
            self.mock_mode = env_mock in ("1", "true", "yes")

        self._genai_client: Any = None
        self._openai_client: Any = None

    def _init_clients(self) -> None:
        """Initialize either google-genai or openai client depending on configuration."""
        if self.mock_mode:
            return

        api_key = self.api_key
        if not api_key:
            api_key = os.getenv("GEMMA_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ConfigurationError(
                "Missing GEMMA_API_KEY. Please set GEMMA_API_KEY or GEMINI_API_KEY in your .env file."
            )
        self.api_key = api_key

        # If a custom non-Google base URL is explicitly provided (e.g. Groq, OpenRouter), use OpenAI client
        if self.base_url and "googleapis.com" not in self.base_url:
            if self._openai_client is None:
                try:
                    from openai import OpenAI
                    self._openai_client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                    )
                except Exception as exc:
                    raise ConfigurationError(f"Failed to initialize OpenAI-compatible client: {exc}") from exc
            return

        # Otherwise, default to Google's official GenAI SDK (supports Gemma 4 natively on Gemini API)
        if self._genai_client is None:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
            except ImportError:
                # Fallback to OpenAI SDK with Google AI Studio OpenAI endpoint if google-genai is missing
                from openai import OpenAI
                endpoint = self.base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
                self._openai_client = OpenAI(
                    api_key=self.api_key,
                    base_url=endpoint,
                    timeout=self.timeout,
                )

    def generate_text(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Send a prompt to the Gemma model and return response text."""
        if self.mock_mode:
            return self._mock_text_response(prompt)

        self._init_clients()

        try:
            # 1. Use Google GenAI SDK if initialized
            if self._genai_client is not None:
                config: dict[str, Any] = {
                    "temperature": temperature,
                }
                if system_instruction:
                    config["system_instruction"] = system_instruction

                response = self._genai_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                return (response.text or "").strip()

            # 2. Use OpenAI-compatible client
            if self._openai_client is not None:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                return content.strip()

            raise ConfigurationError("No valid inference client could be initialized.")

        except Exception as exc:
            err_msg = str(exc)
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[REDACTED_API_KEY]")
            raise ModelRequestError(f"Gemma model inference failed: {err_msg}") from exc

    def generate_json(
        self,
        prompt: str,
        system_instruction: str = "",
    ) -> Any:
        """Prompt the model for a JSON response and parse the output."""
        if self.mock_mode:
            return self._mock_json_response(prompt)

        json_instruction = (
            system_instruction
            + "\nYou must respond ONLY with valid RFC-8259 JSON. Do not include markdown code fences, comments, or intro/outro text."
        ).strip()

        raw_output = self.generate_text(
            prompt=prompt,
            system_instruction=json_instruction,
            temperature=0.1,
        )

        cleaned = clean_json_response(raw_output)

        try:
            parsed = json.loads(cleaned)
            return parsed
        except json.JSONDecodeError as exc:
            raise InvalidModelResponseError(
                f"Model failed to return valid JSON. Error: {exc}. Raw snippet: {raw_output[:200]}"
            ) from exc

    # =================================================================
    # Deterministic Mock Responses for Tests and Offline Environments
    # =================================================================

    def _mock_text_response(self, prompt: str) -> str:
        """Provide structured mock responses when in offline test mode."""
        return (
            "Mock explanation based on provided materials: The concept is clearly defined in "
            "the source document with step-by-step principles."
        )

    def _mock_json_response(self, prompt: str) -> Any:
        """Provide deterministic mock JSON for diagnostic quiz, tutoring, or retest."""
        p_lower = prompt.lower()

        # Conversational Tutor response
        if "tutor" in p_lower or "intuitive_intro" in p_lower or "teaching response" in p_lower:
            return {
                "concept": "Conditional probability",
                "intuitive_intro": "Let's untangle this. Imagine looking at a room of students who took an exam.",
                "explanation": "We already know a student passed. To find if they attended extra classes, we don't divide by all students—we only count those who passed.",
                "example": "If 20 students passed out of 100, and 10 of those who passed went to extra classes, the probability is 10/20 = 50%, not 10/100.",
                "formal_connection": "Your notes formalize this as P(A|B) = P(A and B) / P(B) when P(B) > 0.",
                "source_pages": [1],
                "check_question": {
                    "question": "Why do we narrow our group only to students who passed?",
                    "options": [
                        "Because passing is the condition we already know has occurred",
                        "Because extra classes guarantee passing",
                        "Because we always discard half the sample space"
                    ],
                    "correct_index": 0,
                    "explanation": "In conditional probability, the given condition becomes our new reduced sample space."
                }
            }

        if "retest" in p_lower or "single multiple-choice" in p_lower:
            return {
                "id": "retest_q1",
                "concept": "Conditional probability",
                "question": "If events A and B are independent, what is P(A|B)?",
                "options": [
                    "P(A)",
                    "P(B)",
                    "P(A) * P(B)",
                    "0"
                ],
                "correct_index": 0,
                "explanation": "By definition of independence, the occurrence of B gives no information about A, so P(A|B) = P(A).",
                "source_pages": [1]
            }

        return [
            {
                "id": "q1",
                "concept": "Conditional probability",
                "question": "What does P(A|B) represent in probability theory?",
                "options": [
                    "The probability of event A given that event B has occurred",
                    "The probability of both A and B occurring together",
                    "The probability of event B given that event A has occurred",
                    "The union of events A and B"
                ],
                "correct_index": 0,
                "explanation": "P(A|B) is the conditional probability of A given B.",
                "source_pages": [1]
            },
            {
                "id": "q2",
                "concept": "Conditional probability",
                "question": "Which formula correctly defines P(A|B) when P(B) > 0?",
                "options": [
                    "P(A and B) / P(B)",
                    "P(A and B) / P(A)",
                    "P(A) + P(B)",
                    "P(A) * P(B)"
                ],
                "correct_index": 0,
                "explanation": "By Bayes' theorem and conditional probability, P(A|B) = P(A and B) / P(B).",
                "source_pages": [1]
            },
            {
                "id": "q3",
                "concept": "Probability distributions",
                "question": "What must the sum of all probabilities in a discrete probability distribution equal?",
                "options": [
                    "0",
                    "0.5",
                    "1.0",
                    "Infinity"
                ],
                "correct_index": 2,
                "explanation": "Total probability of all mutually exclusive and exhaustive events in a discrete sample space must sum to 1.",
                "source_pages": [2]
            },
            {
                "id": "q4",
                "concept": "Bayes Theorem",
                "question": "Bayes' theorem updates the prior probability of an event based on what?",
                "options": [
                    "New evidence or data",
                    "Arbitrary weighting",
                    "Averaging prior outcomes",
                    "Discarding priors"
                ],
                "correct_index": 0,
                "explanation": "Bayes' theorem calculates posterior probability by combining prior beliefs with likelihood of observed evidence.",
                "source_pages": [2]
            },
            {
                "id": "q5",
                "concept": "Expected Value",
                "question": "What does the expected value of a random variable measure?",
                "options": [
                    "The most unlikely outcome",
                    "The long-run average value of repetitions of the experiment",
                    "The maximum value of the distribution",
                    "The standard deviation"
                ],
                "correct_index": 1,
                "explanation": "Expected value represents the probability-weighted average of all possible values.",
                "source_pages": [2]
            }
        ]
