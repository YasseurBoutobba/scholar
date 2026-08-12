import json
from dataclasses import dataclass

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

from backend.config import settings
from backend.utils.logging import get_logger

logger = get_logger(__name__)

NETWORK_ISSUE_MESSAGE = "Network issues reaching the LLM provider. Please try again shortly."


def is_network_issue(exc: BaseException) -> bool:
    return isinstance(exc, (RateLimitError, APIConnectionError, InternalServerError))


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int = 0


class LLMService:
    def __init__(self) -> None:
        if settings.OPENROUTER_API_KEY:
            self.provider = "openrouter"
            self._base_url = settings.OPENROUTER_BASE_URL
            self._api_key = settings.OPENROUTER_API_KEY
            self.fast_model = settings.OPENROUTER_FAST_MODEL
            self.strong_model = settings.OPENROUTER_STRONG_MODEL
        else:
            self.provider = "gemini"
            self._base_url = settings.GEMINI_BASE_URL
            self._api_key = settings.GEMINI_API_KEY
            self.fast_model = settings.GEMINI_FAST_MODEL
            self.strong_model = settings.GEMINI_STRONG_MODEL
            if not self._api_key:
                logger.warning(
                    "No OPENROUTER_API_KEY or GEMINI_API_KEY configured — "
                    "LLM calls will fail until one is set."
                )

        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        model = model or self.fast_model

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            logger.info(f"LLM call ({self.provider}/{model}): {tokens} tokens")
            return LLMResponse(content=content, model=model, tokens_used=tokens)

        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}/{model}): {e}")
            raise

    def complete_json(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
    ) -> dict:
        model = model or self.fast_model
        content = "{}"

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response, returning raw text")
            return {"raw": content}
        except Exception as e:
            logger.error(f"LLM JSON call failed ({self.provider}/{model}): {e}")
            raise
