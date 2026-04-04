"""
Слой работы с API языковой модели (Deepseek).
Не зависит от консоли и способа вывода — только запросы и возврат текста ответа.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Загружаем .env из каталога с этим модулем
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
ENV_API_KEY = "DEEPSEEK_API_KEY"

# Допустимые форматы ответа: текст (по умолчанию), схема (JSON Schema), объект (JSON)
ResponseFormatKind = str  # "text" | "schema" | "object"


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Текст ответа модели, счётчики токенов из usage (если есть) и длительность запроса к API."""

    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    elapsed_seconds: float = 0.0


META_PROMPT_SYSTEM = (
    "Составь оптимальный промпт для решения следующей задачи. Готовые ответы не нужны, "
    "нужен ТОЛЬКО промпт, который поможет модели прийти к правильному решению. План для решения задачи."
)


@dataclass(frozen=True, slots=True)
class MetaPromptCompletionResult:
    """Результат двухшагового запроса: сначала уточнение промпта, затем ответ по нему."""

    refined_prompt: str
    final: CompletionResult


class LLMAgent:
    """Агент запросов к API модели: ключ задаётся при создании, параметры вызова — в методах."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL) -> None:
        if not (api_key and api_key.strip()):
            raise ValueError(
                f"Не задан API-ключ. Укажите {ENV_API_KEY} в .env или передайте непустой api_key."
            )
        self._api_key = api_key.strip()
        self._base_url = base_url.strip() if base_url and base_url.strip() else DEFAULT_BASE_URL

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float = 1.0,
        response_format: ResponseFormatKind = "text",
        base_url: str | None = None,
    ) -> CompletionResult:
        """
        Отправляет промпт в модель и возвращает текст ответа и usage (токены).

        :raises ValueError: при неверном response_format
        """
        if response_format not in ("text", "schema", "object"):
            raise ValueError(
                f'response_format должен быть "text", "schema" или "object", получено: {response_format!r}'
            )
        messages: list[dict] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt})
        effective_base_url = base_url if base_url else self._base_url
        client = OpenAI(api_key=self._api_key, base_url=effective_base_url)
        create_kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        if stop:
            create_kwargs["stop"] = stop
        if response_format in ("schema", "object"):
            create_kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        response = client.chat.completions.create(**create_kwargs)
        elapsed = time.perf_counter() - t0
        content = response.choices[0].message.content or ""
        usage = response.usage
        if usage is None:
            return CompletionResult(content=content, elapsed_seconds=elapsed)
        return CompletionResult(
            content=content,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            elapsed_seconds=elapsed,
        )

    def complete_with_meta_prompt(
        self,
        user_task: str,
        *,
        system: str | None = None,
        meta_system: str = META_PROMPT_SYSTEM,
        model: str = DEFAULT_MODEL,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float = 1.0,
        response_format: ResponseFormatKind = "text",
        base_url: str | None = None,
    ) -> MetaPromptCompletionResult:
        """
        Мета-промптинг: запрос для генерации оптимального промпта, затем ответ по нему.

        :raises ValueError: если после первого шага модель вернула пустой промпт
        """
        meta_result = self.complete(
            user_task,
            system=meta_system,
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            response_format="text",
            base_url=base_url,
        )
        refined = meta_result.content.strip()
        if not refined:
            raise ValueError(
                "Модель не вернула оптимизированный промпт (пустой ответ на шаге мета-промпта)."
            )
        final = self.complete(
            refined,
            system=system,
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            response_format=response_format,
            base_url=base_url,
        )
        return MetaPromptCompletionResult(refined_prompt=refined, final=final)
