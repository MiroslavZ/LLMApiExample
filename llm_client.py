"""
Слой работы с API языковой модели (Deepseek).
Не зависит от консоли и способа вывода — только запросы и возврат текста ответа.
"""

import os
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


def create_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """
    Создаёт клиент OpenAI для API Deepseek.

    :param api_key: ключ API (если None — берётся из переменной окружения DEEPSEEK_API_KEY)
    :param base_url: базовый URL API (по умолчанию — Deepseek)
    :raises ValueError: если api_key не передан и не задан в окружении
    """
    key = api_key or os.getenv(ENV_API_KEY)
    if not key:
        raise ValueError(
            f"Не задан API-ключ. Укажите {ENV_API_KEY} в .env или передайте api_key."
        )
    return OpenAI(api_key=key, base_url=base_url or DEFAULT_BASE_URL)


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


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    temperature: float = 1.0,
    response_format: ResponseFormatKind = "text",
    api_key: str | None = None,
    base_url: str | None = None,
) -> CompletionResult:
    """
    Отправляет промпт в модель и возвращает текст ответа и usage (токены).

    :param prompt: текст запроса пользователя
    :param system: опционально — системный промпт (инструкции для модели)
    :param model: имя модели (по умолчанию deepseek-chat)
    :param max_tokens: максимальное число токенов в ответе (None — без ограничения)
    :param stop: список строк, при встрече любой из которых генерация останавливается
    :param temperature: температура генерации (по умолчанию 1.0)
    :param response_format: формат ответа — "text" (по умолчанию), "schema" (JSON Schema), "object" (JSON-объект)
    :param api_key: опционально — ключ API
    :param base_url: опционально — базовый URL API
    :return: текст ответа модели, поля usage (prompt/completion/total), если API их вернул, и elapsed_seconds — время вызова API в секундах
    :raises ValueError: если нет API-ключа или неверный response_format
    :raises Exception: ошибки сети/API (пробрасываются без изменений)
    """
    if response_format not in ("text", "schema", "object"):
        raise ValueError(
            f'response_format должен быть "text", "schema" или "object", получено: {response_format!r}'
        )
    messages: list[dict] = []
    if system and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": prompt})
    client = create_client(api_key=api_key, base_url=base_url)
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


META_PROMPT_SYSTEM = (
    "Составь оптимальный промпт для решения следующей задачи. Готовые ответы не нужны, "
    "нужен ТОЛЬКО промпт, который поможет модели прийти к правильному решению. План для решения задачи."
)


@dataclass(frozen=True, slots=True)
class MetaPromptCompletionResult:
    """Результат двухшагового запроса: сначала уточнение промпта, затем ответ по нему."""

    refined_prompt: str
    final: CompletionResult


def complete_with_meta_prompt(
    user_task: str,
    *,
    system: str | None = None,
    meta_system: str = META_PROMPT_SYSTEM,
    model: str = DEFAULT_MODEL,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    temperature: float = 1.0,
    response_format: ResponseFormatKind = "text",
    api_key: str | None = None,
    base_url: str | None = None,
) -> MetaPromptCompletionResult:
    """
    Мета-промптинг: один запрос для генерации оптимального промпта по задаче, второй — ответ по этому промпту.

    :param user_task: исходная формулировка задачи пользователя
    :param system: системный промпт для финального ответа (как у ``complete``)
    :param meta_system: системный промпт для шага уточнения промпта
    :raises ValueError: если после первого шага модель вернула пустой промпт
    """
    meta_result = complete(
        user_task,
        system=meta_system,
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        response_format="text",
        api_key=api_key,
        base_url=base_url,
    )
    refined = meta_result.content.strip()
    if not refined:
        raise ValueError("Модель не вернула оптимизированный промпт (пустой ответ на шаге мета-промпта).")
    final = complete(
        refined,
        system=system,
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        response_format=response_format,
        api_key=api_key,
        base_url=base_url,
    )
    return MetaPromptCompletionResult(refined_prompt=refined, final=final)
