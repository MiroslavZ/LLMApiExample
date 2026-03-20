"""
Слой работы с API языковой модели (Deepseek).
Не зависит от консоли и способа вывода — только запросы и возврат текста ответа.
"""

import os
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
) -> str:
    """
    Отправляет промпт в модель и возвращает текст ответа.

    :param prompt: текст запроса пользователя
    :param system: опционально — системный промпт (инструкции для модели)
    :param model: имя модели (по умолчанию deepseek-chat)
    :param max_tokens: максимальное число токенов в ответе (None — без ограничения)
    :param stop: список строк, при встрече любой из которых генерация останавливается
    :param temperature: температура генерации (по умолчанию 1.0)
    :param response_format: формат ответа — "text" (по умолчанию), "schema" (JSON Schema), "object" (JSON-объект)
    :param api_key: опционально — ключ API
    :param base_url: опционально — базовый URL API
    :return: текст ответа модели
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
    response = client.chat.completions.create(**create_kwargs)
    content = response.choices[0].message.content
    return content or ""
