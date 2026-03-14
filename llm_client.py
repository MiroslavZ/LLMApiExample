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


def complete(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """
    Отправляет промпт в модель и возвращает текст ответа.

    :param prompt: текст запроса пользователя
    :param model: имя модели (по умолчанию deepseek-chat)
    :param api_key: опционально — ключ API
    :param base_url: опционально — базовый URL API
    :return: текст ответа модели
    :raises ValueError: если нет API-ключа
    :raises Exception: ошибки сети/API (пробрасываются без изменений)
    """
    client = create_client(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return content or ""
