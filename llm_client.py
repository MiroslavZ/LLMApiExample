"""
Слой работы с API языковой модели (Deepseek).
Не зависит от консоли и способа вывода — только запросы и возврат текста ответа.
"""

import json
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import tiktoken
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


@lru_cache(maxsize=1)
def _tiktoken_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def tiktoken_count_text(text: str) -> int:
    return len(_tiktoken_encoding().encode(text or ""))


def tiktoken_count_chat_messages(messages: list[dict[str, str]]) -> int:
    """
    Оценка числа токенов для списка сообщений чата (формат Chat Completions).
    Используется схема подсчёта как у GPT-4 на cl100k — разумное приближение для DeepSeek.
    """
    encoding = _tiktoken_encoding()
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Текст ответа модели, счётчики токенов из usage (если есть) и длительность запроса к API."""

    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    elapsed_seconds: float = 0.0
    # Оценки tiktoken (cl100k): запрос = все сообщения в этом вызове API; ответ = текст ассистента; история = весь диалог после шага (если считали)
    tiktoken_request: int | None = None
    tiktoken_completion: int | None = None
    tiktoken_dialog_total: int | None = None


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

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        clear: bool = False,
    ) -> None:
        if not (api_key and api_key.strip()):
            raise ValueError(f"Не задан API-ключ. Укажите {ENV_API_KEY} в .env или передайте непустой api_key.")
        self._api_key = api_key.strip()
        self._base_url = base_url.strip() if base_url and base_url.strip() else DEFAULT_BASE_URL
        self._history_path = Path.cwd() / "messages.json"
        self._messages: list[dict[str, str]] = []
        if not clear:
            self._load_history()

    def _load_history(self) -> None:
        """Загружает историю диалога из ``messages.json`` в корне текущей рабочей директории."""
        path = self._history_path
        if not path.is_file():
            self._messages = []
            return
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            self._messages = []
            return
        if not raw:
            self._messages = []
            return
        try:
            self._messages = json.loads(raw)
        except json.JSONDecodeError:
            self._messages = []

    def _save_history(self) -> None:
        """Сохраняет ``self._messages`` в JSON-файл истории."""
        self._history_path.write_text(
            json.dumps(self._messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _has_started_system_prompt(self) -> bool:
        """True, если в истории диалога уже есть сообщение с ролью ``system``."""
        return any(msg.get("role") == "system" for msg in self._messages)

    def get_dialog_history(self) -> list[dict[str, str]]:
        """Возвращает историю диалога: список сообщений ``{"role", "content"}`` (тот же объект, что ``self._messages``)."""
        return self._messages

    def clear_dialog_history(self) -> None:
        """Удаляет файл истории и сбрасывает список сообщений в памяти."""
        # по идее, метод не нужен, при создании агента можно просто не загружать историю,
        # после любого успешного запроса история все равно будет перезаписана
        self._history_path.unlink(missing_ok=True)
        self._messages = []

    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float = 1.0,
        response_format: ResponseFormatKind = "text",
        base_url: str | None = None,
    ) -> CompletionResult:
        """
        Внутренний вызов API: один запрос к модели.

        ``messages`` — список словарей ``{"role": ..., "content": ...}`` в формате Chat Completions API.

        :raises ValueError: при неверном response_format
        """
        if response_format not in ("text", "schema", "object"):
            raise ValueError(
                f'response_format должен быть "text", "schema" или "object", получено: {response_format!r}'
            )
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
        req_tok = tiktoken_count_chat_messages(messages)
        comp_tok = tiktoken_count_text(content)
        usage = response.usage
        if usage is None:
            return CompletionResult(
                content=content,
                elapsed_seconds=elapsed,
                tiktoken_request=req_tok,
                tiktoken_completion=comp_tok,
            )
        return CompletionResult(
            content=content,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            elapsed_seconds=elapsed,
            tiktoken_request=req_tok,
            tiktoken_completion=comp_tok,
        )

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
        """Отправляет промпт в модель и возвращает текст ответа и usage (токены)."""
        if system and system.strip() and not self._has_started_system_prompt():
            self._messages.append({"role": "system", "content": system.strip()})
        self._messages.append({"role": "user", "content": prompt})
        result = self._complete(
            self._messages,
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            response_format=response_format,
            base_url=base_url,
        )
        self._messages.append({"role": "assistant", "content": result.content})
        self._save_history()
        dialog_tok = tiktoken_count_chat_messages(self._messages)
        return replace(result, tiktoken_dialog_total=dialog_tok)

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
        # Трудно представляю, как правильно сочетать мета-промптинг с сохранением истории диалога,
        # остановился на том чтобы не включать в историю первую пару промптов (для генерации промпта моделью)
        meta_messages: list[dict[str, str]] = []
        if meta_system and meta_system.strip():
            meta_messages.append({"role": "system", "content": meta_system.strip()})
        meta_messages.append({"role": "user", "content": user_task})
        meta_result = self._complete(
            meta_messages,
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
        if system and system.strip() and not self._has_started_system_prompt():
            self._messages.append({"role": "system", "content": system.strip()})
        self._messages.append({"role": "user", "content": refined})
        final = self._complete(
            self._messages,
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            response_format=response_format,
            base_url=base_url,
        )
        self._messages.append({"role": "assistant", "content": final.content})
        self._save_history()
        dialog_tok = tiktoken_count_chat_messages(self._messages)
        final = replace(final, tiktoken_dialog_total=dialog_tok)
        return MetaPromptCompletionResult(refined_prompt=refined, final=final)
