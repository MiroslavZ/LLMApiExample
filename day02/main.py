import os
import asyncio
import itertools
import logging
from typing import List, Optional, Union, Literal, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 1. Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        logging.FileHandler("matrix_test.log", encoding='utf-8')  # Вывод в файл
    ]
)
logger = logging.getLogger("DeepSeekMatrix")

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"

if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY не найден")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
app = FastAPI(title="DeepSeek Matrix Tester")


# --- Модели ---

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: List[Message]
    temperature: Optional[float] = 1.0

    # Список форматов
    response_format: List[Literal["text", "json_object", "json_schema"]] = Field(...)

    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None


# --- Вспомогательная функция (с логированием) ---

async def perform_chat_request(
        key: str,
        model: str,
        messages: List[dict],
        temperature: float,
        fmt_str: str,
        stop_val: Any,
        token_val: Any
):
    """
    Выполняет запрос и логирует результат сразу после получения ответа.
    """
    try:
        # Подготовка response_format
        api_fmt = {"type": fmt_str}

        # ВАЖНО: Для "json_schema" DeepSeek требует поле 'json_schema' с описанием.
        # Здесь мы добавляем заглушку, чтобы запрос не упал с 400 Bad Request,
        # если пользователь выберет этот тип.
        if fmt_str == "json_schema":
            api_fmt["json_schema"] = {
                "name": "test_schema",
                "schema": {"type": "object", "properties": {"result": {"type": "string"}}}
            }

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=api_fmt,
            stop=stop_val,
            max_tokens=token_val,
            stream=False
        )

        result_data = {
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": response.usage.model_dump()
        }

        # --- ЛОГИРОВАНИЕ УСПЕХА ---
        logger.info(
            f"✅ DONE: {key} | Reason: {result_data['finish_reason']} | Tokens: {result_data['usage']['total_tokens']}")

        return key, result_data

    except Exception as e:
        error_msg = str(e)

        # --- ЛОГИРОВАНИЕ ОШИБКИ ---
        logger.error(f"❌ FAIL: {key} | Error: {error_msg}")

        return key, {"error": error_msg}


# --- Эндпоинт ---

@app.post("/api/matrix-chat")
async def matrix_chat_proxy(request: ChatRequest):
    # Генерация списков опций (если None -> [None], иначе [None, value])
    stop_opts = [None, request.stop] if request.stop is not None else [None]
    token_opts = [None, request.max_tokens] if request.max_tokens is not None else [None]

    # Создаем комбинации (декартово произведение)
    combinations = itertools.product(request.response_format, stop_opts, token_opts)

    tasks = []

    # Создаем задачи
    for i, (fmt, stop_val, token_val) in enumerate(combinations, 1):
        key = (
            f"Scenario_{i}["
            f"fmt={fmt}, "
            f"stop={bool(stop_val)}, "
            f"max_tokens={bool(token_val)}"
            "]"
        )

        logger.info(f"🚀 START: {key}")  # Логируем запуск

        tasks.append(
            perform_chat_request(
                key=key,
                model=request.model,
                messages=[m.model_dump() for m in request.messages],
                temperature=request.temperature,
                fmt_str=fmt,
                stop_val=stop_val,
                token_val=token_val
            )
        )

    if not tasks:
        raise HTTPException(status_code=400, detail="Нет комбинаций для проверки")

    # Запускаем все параллельно и ждем завершения
    results = await asyncio.gather(*tasks)

    logger.info(f"🏁 ALL DONE: Processed {len(results)} combinations.")

    # Собираем итоговый словарь
    return dict(results)