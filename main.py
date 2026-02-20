import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Загружаем переменные окружения
load_dotenv()

# Получаем ключ из .env
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"  # Базовый URL для DeepSeek

if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY не найден в .env файле")

# Инициализация клиента DeepSeek (через SDK OpenAI)
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

app = FastAPI(title="DeepSeek Proxy Service")


# --- Pydantic модели для валидации данных в Swagger ---

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "deepseek-chat"  # Модель по умолчанию
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None


# --- Эндпоинт ---

@app.post("/api/chat", summary="Отправить запрос к DeepSeek")
async def chat_proxy(request: ChatRequest):
    """
    Принимает запрос, пересылает его в DeepSeek API и возвращает ответ.
    """
    try:
        # Выполняем запрос к API
        response = await client.chat.completions.create(
            model=request.model,
            messages=[msg.model_dump() for msg in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False  # Пока без стриминга для простоты
        )

        # Возвращаем контент ответа
        # Можно возвращать весь объект response, но обычно нужен только текст:
        return {
            "content": response.choices[0].message.content,
            "usage": response.usage.model_dump(),  # Статистика токенов
            "full_response": response.model_dump()  # Полный технический ответ (если нужно)
        }

    except Exception as e:
        # Обработка ошибок
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Проверка статуса")
async def root():
    return {"status": "Service is running"}