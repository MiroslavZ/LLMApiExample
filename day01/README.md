# Пример работы с Api языковой модели средствами библиотеки openai

## Зависимости

```commandline
pip install -r requirements.txt
```

## Конфигурационный файл

Создать .env файл в корне, прописать туда свой API ключ *DEEPSEEK_API_KEY=...* 

Использовал DeepSeek для примера.

## Запуск

```commandline
uvicorn main:app --reload 
```

## Интерфейс Swagger
После запуска http://127.0.0.1:8000/docs 