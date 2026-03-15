#!/usr/bin/env python3
"""
Консольная обёртка для запросов к языковой модели.
Использование: python llm_chat.py --user "Ваш промпт здесь"
Опции: --max-tokens N — ограничить ответ N токенами; --stop STR [STR ...] — stop-последовательности; --format text|schema|object — формат ответа (по умолчанию text).
"""

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from llm_client import complete


def parse_args(args: list[str]) -> argparse.Namespace:
    """
    Разбирает аргументы командной строки.
    :return: объект с полями prompt_str, max_tokens (или None), stop_sequences (или None), response_format
    """
    parser = argparse.ArgumentParser(
        description="Запрос к языковой модели Deepseek",
        epilog='Пример: python llm_chat.py --user "Кратко объясни квантовую запутанность" --max-tokens 500 --stop "---" "\\n\\n"',
    )
    parser.add_argument(
        "--user",
        dest="prompt_str",
        required=True,
        metavar="TEXT",
        help="Текст промпта пользователя",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Максимальное число токенов в ответе (ограничение длины)",
    )
    parser.add_argument(
        "--stop",
        dest="stop_sequences",
        nargs="*",
        default=None,
        metavar="STR",
        help="Stop-последовательности: генерация остановится при появлении любой из строк (можно указать несколько)",
    )
    parser.add_argument(
        "--format",
        dest="response_format",
        choices=["text", "schema", "object"],
        default="text",
        metavar="FMT",
        help="Формат ответа: text — обычный текст (по умолчанию), schema — JSON Schema, object — JSON-объект",
    )
    parsed = parser.parse_args(args)
    return parsed


def main() -> None:
    console = Console()
    ns = parse_args(sys.argv[1:])

    prompt = ns.prompt_str.strip()
    if not prompt:
        console.print(
            '[yellow]Использование:[/yellow] python llm_chat.py --user "Ваш промпт" [--max-tokens N] [--stop STR ...]',
            style="bold",
        )
        sys.exit(1)
    console.print(
        Panel(f"[dim]{prompt}[/dim]", title="[bold]Промпт[/bold]", border_style="blue")
    )

    try:
        with console.status("[bold green]Запрос к модели..."):
            content = complete(
                prompt,
                max_tokens=ns.max_tokens,
                stop=ns.stop_sequences if ns.stop_sequences else None,
                response_format=ns.response_format,
            )
    except ValueError as e:
        console.print(f"[red]Ошибка:[/red] {e}", style="bold")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Ошибка API:[/red] {e}", style="bold")
        sys.exit(1)

    if not content:
        console.print("[yellow]Модель вернула пустой ответ.[/yellow]")
        return

    console.print(
        Panel(
            Markdown(content),
            title="[bold]Ответ Deepseek[/bold]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
