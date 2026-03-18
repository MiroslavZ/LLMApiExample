#!/usr/bin/env python3
"""
Консольная обёртка для запросов к языковой модели.
Использование: python llm_chat.py --user "Ваш промпт здесь"
Опции: --system TEXT — системный промпт; --meta-prompt — сначала сгенерировать оптимальный промпт, затем выполнить запрос с ним; --max-tokens N, --stop, --format text|schema|object.
"""

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from llm_client import complete

META_PROMPT_SYSTEM = (
    "Составь оптимальный промпт для решения следующей задачи. Готовые ответы не нужны, нужен ТОЛЬКО промпт, который поможет модели прийти к правильному решению. План для решения задачи."
)


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
        "--system",
        dest="system_prompt",
        default=None,
        metavar="TEXT",
        help="Системный промпт (инструкции для модели, задают роль/стиль ответа)",
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
    parser.add_argument(
        "--meta-prompt",
        dest="meta_prompt",
        action="store_true",
        default=False,
        help="Сначала сгенерировать оптимальный промпт для задачи, затем выполнить запрос с ним",
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

    system_for_request = (
        ns.system_prompt.strip() if ns.system_prompt and ns.system_prompt.strip() else None
    )
    effective_prompt = prompt

    if ns.meta_prompt:
        try:
            with console.status("[bold green]Генерация оптимального промпта..."):
                effective_prompt = complete(
                    prompt,
                    system=META_PROMPT_SYSTEM,
                    max_tokens=ns.max_tokens,
                    stop=ns.stop_sequences if ns.stop_sequences else None,
                    response_format="text",
                ).strip()
            if not effective_prompt:
                console.print("[red]Модель не вернула оптимизированный промпт.[/red]")
                sys.exit(1)
            console.print(
                Panel(
                    f"[dim]{effective_prompt}[/dim]",
                    title="[bold]Оптимизированный промпт[/bold]",
                    border_style="cyan",
                )
            )
        except Exception as e:
            console.print(f"[red]Ошибка при генерации промпта:[/red] {e}", style="bold")
            sys.exit(1)

    try:
        with console.status("[bold green]Запрос к модели..."):
            content = complete(
                effective_prompt,
                system=system_for_request,
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
