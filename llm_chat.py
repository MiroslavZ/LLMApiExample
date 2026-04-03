#!/usr/bin/env python3
"""
Консольная обёртка для запросов к языковой модели.
Использование: python llm_chat.py --user "Ваш промпт здесь"
Опции: --model NAME (по умолчанию deepseek-chat); --system TEXT — системный промпт; --meta-prompt — сначала сгенерировать оптимальный промпт, затем выполнить запрос с ним; --max-tokens N, --stop, --format text|schema|object, --temperature FLOAT.
"""

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from llm_client import (
    DEFAULT_MODEL,
    CompletionResult,
    complete,
    complete_with_meta_prompt,
)

def print_usage_stats(console: Console, result: CompletionResult) -> None:
    """Выводит prompt / completion / total из usage и время запроса после ответа модели."""
    parts: list[str] = []
    if result.prompt_tokens is not None:
        parts.append(f"prompt: {result.prompt_tokens}")
    if result.completion_tokens is not None:
        parts.append(f"completion: {result.completion_tokens}")
    if result.total_tokens is not None:
        parts.append(f"total: {result.total_tokens}")
    if result.elapsed_seconds > 0:
        parts.append(f"время: {result.elapsed_seconds:.2f} с")
    if not parts:
        return
    has_usage = any(
        x is not None
        for x in (result.prompt_tokens, result.completion_tokens, result.total_tokens)
    )
    if has_usage:
        console.print(f"[dim]Токены (usage): {' · '.join(parts)}[/dim]")
    else:
        console.print(f"[dim]Время запроса: {result.elapsed_seconds:.2f} с[/dim]")


def parse_args(args: list[str]) -> argparse.Namespace:
    """
    Разбирает аргументы командной строки.
    :return: объект с полями model, prompt_str, max_tokens (или None), stop_sequences (или None), response_format, temperature
    """
    parser = argparse.ArgumentParser(
        description="Запрос к языковой модели Deepseek",
        epilog='Пример: python llm_chat.py --user "Кратко объясни квантовую запутанность" --max-tokens 500 --stop "---" "\\n\\n"',
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="NAME",
        help=f"Идентификатор модели (по умолчанию: {DEFAULT_MODEL})",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="FLOAT",
        help="Температура генерации (по умолчанию: 1.0)",
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
    stop = ns.stop_sequences if ns.stop_sequences else None

    try:
        if ns.meta_prompt:
            with console.status("[bold green]Мета-промпт и запрос к модели..."):
                meta_out = complete_with_meta_prompt(
                    prompt,
                    system=system_for_request,
                    model=ns.model,
                    max_tokens=ns.max_tokens,
                    stop=stop,
                    response_format=ns.response_format,
                    temperature=ns.temperature,
                )
            completion = meta_out.final
            content = completion.content
            console.print(
                Panel(
                    f"[dim]{meta_out.refined_prompt}[/dim]",
                    title="[bold]Оптимизированный промпт[/bold]",
                    border_style="cyan",
                )
            )
        else:
            with console.status("[bold green]Запрос к модели..."):
                completion = complete(
                    prompt,
                    system=system_for_request,
                    model=ns.model,
                    max_tokens=ns.max_tokens,
                    stop=stop,
                    response_format=ns.response_format,
                    temperature=ns.temperature,
                )
                content = completion.content
    except ValueError as e:
        console.print(f"[red]Ошибка:[/red] {e}", style="bold")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Ошибка API:[/red] {e}", style="bold")
        sys.exit(1)

    if not content:
        console.print("[yellow]Модель вернула пустой ответ.[/yellow]")
        print_usage_stats(console, completion)
        return

    console.print(
        Panel(
            Markdown(content),
            title="[bold]Ответ Deepseek[/bold]",
            border_style="green",
        )
    )
    print_usage_stats(console, completion)


if __name__ == "__main__":
    main()
