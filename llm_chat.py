#!/usr/bin/env python3
"""
Консольная обёртка для запросов к языковой модели.
Использование: python llm_chat.py "Ваш промпт здесь"
"""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from llm_client import complete


def parse_prompt(args: list[str]) -> str | None:
    """
    Извлекает промпт из аргументов командной строки.
    :return: строка промпта или None, если аргументов нет/пусто
    """
    if not args:
        return None
    text = " ".join(args).strip()
    return text if text else None


def main() -> None:
    console = Console()

    prompt = parse_prompt(sys.argv[1:])
    if prompt is None:
        console.print(
            '[yellow]Использование:[/yellow] python llm_chat.py "Ваш промпт"',
            style="bold",
        )
        sys.exit(1)

    console.print(
        Panel(f"[dim]{prompt}[/dim]", title="[bold]Промпт[/bold]", border_style="blue")
    )

    try:
        with console.status("[bold green]Запрос к модели..."):
            content = complete(prompt)
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
