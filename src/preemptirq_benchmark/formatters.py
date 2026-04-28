from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


def format_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    fmt: str,
    *,
    col_styles: dict[int, str] | None = None,
) -> str:
    """Render a table in the requested format.

    Args:
        title: Table title displayed above the table.
        headers: Column header strings.
        rows: List of rows, each a list of cell strings matching headers.
        fmt: Output format — one of "ascii", "txt", "markdown", "json".
        col_styles: Optional mapping of column index to rich style name,
            applied only in ascii format.

    Returns:
        Fully rendered table as a string.
    """
    if not headers or not rows:
        return f"{title}\n(no data)\n"

    formatters = {
        "ascii": format_ascii,
        "txt": format_txt,
        "markdown": format_markdown,
        "json": format_json,
    }
    return formatters[fmt](title, headers, rows, col_styles=col_styles)


def format_ascii(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    col_styles: dict[int, str] | None = None,
) -> str:
    """Render a rich box-drawing table with automatic color coding.

    Cells containing percentage deltas are colored green (improvement)
    or red (regression).  Significance markers (ns), (*), (**) are
    styled accordingly.

    Args:
        title: Table title displayed above the table.
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.
        col_styles: Optional mapping of column index to rich style name.

    Returns:
        Rendered table with ANSI escape codes for terminal display.
    """
    console = Console(file=None, force_terminal=True, width=200)
    table = Table(title=title, show_lines=False)

    for i, h in enumerate(headers):
        justify = "left" if i == 0 else "right"
        table.add_column(h, justify=justify)

    for row in rows:
        styled: list[str | Text] = []
        for i, cell in enumerate(row):
            if col_styles and i in col_styles:
                styled.append(Text(cell, style=col_styles[i]))
            else:
                styled.append(auto_style_cell(cell))
        table.add_row(*styled)

    with console.capture() as capture:
        console.print(table)
    return capture.get()


def auto_style_cell(cell: str) -> Text:
    """Apply color based on cell content conventions.

    Args:
        cell: Cell text to style.

    Returns:
        A rich Text object with the appropriate style applied:
        red for regression (+N.N%), green for improvement (-N.N%),
        dim for not significant (ns), yellow for p < 0.05 (*),
        bold yellow for p < 0.01 (**).
    """
    stripped = cell.strip()
    if stripped.endswith("(ns)"):
        return Text(cell, style="dim")
    if stripped.endswith("(**)"):
        return Text(cell, style="bold yellow")
    if stripped.endswith("(*)"):
        return Text(cell, style="yellow")
    if stripped.startswith("+") and "%" in stripped:
        return Text(cell, style="red")
    if stripped.startswith("-") and "%" in stripped:
        return Text(cell, style="green")
    return Text(cell)


def format_txt(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    **_kwargs: Any,
) -> str:
    """Render a plain-text table using +, -, and | characters.

    No color codes or box-drawing characters — suitable for piping to
    files or pasting into plain-text documents.

    Args:
        title: Table title printed on its own line above the table.
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.

    Returns:
        Plain-text table with no ANSI escape codes.
    """
    all_rows = [headers] + rows
    widths = [max(len(cell) for cell in col) for col in zip(*all_rows)]

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(row: list[str]) -> str:
        cells = []
        for i, (cell, w) in enumerate(zip(row, widths)):
            if i == 0:
                cells.append(f" {cell:<{w}} ")
            else:
                cells.append(f" {cell:>{w}} ")
        return "|" + "|".join(cells) + "|"

    lines = [title, sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines) + "\n"


def format_markdown(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    **_kwargs: Any,
) -> str:
    """Render a GitHub-flavored markdown table.

    Numeric columns are right-aligned via the `: ` separator syntax.

    Args:
        title: Section heading rendered as a level-3 markdown header.
        headers: Column header strings.
        rows: List of rows, each a list of cell strings.

    Returns:
        Markdown-formatted table string.
    """
    widths = [max(len(cell) for cell in col) for col in zip(*([headers] + rows))]

    def fmt_row(row: list[str]) -> str:
        cells = []
        for i, (cell, w) in enumerate(zip(row, widths)):
            if i == 0:
                cells.append(f" {cell:<{w}} ")
            else:
                cells.append(f" {cell:>{w}} ")
        return "|" + "|".join(cells) + "|"

    sep_cells = []
    for i, w in enumerate(widths):
        if i == 0:
            sep_cells.append(" " + "-" * w + " ")
        else:
            sep_cells.append(" " + "-" * (w - 1) + ": ")
    sep = "|" + "|".join(sep_cells) + "|"

    lines = [f"### {title}", "", fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append("")
    return "\n".join(lines)


def format_json(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    **_kwargs: Any,
) -> str:
    """Serialize table data as a JSON object.

    Args:
        title: Table title included in the JSON output.
        headers: Column header strings used as dict keys for each row.
        rows: List of rows, each a list of cell strings.

    Returns:
        Pretty-printed JSON string with title, headers, and rows
        (each row is a dict keyed by header name).
    """
    data = {
        "title": title,
        "headers": headers,
        "rows": [dict(zip(headers, row)) for row in rows],
    }
    return json.dumps(data, indent=2)
