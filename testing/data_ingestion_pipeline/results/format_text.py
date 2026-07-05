"""Reformat a parsed-sections text file into readable, word-wrapped paragraphs.

The PDF parser dumps each section's content as-is, so long paragraphs (e.g. the
abstract) end up on a single very long line that is hard to read. This script
rewraps that text to a fixed column width so every paragraph reads like it does
in the original PDF, and normalises the vertical spacing between sections.

Usage:
    python format_text.py <sections.txt> [width]
    python format_text.py 2606.32038v1_sections.txt 100

Writes a sibling file named "<stem>_formatted.txt".
"""

import sys
import textwrap
from pathlib import Path

DEFAULT_WIDTH = 100

# Structural lines we keep verbatim (they are markers, not prose to be wrapped).
SECTION_PREFIX = "[Section "


def is_separator(line: str) -> bool:
    """A ruler line made entirely of '=' or '-'."""
    stripped = line.strip()
    return len(stripped) >= 3 and set(stripped) <= {"=", "-"}


def format_lines(lines: list[str], width: int) -> list[str]:
    """Rewrap prose lines to `width` while leaving structural lines untouched."""
    out: list[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Preserve blank lines, separators, and section headers exactly.
        if not stripped:
            out.append("")
            continue
        if is_separator(line):
            # Normalise ruler width so everything lines up horizontally.
            out.append(stripped[0] * width)
            continue
        if stripped.startswith(SECTION_PREFIX) or stripped.startswith("("):
            out.append(line)
            continue

        # Word-wrap prose. break_long_words=False keeps tokens like URLs and
        # math notation (M ( x \ C )) intact instead of chopping them mid-token.
        wrapped = textwrap.fill(
            stripped,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        out.extend(wrapped.split("\n"))

    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python format_text.py <sections.txt> [width]")
        sys.exit(1)

    in_path = Path(sys.argv[1]).expanduser()
    if not in_path.is_absolute():
        # Resolve relative paths against this script's directory for convenience.
        in_path = (Path(__file__).resolve().parent / in_path).resolve()

    if not in_path.is_file():
        print(f"No text file found at {in_path}")
        sys.exit(1)

    width = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_WIDTH

    lines = in_path.read_text(encoding="utf-8").splitlines()
    formatted = format_lines(lines, width)

    out_path = in_path.with_name(f"{in_path.stem}_formatted.txt")
    out_path.write_text("\n".join(formatted) + "\n", encoding="utf-8")
    print(f"Formatted {in_path.name} -> {out_path.name} (width={width})")


if __name__ == "__main__":
    main()
