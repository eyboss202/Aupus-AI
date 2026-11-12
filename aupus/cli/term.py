"""Terminal styling for ``chat``: Aupus's colours and owl, panels, a spinner and word wrapping.

The look follows the browser chat (``chat_web.html``): paper-and-ink greys with Aupus blue, the
colour of its owl, as the one accent. Standard library only, like the rest of Aupus: plain ANSI
escape sequences, which every current terminal understands (on Windows 10+ they are switched on
with ``SetConsoleMode``).

Everything degrades by itself:

* output that isn't a terminal (a pipe, a file, the tests) gets no colours, no cursor movement
  and no spinner: just the text;
* ``NO_COLOR`` (https://no-color.org) turns the colours off but keeps the layout;
* a terminal whose encoding can't show the box-drawing characters gets ASCII ones;
* ``AUPUS_THEME=light`` picks darker colours for a light terminal background (guessed from
  ``COLORFGBG`` when that is set; dark otherwise).
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
import textwrap
import threading
import time
import unicodedata
from collections.abc import Callable
from typing import TextIO

CSI = "\x1b["
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
MAX_WIDTH = 100  # longer lines are hard to read, even in a wide terminal

RGB = tuple[int, int, int]

#: The web page's colours (``--brand``, ``--muted``, ...), for a dark and a light background.
DARK: dict[str, RGB] = {
    "brand": (142, 170, 245),
    "muted": (163, 161, 154),
    "faint": (125, 123, 117),
    "rule": (74, 74, 82),
    "warn": (224, 176, 96),
    "danger": (240, 138, 111),
}
LIGHT: dict[str, RGB] = {
    "brand": (20, 48, 140),
    "muted": (105, 103, 96),
    "faint": (139, 136, 128),
    "rule": (205, 201, 190),
    "warn": (138, 90, 0),
    "danger": (180, 50, 31),
}

#: The owl of ``logo.svg``, traced onto a 30 x 20 grid of quarter-character blocks: the two horns
#: and the arc between them, the brow, closed eyes and beak, then the wing and the long tail
#: feather on the left and the "3" of its right side.
OWL = [
    "▜▄▄ ▗▄▀▀▀▄▖ ▄▄▛",
    " ▐▛▀▙▄▖ ▗▄▟▀▜▌ ",
    " █ ▗  ▀█▀  ▖ █ ",
    " ▜▄ ▀▀ █ ▀▀ ▄▛ ",
    "  ▗▄▀▀▀▄  ▝▀█▖ ",
    " ▟▘    ▟▌    █ ",
    "▐▌    ▄█▘   ▗▛ ",
    "▐▌  ▄█▀    ▄▛  ",
    " ▘▄▛▀     ▀▘   ",
    "▗▛▘            ",
]
OWL_ASCII = [
    "  ,_,  ",
    " (-v-) ",
    " (   ) ",
    ' -"-"- ',
]

UNICODE_GLYPHS = {
    "h": "─", "v": "│", "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "lt": "├", "rt": "┤",
    "bullet": "●", "prompt": "❯", "dot": "·", "more": "…", "elbow": "└", "bar": "━", "warn": "▲",
}
ASCII_GLYPHS = {
    "h": "-", "v": "|", "tl": "+", "tr": "+", "bl": "+", "br": "+", "lt": "+", "rt": "+",
    "bullet": "*", "prompt": ">", "dot": "-", "more": "...", "elbow": "`-", "bar": "=", "warn": "!",
}
SPINNER_UNICODE = "·✢✳✶✻✽✻✶✳✢"
SPINNER_ASCII = "-\\|/"


def cell_width(text: str) -> int:
    """Columns ``text`` takes in a terminal: wide (CJK, emoji) characters take two, accents none."""
    width = 0
    for ch in ANSI.sub("", text):
        if unicodedata.combining(ch) or ch in "​‍️":
            continue
        width += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
    return width


def pad(text: str, width: int, align: str = "left") -> str:
    """``text`` (which may hold colour codes) padded with spaces to ``width`` columns."""
    gap = max(0, width - cell_width(text))
    if align == "center":
        return " " * (gap // 2) + text + " " * (gap - gap // 2)
    if align == "right":
        return " " * gap + text
    return text + " " * gap


def clip(text: str, width: int, more: str = "…") -> str:
    """Plain ``text`` cut to ``width`` columns, ending in ``more`` when something was cut."""
    if cell_width(text) <= width:
        return text
    out = ""
    for ch in text:
        if cell_width(out + ch + more) > width:
            break
        out += ch
    return out.rstrip() + more


def wrap(text: str, width: int) -> list[str]:
    """Word-wrap plain ``text`` to ``width`` columns, keeping its line breaks (and blank lines)."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines += textwrap.wrap(paragraph, max(1, width), break_on_hyphens=False) or [""]
    return lines


def _isatty(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def _can_encode(stream: TextIO, sample: str) -> bool:
    encoding = getattr(stream, "encoding", None)
    if not encoding:  # io.StringIO: any text goes
        return True
    try:
        sample.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _enable_ansi(stream: TextIO) -> bool:
    """Make sure the terminal behind ``stream`` interprets escape sequences (only Windows needs asking)."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enable_virtual_terminal_processing = 0x0004
        if mode.value & enable_virtual_terminal_processing:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing))
    except Exception:  # noqa: BLE001 (no console, or an old Windows: plain text then)
        return False


def _light_background() -> bool:
    theme = os.environ.get("AUPUS_THEME", "").lower()
    if theme in ("light", "dark"):
        return theme == "light"
    fgbg = os.environ.get("COLORFGBG", "")  # "15;0": white on black (rxvt, Konsole, iTerm2, ...)
    background = fgbg.rsplit(";", 1)[-1]
    return background.isdigit() and int(background) in (7, 15)


def _truecolor() -> bool:
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return True
    # Windows 10+ consoles and Windows Terminal, and the usual GUI terminals, all take 24-bit colour.
    if sys.platform == "win32" or os.environ.get("WT_SESSION"):
        return True
    return os.environ.get("TERM_PROGRAM", "") in ("vscode", "iTerm.app", "WezTerm", "ghostty")


def _xterm256(rgb: RGB) -> int:
    """The nearest colour of the 256-colour palette's 6x6x6 cube or its grey ramp."""
    r, g, b = rgb
    if max(rgb) - min(rgb) < 12:  # a grey
        return 232 + min(23, max(0, round((sum(rgb) / 3 - 8) / 10)))
    return 16 + 36 * round(r / 255 * 5) + 6 * round(g / 255 * 5) + round(b / 255 * 5)


class Term:
    """Where the chat writes, and what that place can show.

    ``fancy``: a real terminal that takes escape sequences (cursor movement, the spinner);
    ``color``: colours as well (``fancy`` and no ``NO_COLOR``); ``unicode``: box-drawing characters.
    Each can be forced for tests.
    """

    def __init__(
        self,
        out: TextIO,
        *,
        fancy: bool | None = None,
        color: bool | None = None,
        unicode: bool | None = None,
        light: bool | None = None,
    ) -> None:
        self.out = out
        if fancy is None:
            fancy = _isatty(out) and os.environ.get("TERM") != "dumb" and _enable_ansi(out)
        self.fancy = fancy
        self.color = (fancy and "NO_COLOR" not in os.environ) if color is None else color
        if unicode is None:
            unicode = _can_encode(out, "".join(UNICODE_GLYPHS.values()) + SPINNER_UNICODE + "".join(OWL))
        self.unicode = unicode
        self.g = UNICODE_GLYPHS if unicode else ASCII_GLYPHS
        self.palette = LIGHT if (_light_background() if light is None else light) else DARK
        self.truecolor = _truecolor()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ output

    def write(self, text: str) -> None:
        with self._lock:
            self.out.write(text)
            self.out.flush()

    def width(self) -> int:
        """Columns to lay text out in. The last column is left free, so the cursor never wraps by itself."""
        if not self.fancy:
            return 80
        return max(24, min(MAX_WIDTH, shutil.get_terminal_size((80, 24)).columns - 1))

    def columns(self) -> int:
        return shutil.get_terminal_size((80, 24)).columns if self.fancy else 80

    def paint(self, text: str, color: str | None = None, *, bold: bool = False, dim: bool = False) -> str:
        """``text`` in one of the palette's colours (``"brand"``, ``"muted"``, ...), when colours are on."""
        if not self.color or not text:
            return text
        codes = []
        if bold:
            codes.append("1")
        if dim:
            codes.append("2")
        if color:
            rgb = self.palette[color]
            codes.append("38;2;{};{};{}".format(*rgb) if self.truecolor else f"38;5;{_xterm256(rgb)}")
        return f"{CSI}{';'.join(codes)}m{text}{CSI}0m" if codes else text

    def rule(self, width: int | None = None) -> str:
        return self.paint(self.g["h"] * (width or self.width()), "rule")

    # ------------------------------------------------------------ cursor (fancy only)

    def up(self, lines: int) -> str:
        return f"\r{CSI}{lines}A" if self.fancy and lines > 0 else ""

    def clear_below(self) -> str:
        return f"\r{CSI}J" if self.fancy else ""

    def cursor(self, visible: bool) -> None:
        if self.fancy:
            self.write(f"{CSI}?25{'h' if visible else 'l'}")

    # ------------------------------------------------------------ building blocks

    def owl(self) -> list[str]:
        return OWL if self.unicode else OWL_ASCII

    def panel(self, rows: list[str], width: int, title: str = "") -> list[str]:
        """A box with rounded corners around ``rows`` (already padded to ``width - 4`` columns)."""
        g = self.g
        top_fill = width - 2 - (cell_width(title) + 5 if title else 0)
        border = lambda s: self.paint(s, "brand")  # noqa: E731
        if title:
            top = border(g["tl"] + g["h"] * 3) + " " + title + " " + border(g["h"] * top_fill + g["tr"])
        else:
            top = border(g["tl"] + g["h"] * (width - 2) + g["tr"])
        body = [border(g["v"]) + " " + pad(r, width - 4) + " " + border(g["v"]) for r in rows]
        return [top, *body, border(g["bl"] + g["h"] * (width - 2) + g["br"])]

    def meter(self, used: int, window: int, full: bool, cells: int = 10) -> str:
        """A small bar for how much of the model's memory (its context window) the chat fills."""
        share = min(1.0, used / window) if window else 0.0
        filled = min(cells, max(1 if used else 0, round(share * cells)))
        color = "warn" if full else "brand"
        bar = self.paint(self.g["bar"] * filled, color) + self.paint(self.g["bar"] * (cells - filled), "rule")
        label = f"{share:.0%} of memory" + (" (full: oldest messages are left out)" if full else "")
        return bar + " " + self.paint(label, "warn" if full else "muted")


class Spinner:
    """``✻ Pondering… (3s · Ctrl-C to stop)``, on one line until the first words; fancy terminals only."""

    def __init__(self, term: Term, label: str, hint: str = "Ctrl-C to stop") -> None:
        self.term, self.label, self.hint = term, label, hint
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if not self.term.fancy or self._thread:
            return
        self._start = time.monotonic()
        self.term.cursor(False)
        self._thread = threading.Thread(target=self._run, name="aupus-spinner", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        frames = SPINNER_UNICODE if self.term.unicode else SPINNER_ASCII
        i = 0
        while True:
            seconds = int(time.monotonic() - self._start)
            hint = f" ({seconds}s {self.term.g['dot']} {self.hint})" if self.hint else ""
            line = (
                self.term.paint(frames[i % len(frames)], "brand")
                + " "
                + self.term.paint(self.label + self.term.g["more"], "brand")
                + self.term.paint(hint, "muted")
            )
            self.term.write(f"\r{CSI}K{line}")
            i += 1
            if self._stop.wait(0.12):
                return

    def stop(self) -> None:
        """Remove the spinner (the cursor is left at the start of its now empty line)."""
        if not self._thread:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        self.term.write(f"\r{CSI}K")
        self.term.cursor(True)


class ReplyWriter:
    """Writes a reply as it streams in: ``● `` and the text, word-wrapped under a two-column indent.

    Leading blank space is dropped, runs of blank lines become one blank line and trailing ones
    are never written. A word that reaches the right edge is moved to the next line as a whole
    (in a fancy terminal the cursor goes back over it), unless it is longer than a line.
    """

    INDENT = 2

    def __init__(self, term: Term, width: int, color: str | None = None) -> None:
        self.term, self.width, self.color = term, width, color
        self.started = False
        self.col = 0
        self.word = ""
        self.newlines = 0
        self.space = False

    def _out(self, text: str) -> None:
        self.term.write(self.term.paint(text, self.color))

    def _newline(self) -> None:
        self.term.write("\n" + " " * self.INDENT)
        self.col = self.INDENT

    def push(self, piece: str) -> None:
        for ch in piece.replace("\r", "").replace("\t", "    "):
            if ch == "\n":
                if self.started:
                    self.newlines += 1
                    self.space = False
                    self.word = ""
            elif ch == " ":
                if self.started:
                    self.space = True
                    self.word = ""
            elif unicodedata.category(ch) != "Cc":
                self._char(ch)

    def _char(self, ch: str) -> None:
        if not self.started:
            self.term.write(self.term.paint(self.term.g["bullet"], "brand") + " ")
            self.started, self.col = True, self.INDENT
        if self.newlines:
            self.term.write("\n" * min(self.newlines, 2) + " " * self.INDENT)
            self.newlines, self.space, self.col = 0, False, self.INDENT
        if self.space:
            if self.col + 1 >= self.width:
                self._newline()
            else:
                self.term.write(" ")
                self.col += 1
            self.space = False
        w = cell_width(ch)
        if self.col + w > self.width:
            word_w = cell_width(self.word)
            if self.word and self.term.fancy and word_w + w <= self.width - self.INDENT:
                self.term.write(f"{CSI}{word_w}D{CSI}K")  # take the started word back ...
                self._newline()
                self._out(self.word)  # ... and write it again on the next line
                self.col += word_w
            else:
                self._newline()
                self.word = ""
        self._out(ch)
        self.col += w
        self.word += ch

    def finish(self) -> bool:
        """End the reply's line. Returns whether any text was written."""
        if self.started:
            self.term.write("\n")
        return self.started


def number(n: float) -> str:
    """``1,024``; one decimal below 10 for rates (``7.4``)."""
    if isinstance(n, float) and n < 10:
        return f"{n:.1f}"
    return f"{round(n):,}"


def run_with_spinner(term: Term, label: str, work: Callable[[], object], hint: str = "") -> object:
    """``work()`` with a spinner on screen meanwhile (fancy terminals only)."""
    spinner = Spinner(term, label, hint=hint)
    spinner.start()
    try:
        return work()
    finally:
        spinner.stop()


@contextlib.contextmanager
def visible_cursor(term: Term):  # noqa: ANN201 (a context manager)
    """Show the cursor again however the chat ends (Ctrl-C included)."""
    try:
        yield
    finally:
        term.cursor(True)
