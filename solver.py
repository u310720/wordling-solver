import re
import sys
from functools import partial
from typing import Iterable

import nltk
import readchar
from nltk.corpus import words
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.table import Table
from rich.text import Text

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')


RuleText = partial(Text, style="bold yellow")
HighlightText = partial(Text, style="bold cyan")
PlainText = partial(Text, style="")

console = Console()


class WordlingSolver:
    def __init__(self, words: Iterable[str], page_size: int = 10):
        self.word_set = set(w.lower() for w in words)
        self.page_size = page_size
        self.pattern = ""
        self.include = ""
        self.exclude = ""
        self.cached_pattern = ""
        self.cached_exclude = ""
        self.cached_include = ""
        self.cached_matches = []

    def run(self):
        current_page = 0
        self._render_page(current_page)
        self._input_pattern()

        while True:
            self._render_page(current_page)
            match readchar.readkey():
                case readchar.key.RIGHT:
                    if current_page < self.total_pages - 1:
                        current_page += 1
                case readchar.key.LEFT:
                    if current_page > 0:
                        current_page -= 1
                case 'p':
                    self._input_pattern()
                    current_page = 0
                case 'i':
                    self._input_include()
                    current_page = 0
                case 'e':
                    self._input_exclude()
                    current_page = 0
                case 'r':
                    self.pattern = ""
                    self.include = ""
                    self.exclude = ""
                    current_page = 0
                case 'q':
                    sys.exit(0)
                case _:
                    continue

    def _prompt_input(self, prompt_msg):
        symbol = HTML('<b><purple>>> </purple></b>')
        console.rule(RuleText("Prompt"))
        console.print(prompt_msg)
        console.show_cursor(True)
        input_ = prompt(symbol).lower()
        return input_

    def _render_page(self, n: int = 0):
        console.clear()
        console.show_cursor(False)

        self._render_page_clues()
        self._render_page_matches(n)
        self._render_page_controls()

    def _render_page_clues(self):
        console.rule(RuleText("Clue"))
        pattern = HighlightText("Pattern    ")
        include = HighlightText("Include    ")
        exclude = HighlightText("Exclude    ")
        console.print(pattern, self.pattern, highlight=False)
        console.print(include, self.include, highlight=False)
        console.print(exclude, self.exclude, highlight=False)

    def _render_page_matches(self, n: int):
        page_text = Text.assemble(
            RuleText("Page "),
            HighlightText(f"{n + 1}/{self.total_pages}")
        )
        console.rule(page_text)

        # Display the matches in a table
        table = Table(box=None, show_header=False)
        start = n * self.page_size
        end = start + self.page_size
        for i in range(start, end):
            if i < len(self.matches):
                # Align the output with the header 'Pattern', 'Include', and
                # 'Exclude', which are 7 characters long, minus 1 for the '.'
                # character, so rjust(6, ' ')
                index = str(i + 1).rjust(6, ' ')
                table.add_row(f"{index}.    {self.matches[i]}")
            else:
                table.add_row("")
        console.print(table)

    def _render_page_controls(self, vertical_layout=False):
        console.rule(RuleText("Control"))
        table = Table(box=None, show_header=False,
                      expand=False, padding=(0, 4, 0, 0))

        ctrl_p = Text.assemble(
            HighlightText('p'),
            PlainText(" input pattern")
        )
        ctrl_i = Text.assemble(
            HighlightText('i'),
            PlainText(" input include characters")
        )
        ctrl_e = Text.assemble(
            HighlightText('e'),
            PlainText(" input exclude characters")
        )

        ctrl_left = Text.assemble(
            HighlightText('←'),
            PlainText(" previous page")
        )
        ctrl_right = Text.assemble(
            HighlightText('→'),
            PlainText(" next page")
        )

        ctrl_r = Text.assemble(HighlightText('r'), PlainText(" reset"))
        ctrl_q = Text.assemble(HighlightText('q'), PlainText(" quit"))

        if vertical_layout is True:
            table.add_row(ctrl_p, ctrl_left, ctrl_r)
            table.add_row(ctrl_i, ctrl_right, ctrl_q)
            table.add_row(ctrl_e)
        else:
            table.add_row(ctrl_p, ctrl_i, ctrl_e)
            table.add_row(ctrl_left, ctrl_right)
            table.add_row(ctrl_r, ctrl_q)

        console.print(table)

    @property
    def matches(self):
        if (self.pattern == self.cached_pattern and self.include == self.cached_include and self.exclude == self.cached_exclude):
            return self.cached_matches

        self.cached_pattern = self.pattern
        self.cached_include = self.include
        self.cached_exclude = self.exclude

        matches = []
        include_set = set(self.include)
        exclude_set = set(self.exclude)
        for w in self.word_set:
            if not re.fullmatch(self.pattern, w):
                continue
            if not include_set.issubset(w):
                continue
            if not exclude_set.isdisjoint(w):
                continue
            matches.append(w)
        self.cached_matches = matches
        return self.cached_matches

    @property
    def total_pages(self):
        return max((len(self.matches) + self.page_size - 1) // self.page_size, 1)

    def _input_pattern(self):
        prompt_msg = "Enter a regex pattern: "
        self.pattern = self._prompt_input(prompt_msg)

    def _input_include(self):
        prompt_msg = "Enter characters to include (e.g., 'ab', '+cde', '-fgh'): "
        self.include = self._common_input_include_exclude(
            prompt_msg, set(self.include))

    def _input_exclude(self):
        prompt_msg = "Enter characters to exclude (e.g., 'ab', '+cde', '-fgh')"
        self.exclude = self._common_input_include_exclude(
            prompt_msg, set(self.exclude))

    def _common_input_include_exclude(self, prompt_msg, char_set):
        valid_format = re.compile(r"^[+-]?[a-z]*$")
        input_ = self._prompt_input(prompt_msg)
        while not valid_format.fullmatch(input_):
            input_ = self._prompt_input(prompt_msg)

        if input_.startswith('+'):
            char_set |= set(input_[1:])
        elif input_.startswith('-'):
            char_set -= set(input_[1:])
        else:
            char_set = set(input_)

        return ''.join(sorted(char_set))


def main():
    word_list = words.words()
    solver = WordlingSolver(word_list)
    solver.run()


if __name__ == "__main__":
    main()
