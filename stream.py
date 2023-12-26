from types import TracebackType
from typing import Optional, Type


class Stream:
    BY_LINE = "line"
    BY_WORD = "word"
    BY_CHAR = "char"

    def __init__(self, data: str, by: str = "line"):
        self.data = data
        self.pointer = 0
        self.by = by

    def __iter__(self) -> 'Stream':
        return self

    def __enter__(self) -> 'Stream':
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        pass

    def __next__(self) -> str:
        if self.pointer >= len(self.data):
            raise StopIteration
        if self.by == self.BY_LINE:
            return self._next_line()
        elif self.by == self.BY_WORD:
            return self._next_word()
        elif self.by == self.BY_CHAR:
            return self._next_char()
        raise ValueError(f"Unknown stream type: {self.by}")

    def next(self) -> str:
        return self.__next__()

    def has_next(self) -> bool:
        return 0 <= self.pointer < len(self.data)

    def _next_line(self) -> str:
        start = self.pointer
        end = self.data.find("\n", start)
        if end == -1:
            end = len(self.data)
        self.pointer = end + 1 if end < len(self.data) else end
        return self.data[start:end]

    def _next_word(self) -> str:
        start = self.pointer
        end = start
        while end < len(self.data) and not self.data[end].isspace():
            end += 1
        self.pointer = end + 1 if end < len(self.data) else end
        return self.data[start:end]

    def _next_char(self) -> str:
        char = self.data[self.pointer]
        self.pointer += 1
        return char

    def reset(self) -> None:
        self.pointer = 0

    def seek(self, pos):
        if pos < 0 or pos >= len(self.data):
            raise ValueError(f"Invalid position: {pos}")
        self.pointer = pos

    def tell(self):
        return self.pointer

    def at(self, pos):
        if self.by == self.BY_LINE:
            return self._at_line(pos)
        elif self.by == self.BY_WORD:
            return self._at_word(pos)
        elif self.by == self.BY_CHAR:
            return self._at_char(pos)
        raise ValueError(f"Unknown stream type: {self.by}")

    def _at_line(self, pos):
        lines = self.lines()
        if pos < 0 or pos >= len(lines):
            raise ValueError(f"Invalid position: {pos}")
        return lines[pos]

    def _at_word(self, pos):
        words = self.words()
        if pos < 0 or pos >= len(words):
            raise ValueError(f"Invalid position: {pos}")
        return words[pos]

    def _at_char(self, pos):
        if pos < 0 or pos >= len(self.data):
            raise ValueError(f"Invalid position: {pos}")
        return self.data[pos]

    def lines(self):
        return self.data.splitlines()

    def words(self):
        return self.data.split()

    def chars(self):
        return list(self.data)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f"Stream(data={self.data}, by={self.by})"

    def __repr__(self):
        return str(self)


def main():
    text = "The disaster killed 35 persons on the airship, and one member of the ground crew, but miraculously 62 of the 97 passengers and crew survived.\nAfter the Hindenburg disaster, the airship era quickly faded, although hundreds of Zeppelins had been used during World War I, and thousands of non-rigid airships (often called \"blimps\") were used as naval patrol craft during World War II."
    stream = Stream(text, by=Stream.BY_WORD)

    for word in stream:
        print(word)


if __name__ == "__main__":
    main()
