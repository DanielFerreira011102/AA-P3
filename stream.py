class Stream:
    """
    Stream represents a file stream that can be read by line, word or character.
    """

    BY_LINE = "l"
    BY_WORD = "w"
    BY_CHAR = "c"

    def __init__(self, file, by=BY_LINE, mode='r', encoding='utf-8', filter=None, map=None):
        """
        Initialize the Stream with the given file, granularity and mode.

        :param file: The file to be read.
        :param by: The way the file should be read (by line, word or character).
        :param mode: The mode in which the file should be opened.
        :param filter: The filter to be applied to each line.
        :param map: The map to be applied to each line.
        :return: The stream.
        """
        self.file = file
        self.open_file = open(file, mode, encoding=encoding)
        self.by = by
        self.filter = filter
        self.map = map

    def __enter__(self):
        """
        Enter the context of the Stream.

        :return: The stream.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context of the Stream.

        :param exc_type: The type of the exception.
        :param exc_value: The value of the exception.
        :param traceback: The traceback of the exception.
        """
        if self.open_file and not self.open_file.closed:
            self.open_file.close()

    def seek(self, offset, whence=0):
        """
        Seek to the given offset.

        :param offset: The offset to seek to.
        :param whence: The reference point from which the offset is added.
        """
        self.open_file.seek(offset, whence)

    def reset(self):
        """
        Reset the stream to the beginning.
        """
        self.open_file.seek(0)

    def tell(self):
        """
        Get the current position of the stream.

        :return: The current position of the stream.
        """
        return self.open_file.tell()
    
    def read(self, size=-1):
        """
        Read the given number of bytes from the stream.

        :param size: The number of bytes to read.
        :return: The bytes read.
        """
        return self.open_file.read(size)

    def __iter__(self):
        """
        Iterate through the stream with the given granularity.

        :return: The iterator of the stream with the given granularity.
        """
        if self.by == Stream.BY_LINE:
            return self._iter_lines()
        elif self.by == Stream.BY_WORD:
            return self._iter_words()
        elif self.by == Stream.BY_CHAR:
            return self._iter_chars()
        raise ValueError("Invalid by value: {}".format(self.by))
        
    def _iter(self, func):
        """
        Iterate through the stream with the given function.

        :param func: The function to be applied to each element.
        :return: The iterator of the stream with the given function.
        """
        for line in self.open_file:
            elements = func(line)
            if self.filter:
                elements = filter(self.filter, elements)
            if self.map:
                elements = map(self.map, elements)
            yield from elements

    def _iter_lines(self):
        """
        Iterate through the stream by line.

        :return: The iterator of the stream by line.
        """
        return self._iter(lambda line: (line.rstrip('\n'),))

    def _iter_words(self):
        """
        Iterate through the stream by word.

        :return: The iterator of the stream by word.
        """
        return self._iter(lambda line: line.split())

    def _iter_chars(self):
        """
        Iterate through the stream by character.

        :return: The iterator of the stream by character.
        """
        return self._iter(lambda line: line)

    def all(self):
        """
        Get all the lines, words or characters in the stream at once.

        :return: All the lines, words or characters in the stream.
        """
        return list(self)

    def __str__(self) -> str:
        """
        String representation of the stream.

        :return: The string representation of the stream.
        """
        return "Stream(file={}, by={})".format(self.file, self.by)
    
    def __repr__(self) -> str:
        """
        Representation of the stream.

        :return: The representation of the stream.
        """
        return str(self)

def main():
    file = "works/test.txt"

    for lines in Stream(file, by=Stream.BY_CHAR):
        print(lines)

if __name__ == "__main__":
    main()