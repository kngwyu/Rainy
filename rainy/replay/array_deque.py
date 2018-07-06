from typing import Any


class ArrayDeque:
    def __init__(self, default_size: int=100, fixed_len: bool=False) -> None:
        self.num_elems = 0
        self.base = 0
        self.buf = [None for _ in range(default_size)]
        self.fixed_len = fixed_len
        self.default_size = default_size

    def _add(self, pos: int, item: Any) -> None:
        if len(self.buf) <= self.num_elems:
            self._resize()
        buf_len = len(self.buf)
        if pos < self.num_elems // 2:  # shift left
            self.base = buf_len - 1 if self.base == 0 else self.base - 1
            for i in range(self.base, self.base + pos):
                self.buf[i % buf_len] = self.buf[(i + 1) % buf_len]
        else:  # shift right
            for i in reversed(range(self.base + pos,
                                    self.base + self.num_elems)):
                self.buf[(i + 1) % buf_len] = self.buf[i % buf_len]
        self.buf[(self.base + pos) % buf_len] = item
        self.num_elems += 1

    def _remove(self, pos: int) -> Any:
        buf_len = len(self.buf)
        res = self.buf[(self.base + pos) % buf_len]
        if pos < self.num_elems // 2:  # shift right
            for i in reversed(range(self.base, self.base + pos)):
                self.buf[(i + 1) % buf_len] = self.buf[i % buf_len]
            self.base = (self.base + 1) % buf_len
        else:  # shift left
            for i in range(self.base + pos, self.base + self.num_elems - 1):
                self.buf[i % buf_len] = self.buf[(i + 1) % buf_len]
        self.num_elems -= 1
        if not self.fixed_len and buf_len > self.num_elems * 5:
            self._resize()
        return res

    def _resize(self):
        if self.num_elems > 0:
            new_len = self.num_elems * 2
            new_buf = [None for _ in range(new_len)]
            buf_len = len(self.buf)
            for i in range(self.num_elems):
                new_buf[i] = self.buf[(self.base + i) % buf_len]
            self.buf = new_buf
        self.base = 0

    def __iter__(self):
        for i in range(self.num_elems):
            yield self[i]

    def __repr__(self):
        return "ArrayDeque({})".format(str(list(iter(self))))

    def __getitem__(self, pos: int) -> Any:
        return self.buf[(self.base + pos) % len(self.buf)]

    def __setitem__(self, pos: int, item: Any) -> None:
        self.buf[(self.base + pos) % len(self.buf)] = item

    def push_back(self, item: Any) -> None:
        if self.fixed_len and self.num_elems == len(self.buf):
            self.pop_front()
        self._add(self.num_elems, item)

    def pop_back(self) -> Any:
        return self._remove(self.num_elems - 1)

    def push_front(self, item: Any) -> None:
        if self.fixed_len and self.num_elems == len(self.buf):
            self.pop_back()
        self._add(0, item)

    def pop_front(self) -> Any:
        return self._remove(0)

    def clear(self) -> None:
        self.num_elems = 0
        self.base = 0
        buf_len = len(self.buf)
        if buf_len > self.default_size:
            self.buf = [None for _ in range(self.default_size)]
        else:
            for i in range(buf_len):
                self.buf[i] = None


