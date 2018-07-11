"""Implementation of deque using list
"""
from typing import Any

MOD = 2147483648
MOD_ = 2147483647


class ArrayDeque:
    def __init__(self, capacity: int=7, fixed_len: bool=False) -> None:
        self.cap = capacity
        self.buf_cap = 1
        while self.buf_cap < capacity:
            self.buf_cap <<= 1
        self.tail = 0
        self.head = 0
        self.num_elems = 0
        self.buf = [None for _ in range(self.buf_cap)]
        self.fixed_len = fixed_len

    def __double(self):
        old_cap = self.buf_cap
        self.buf.extend([None for _ in range(old_cap)])
        if self.tail <= self.head:
            pass
        elif self.head < old_cap - self.tail:
            for i in range(self.head):
                self.buf[i + old_cap] = self[i]
            self.head += old_cap
        else:
            for i in range(self.tail, old_cap):
                self[i + old_cap] = self[i]
            self.tail += old_cap
        self.buf_cap += old_cap

    def __getitem__(self, i: int) -> Any:
        if i < self.num_elems:
            idx = ((self.tail + i) & MOD_) & (self.buf_cap - 1)
            return self.buf[idx]
        else:
            return None

    def __setitem__(self, i: int, item: Any):
        if i < self.num_elems:
            idx = ((self.tail + i) & MOD_) & (self.buf_cap - 1)
            self.buf[idx] = item

    def __len__(self):
        return self.num_elems

    def pop_front(self) -> Any:
        if self.head == self.tail:
            return None
        else:
            self.num_elems -= 1
            tail = self.tail
            self.tail = ((tail + 1) & MOD_) & (self.buf_cap - 1)
            return self.buf[tail]

    def push_front(self, item: Any):
        if not self.fixed_len and self.buf_cap - self.num_elems is 1:
            self.__double()
        self.tail = ((self.tail + MOD - 1) & MOD_) & (self.buf_cap - 1)
        self.buf[self.tail] = item
        self.num_elems += 1
        if self.fixed_len and self.num_elems == self.cap + 1:
            self.pop_back()

    def pop_back(self) -> Any:
        if self.head == self.tail:
            return None
        else:
            self.num_elems -= 1
            self.head = ((self.head + MOD - 1) & MOD_) & (self.buf_cap - 1)
            return self.buf[self.head]

    def push_back(self, item: Any):
        if not self.fixed_len and self.buf_cap - self.num_elems is 1:
            self.__double()
        head = self.head
        self.head = ((head + 1) & MOD_) & (self.buf_cap - 1)
        self.buf[head] = item
        self.num_elems += 1
        if self.fixed_len and self.num_elems == self.cap + 1:
            self.pop_front()

