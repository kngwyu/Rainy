"""Implementation of deque using 2 lists
"""
from typing import Any, Optional


class ArrayDeque:
    def __init__(self, capacity: Optional[int] = None) -> None:
        self.front = []
        self.back = []
        self.capacity = capacity

    def __len__(self):
        return len(self.front) + len(self.back)

    def __getitem__(self, i: int) -> Any:
        front_len = len(self.front)
        if i < front_len:
            return self.front[~i]
        else:
            return self.back[i - front_len]

    def __setitem__(self, i: int, item: Any) -> None:
        front_len = len(self.front)
        if i < front_len:
            self.front[~i] = item
        else:
            self.back[i - front_len] = item

    def push_back(self, item: Any) -> None:
        self.back.append(item)
        if self.capacity and len(self) > self.capacity:
            self.pop_front()

    def push_front(self, item: Any) -> None:
        self.front.append(item)
        if self.capacity and len(self) > self.capacity:
            self.pop_back()

    def pop_back(self) -> Any:
        if not self.back:
            if len(self) >= 2:
                self.balance()
            else:
                return self.front.pop()
        return self.back.pop()

    def pop_front(self) -> Any:
        self.balance()
        return self.front.pop()

    def clear(self) -> None:
        self.front.clear()
        self.back.clear()

    def balance(self) -> None:
        front_len = len(self.front)
        back_len = len(self.back)
        n = front_len + back_len
        if 3 * front_len < back_len:
            front_len_new = max(n // 2, 1)
            ext_len = front_len_new - front_len
            front = [self.back[i] for i in reversed(range(ext_len))]
            self.front = front + self.front
            self.back = [self.back[i] for i in range(ext_len, n - front_len)]
        elif 3 * back_len < front_len:
            front_len_new = max(n // 2, 1)
            extend_len = front_len - front_len_new
            back = [self.front[i] for i in reversed(range(extend_len))]
            self.back = back + self.back
            self.front = [self.front[i] for i in range(extend_len, front_len)]

