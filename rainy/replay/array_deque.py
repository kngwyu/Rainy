"""Implementation of deque using 2 lists
"""
from typing import Any, Optional
from ..util.sample import sample_indices


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
        if 3 * front_len < back_len:
            front_len_new = max((front_len + back_len) // 2, 1)
            ext_len = front_len_new - front_len
            self.front[:0] = [x for x in reversed(self.back[:ext_len])]
            del self.back[:ext_len]
        elif 3 * back_len < front_len:
            front_len_new = max((front_len + back_len) // 2, 1)
            ext_len = front_len - front_len_new
            self.back[:0] = [x for x in reversed(self.front[:ext_len])]
            del self.front[:ext_len]

    def sample(self, k):
        front_len = len(self.front)
        n = front_len + len(self.back)
        return [self.front[i] if i < front_len else self.back[i - front_len]
                for i in sample_indices(n, k)]
