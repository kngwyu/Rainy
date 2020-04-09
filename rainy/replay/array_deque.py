"""Implementation of deque using 2 lists
"""
from collections.abc import Sequence
from typing import Any, List, Optional

from ..utils.sample import sample_indices


class ArrayDeque(Sequence):
    def __init__(
        self, capacity: Optional[int] = None, init_list: Optional[List[Any]] = None
    ) -> None:
        self.capacity = capacity
        self.front: List[Any] = []
        if init_list is None:
            init_list = []
        else:
            if capacity is not None:
                init_list = init_list[-capacity:].copy()
            else:
                init_list = init_list.copy()
        self.back: List[Any] = init_list

    def __len__(self):
        return len(self.front) + len(self.back)

    def __getitem__(self, i: int) -> Any:  # type: ignore
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

    def push_back(self, item: Any) -> Optional[Any]:
        self.back.append(item)
        if self.capacity and len(self) > self.capacity:
            return self.pop_front()

    def push_front(self, item: Any) -> Optional[Any]:
        self.front.append(item)
        if self.capacity and len(self) > self.capacity:
            return self.pop_back()

    def pop_back(self) -> Any:
        if not self.back:
            n = len(self)
            if n >= 2:
                self._balance()
            elif n >= 1:
                return self.front.pop()
            else:
                raise IndexError("[ArrayDeque::pop_back] Empty")
        return self.back.pop()

    def pop_front(self) -> Any:
        self._balance()
        if not self.front:
            raise IndexError("[ArrayDeque::pop_front] Empty")
        return self.front.pop()

    def clear(self) -> None:
        self.front.clear()
        self.back.clear()

    def _balance(self) -> None:
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
        if n < k:
            raise ValueError("[ArrayDeque::sample] n < k")
        return [
            self.front[i] if i < front_len else self.back[i - front_len]
            for i in sample_indices(n, k)
        ]

    def __repr__(self):
        return "ArrayDeque({})".format(str(list(self)))
