from .array_deque import ArrayDeque
from collections import deque
import random


def test_deque_push():
    deq = ArrayDeque(default_size=10, fixed_len=True)
    for i in range(14):
        deq.push(i)
    for i in range(4, 14):
        assert deq[i - 4] == i


def test_deque_push_left():
    deq = ArrayDeque(default_size=10, fixed_len=True)
    for i in range(14):
        deq.push_left(i)
    print(deq)
    for i in range(4, 14):
        assert deq[13 - i] == i


def test_deque_stress():
    mydeq = ArrayDeque(default_size=2000)
    deq = deque()
    N = 10000
    for i in range(N):
        num = random.randint(10, 1000000000)
        deq.append(num)
        mydeq.push(num)

    for i in range(N // 2):
        cond = random.randint(1, 4)
        num = random.randint(10, 1000000000)
        if cond is 1:
            deq.append(num)
            mydeq.push(num)
        elif cond is 2:
            deq.appendleft(num)
            mydeq.push_left(num)
        elif cond is 3:
            assert deq.pop() == mydeq.pop()
        else:
            assert deq.popleft() == mydeq.pop_left()
