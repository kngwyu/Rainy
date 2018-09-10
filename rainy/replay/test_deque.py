from .array_deque import ArrayDeque
from collections import deque
import random


def test_deque_push_back():
    deq = ArrayDeque(capacity=10)
    for i in range(14):
        deq.push_back(i)

    for i in range(4, 14):
        assert deq[i - 4] == i


def test_deque_push_front():
    deq = ArrayDeque(capacity=10)
    for i in range(14):
        deq.push_front(i)
    print(deq)
    for i in range(4, 14):
        assert deq[13 - i] == i


def test_deque_stress():
    mydeq = ArrayDeque()
    deq = deque()
    N = 10000
    for i in range(N):
        num = random.randint(10, 1000000000)
        deq.append(num)
        mydeq.push_back(num)

    for i in range(N // 2):
        cond = random.randint(1, 4)
        num = random.randint(10, 1000000000)
        if cond is 1:
            deq.append(num)
            mydeq.push_back(num)
        elif cond is 2:
            deq.appendleft(num)
            mydeq.push_front(num)
        elif cond is 3:
            assert deq.pop() == mydeq.pop_back()
        else:
            assert deq.popleft() == mydeq.pop_front()


def test_deque_sample():
    deq = ArrayDeque()
    N = 10000
    for i in range(N):
        deq.push_back(i)
    K = 8000
    samples = deq.sample(K)
    samples = list(set(samples))
    assert len(samples) == K