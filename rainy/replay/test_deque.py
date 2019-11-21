from .array_deque import ArrayDeque
from collections import deque
import random


def test_deque_push_back() -> None:
    deq = ArrayDeque(capacity=10)
    for i in range(14):
        deq.push_back(i)

    for i in range(4, 14):
        assert deq[i - 4] == i


def test_deque_push_front() -> None:
    deq = ArrayDeque(capacity=10)
    for i in range(14):
        deq.push_front(i)
    print(deq)
    for i in range(4, 14):
        assert deq[13 - i] == i


def test_deque_stress() -> None:
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
        if cond == 1:
            deq.append(num)
            mydeq.push_back(num)
        elif cond == 2:
            deq.appendleft(num)
            mydeq.push_front(num)
        elif cond == 3:
            assert deq.pop() == mydeq.pop_back()
        else:
            assert deq.popleft() == mydeq.pop_front()


def test_deque_sample() -> None:
    deq = ArrayDeque()
    N = 10000
    for i in range(N):
        deq.push_back(i)
    K = 8000
    samples = deq.sample(K)
    samples = list(set(samples))
    assert len(samples) == K


def test_deque_sequence() -> None:
    deq = ArrayDeque(capacity=10)
    for i in range(14):
        deq.push_back(i)
    assert 12 in deq
    assert 18 not in deq
    assert min(deq) == 4
    assert len(deq) == 10
    assert deq.count(4) == 1


def test_deque_init_list() -> None:
    deq = ArrayDeque(capacity=10, init_list=[1, 2, 3])
    assert list(deq) == [1, 2, 3]
    deq = ArrayDeque(capacity=5, init_list=[1, 2, 3, 4, 5, 6, 7])
    assert list(deq) == [3, 4, 5, 6, 7]
