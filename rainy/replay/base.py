from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

ReplayFeed = TypeVar('ReplayFeed')


class ReplayBuffer(ABC, Generic[ReplayFeed]):
    def __init__(self, feed: ReplayFeed) -> None:
        self.feed = feed

    @abstractmethod
    def append(self, *args) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[ReplayFeed]:
        pass

    @abstractmethod
    def __len__(self):
        pass

