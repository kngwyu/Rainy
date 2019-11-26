from abc import ABC, abstractmethod
from typing import Generic, List, Type, TypeVar

ReplayFeed = TypeVar("ReplayFeed")


class ReplayBuffer(ABC, Generic[ReplayFeed]):
    def __init__(self, feed: Type[ReplayFeed]) -> None:
        self.feed = feed
        self.allow_overlap = False

    @abstractmethod
    def append(self, *args) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[ReplayFeed]:
        pass

    @abstractmethod
    def __len__(self):
        pass
