from abc import ABC, abstractmethod
from typing import Generic, List, Type, TypeVar

ReplayFeed = TypeVar("ReplayFeed")


class ReplayBuffer(ABC, Generic[ReplayFeed]):
    def __init__(self, feed: Type[ReplayFeed], allow_overlap: bool = False) -> None:
        self.feed = feed
        self.allow_overlap = allow_overlap

    @abstractmethod
    def append(self, *args) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[ReplayFeed]:
        pass

    @abstractmethod
    def __len__(self):
        pass
