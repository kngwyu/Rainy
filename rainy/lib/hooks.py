"""Training/evaluation hooks which make visualization or custom logging easier
"""
from abc import ABC
import cv2
import numpy as np
from torch import Tensor
from typing import Any, Dict
from ..envs import EnvExt, EnvTransition
from ..prelude import Action, State

Agent, Config = Any, Any


class EvalHook(ABC):
    """Evaluation hooks: do some stuff when evaluating
    """

    def setup(self, config: Config) -> None:
        pass

    def reset(self, agent: Agent, env: EnvExt, initial_state: State) -> None:
        pass

    def step(
        self,
        env: EnvExt,
        action: Action,
        transition: EnvTransition,
        net_outputs: Dict[str, Tensor],
    ) -> None:
        pass

    def close(self) -> None:
        pass


class VideoWriterHook(EvalHook):
    """Record video using env.render
    """

    def __init__(
        self, fps: float = 20.0, image_shape: str = "HWC", video_name: str = "video"
    ) -> None:
        self.logdir = None
        self.writer = None
        self.video_id = 0
        self.video_name = video_name
        self._fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._fps = fps
        self._w_index = image_shape.find("W")
        self._h_index = image_shape.find("H")
        if self._w_index < 0 or self._h_index < 0:
            raise ValueError(f"Invalid shape: {image_shape}")
        c_index = image_shape.find("C")
        if c_index < 0:
            self._transpose = self._h_index, self._w_index
        else:
            self._transpose = self._h_index, self._w_index, c_index

    def setup(self, config: Config) -> None:
        self.logdir = config.logger.logdir

    def reset(self, _agent: Agent, env: EnvExt, _initial_state: State) -> None:
        file_ = self.logdir.joinpath(f"{self.video_name}-{self.video_id}.avi")
        self.video_id += 1
        image = env.render(mode="rgb_array")
        shape = image.shape
        h, w = shape[self._h_index], shape[self._w_index]
        color = len(shape) == 3
        self.writer = cv2.VideoWriter(
            file_.as_posix(), self._fourcc, self._fps, (w, h), color
        )
        self.writer.write(np.transpose(image, self._transpose))

    def step(
        self,
        env: EnvExt,
        _action: Action,
        transition: EnvTransition,
        _net_outputs: Dict[str, Tensor],
    ) -> None:
        image = env.render(mode="rgb_array")
        self.writer.write(np.transpose(image, self._transpose))
        if transition.terminal:
            self.writer.release()
            self.writer = None

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
