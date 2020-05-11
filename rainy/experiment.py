import warnings
from pathlib import Path
from typing import List, Optional

import click

from .agents import Agent, DQNLikeAgent, DQNLikeParallel, EpisodeResult
from .lib.mpi import IS_MPI_ROOT


class Experiment:
    SAVE_FILE_DEFAULT = "rainy-agent.pth"
    ACTION_FILE_DEFAULT = "actions.json"

    def __init__(
        self,
        ag: Agent,
        save_file_name: Optional[str] = None,
        action_file_name: Optional[str] = None,
    ) -> None:
        if isinstance(ag, DQNLikeAgent) and ag.config.nworkers > 1:
            self.ag = DQNLikeParallel(ag)
        else:
            self.ag = ag
        self.logger = ag.logger
        self.config = ag.config
        self.logger.summary_setting(
            "train",
            ["episodes", "total_steps", "update_steps"],
            interval=ag.config.episode_log_freq,
            color="red",
        )
        self.logger.summary_setting(
            "eval",
            ["total_steps", "update_steps"],
            interval=ag.config.eval_times,
            color="green",
        )
        action_file = action_file_name or self.ACTION_FILE_DEFAULT
        self._action_file = Path(action_file)
        self._save_file_name = save_file_name or self.SAVE_FILE_DEFAULT
        self._has_eval_parallel = hasattr(self.ag, "eval_parallel")
        self.episode_offset = 0
        self.config.initialize_hooks()

    def log_episode(self, episodes: int, results: List[EpisodeResult]) -> None:
        for i, res in enumerate(results):
            self.logger.submit(
                "train",
                episodes=episodes + i + self.episode_offset,
                total_steps=self.ag.total_steps,
                update_steps=self.ag.update_steps,
                reward=res.return_,
                length=res.length,
            )

    def action_file(self) -> Path:
        return self.logger.logdir.joinpath(
            "{}-{}{}".format(self.action_file.stem, episodes, self.action_file.suffix)
        )

    def log_eval(self, episodes: int, eval_render: bool = False) -> None:
        logdir = self.logger.logdir
        if self.config.save_eval_actions and logdir is not None:
            fname = logdir.joinpath(
                "{}-{}{}".format(action_file.stem, episodes, action_file.suffix)
            )
            results = self._eval_impl(render=eval_render, action_file=fname)
        else:
            results = self._eval_impl(render=eval_render)
        for res in results:
            self.logger.submit(
                "eval",
                total_steps=self.ag.total_steps,
                update_steps=self.ag.update_steps,
                rewards=res.return_,
                length=res.length,
            )

    @staticmethod
    def _has_period(turn: int, width: int, freq: Optional[int]) -> bool:
        return freq and turn != 0 and turn // freq != (turn - width) // freq

    def _save(self, suffix: str = "") -> None:
        self.ag.save(self._save_file_name + suffix, self.logger.logdir)

    def train(self, saveid_start: int = 0, eval_render: bool = False) -> None:
        if not self.logger.ready:
            self.logger.setup_logdir()
        logdir = self.logger.logdir.as_posix()
        click.secho(f"Train stared :) Logdir: {logdir}", bg="white", fg="black")

        episodes = 0
        steps = 0
        save_id = saveid_start

        for res in self.ag.train_episodes(self.config.max_steps):
            ep_len = len(res)
            if ep_len > 0 and IS_MPI_ROOT:
                self.log_episode(episodes, res)
            episodes += ep_len
            step_diff = self.ag.total_steps - steps
            steps = self.ag.total_steps

            if not IS_MPI_ROOT:
                continue

            if self._has_period(steps, step_diff, self.config.eval_freq):
                self.log_eval(episodes, eval_render)
            if self._has_period(steps, step_diff, self.config.save_freq):
                self._save(suffix=".{}".format(save_id))
                save_id += 1

        self.log_eval(episodes, eval_render)
        self._save()
        self.ag.close()

    def _eval_impl(
        self,
        render: bool = False,
        replay: bool = False,
        pause: bool = False,
        action_file: Optional[str] = None,
    ) -> List[EpisodeResult]:
        n = self.config.eval_times
        self.ag.set_mode(train=False)
        if action_file is not None:
            res = [
                self.ag.eval_and_save(action_file, render=render, pause=pause)
                for _ in range(n)
            ]
        elif self._has_eval_parallel and not (render or replay) and n > 1:
            res = self.ag.eval_parallel(n)  # type: ignore
        else:
            res = [self.ag.eval_episode(render=render, pause=pause) for _ in range(n)]
        self.ag.set_mode(train=True)
        return res

    def _load_agent(self, logdir: Path) -> bool:
        if logdir.exists():
            self.ag.load(self._save_file_name, logdir)
            return True
        return False

    def retrain(
        self, logdir_: str, additional_steps: int = 100, eval_render: bool = False
    ) -> None:
        logdir = Path(logdir_)
        if not self._load_agent(logdir):
            raise ValueError("File'{}' does not exists".format(self._save_file_name))
        total_steps, episodes = self.logger.retrive(logdir)
        save_files = list(logdir.glob(self.SAVE_FILE_DEFAULT + ".*"))
        if len(save_files) > 0:
            save_id = len(save_files)
        else:
            save_id = 0
        self.ag.total_steps = total_steps
        self.ag.config.max_steps += additional_steps
        self.episode_offset = episodes
        self.train(save_id, eval_render)

    def evaluate(
        self, render: bool = False, replay: bool = False, pause: bool = False,
    ) -> None:
        if self.config.save_eval_actions:
            action_file = "eval-" + self._action_file.as_posix()
        else:
            action_file = None
        result = self._eval_impl(render, replay, pause, action_file)
        click.secho("======Results======", bg="blue", fg="yellow")
        if len(result) == 0:
            click.echo(result[0])
        else:
            from rainy.utils import LogStore

            eval_log = LogStore()
            for res in result:
                eval_log.submit(dict(rewards=res.return_, length=res.length))
            df = eval_log.into_df()
            click.echo(df)
            click.secho("===Summary===", bg="blue", fg="white")
            click.echo(df.describe())
        if render:
            click.pause("---Press any key to exit---")
        if replay:
            try:
                self.config.eval_env.unwrapped.replay()
            except Exception:
                warnings.warn("This environment does not support replay")
        self.ag.close()

    def load_and_evaluate(
        self,
        logdir_: str,
        render: bool = False,
        replay: bool = False,
        pause: bool = False,
    ) -> None:
        logdir = Path(logdir_)
        if not self._load_agent(logdir):
            raise ValueError("File'{}' does not exists".format(self._save_file_name))
        self.evaluate(render=render, replay=replay, pause=pause)

    def random(
        self, render: bool = False, replay: bool = False, pause: bool = False,
    ) -> None:

        if self.config.save_eval_actions:
            action_file = "random-" + self._action_file.as_posix()
            res = self.ag.random_and_save(action_file, render=render, pause=pause)
        else:
            res = self.ag.random_episode(render=render, pause=pause)
        click.echo(res)
        if render:
            click.pause("---Press any key to exit---")
        if replay:
            try:
                self.config.eval_env.unwrapped.replay()
            except Exception:
                warnings.warn("This environment does not support replay")
        self.ag.close()
