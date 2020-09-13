import warnings
from pathlib import Path
from typing import List, Optional

import click

from .agents import Agent, DQNLikeAgent, DQNLikeParallel, EpisodeResult
from .lib.mpi import IS_MPI_ROOT
from .prelude import DEFAULT_ACTIONFILE_NAME, DEFAULT_SAVEFILE_NAME


class Experiment:
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
        action_file = action_file_name or DEFAULT_ACTIONFILE_NAME
        self._action_file = Path(action_file)
        self._save_file_name = save_file_name or DEFAULT_SAVEFILE_NAME
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
                return_=res.return_,
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
                return_=res.return_,
                length=res.length,
            )

    @staticmethod
    def _has_period(turn: int, width: int, freq: Optional[int]) -> bool:
        return freq and turn != 0 and turn // freq != (turn - width) // freq

    def _save(self, suffix: str = "") -> None:
        self.ag.save(self._save_file_name + suffix, self.logger.logdir)

    def _msg(self, msg: str) -> None:
        click.secho("☔ " + msg + " ☔", bg="white", fg="black")

    def train(self, saveid_start: int = 0, eval_render: bool = False) -> None:
        if not self.logger.ready:
            self.logger.setup_logdir()
        logdir = self.logger.logdir.as_posix()
        self._msg(f"Train started (Logdir: {logdir})")

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

            # Evaluate the agent
            if self._has_period(steps, step_diff, self.config.eval_freq):
                self.log_eval(episodes, eval_render)
            # Save models
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

    def _load_agent(self, logdir_or_file: Path) -> Path:
        if not logdir_or_file.exists():
            return False
        if logdir_or_file.is_file():
            fullpath = logdir_or_file
        else:
            fullpath = logdir_or_file.joinpath(self._save_file_name)
        if self.ag.load(fullpath):
            return fullpath
        else:
            raise ValueError(f"Failed to load {fullpath}")

    def _retrain_impl(self, additional_steps: int, eval_render: bool = False) -> None:
        self.ag.config.max_steps += additional_steps
        self.ag.initialize_rollouts()
        save_files = list(self.logger.logdir.glob(self._save_file_name + ".*"))
        if len(save_files) > 0:
            save_id = len(save_files)
        else:
            save_id = 0
        self.train(save_id, eval_render)

    def retrain(
        self,
        logdir_or_file: str,
        additional_steps: int = 100,
        eval_render: bool = False,
    ) -> None:
        agent_file = self._load_agent(Path(logdir_or_file))
        self._msg(f"Loaded {agent_file} for re-training")
        total_steps, episodes = self.logger.retrive(agent_file.parent)
        self.ag.total_steps = total_steps
        self.episode_offset = episodes
        self._retrain_impl(eval_render)

    def evaluate(
        self, render: bool = False, replay: bool = False, pause: bool = False,
    ) -> None:
        if self.config.save_eval_actions:
            action_file = "eval-" + self._action_file.as_posix()
        else:
            action_file = None
        result = self._eval_impl(render, replay, pause, action_file)
        click.secho("====== Results ======", bg="cyan", fg="black")
        if len(result) == 0:
            click.echo(result[0])
        else:
            from rainy.utils import LogStore

            eval_log = LogStore()
            for res in result:
                eval_log.submit(dict(rewards=res.return_, length=res.length))
            df = eval_log.into_df()
            click.echo(df)
            click.secho("====== Summary ======", bg="cyan", fg="black")
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
        logdir_or_file: str,
        render: bool = False,
        replay: bool = False,
        pause: bool = False,
    ) -> None:
        agent_file = self._load_agent(Path(logdir_or_file))
        self._msg(f"Loaded {agent_file} for evaluation")
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
