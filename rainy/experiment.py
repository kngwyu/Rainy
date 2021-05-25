import warnings
from pathlib import Path
from typing import List, Optional

import click

from .agents import Agent, DQNLikeAgent, DQNLikeParallel, EpisodeResult
from .lib.mpi import IS_MPI_ROOT
from .prelude import DEFAULT_SAVEFILE_NAME


class Experiment:
    def __init__(
        self,
        ag: Agent,
        save_file_name: Optional[str] = None,
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
        self._save_file_name = save_file_name or DEFAULT_SAVEFILE_NAME
        self._has_eval_parallel = hasattr(self.ag, "eval_parallel")
        self.episode_offset = 0
        self.config.initialize_hooks()
        self.noeval = False

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

    def log_eval(self, episodes: int, eval_render: bool = False) -> None:
        if self.noeval:
            self._msg("Do not run evaluation since noeval=True")
            return

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

    def _msg(self, msg: str, fg: str = "black", error: bool = False) -> None:
        click.secho("☔ " + msg + " ☔", bg="white", fg=fg, err=error)

    def abort(self, msg: str) -> None:
        self._msg(msg, fg="red", error=True)
        self.ag.close()

    def switch_agent(self, new_ag: Agent, load: bool = False) -> Agent:
        new_ag.logger = self.logger
        new_ag.total_steps = self.ag.total_steps
        old_ag = self.ag
        self.ag = new_ag
        self._load_agent(self.logger.logdir)
        return old_ag

    def train(
        self,
        saveid_start: int = 0,
        eval_render: bool = False,
        pretrained_agent_path: Optional[str] = None,
    ) -> None:
        if not self.logger.ready:
            self.logger.setup_logdir()
        logdir = self.logger.logdir.as_posix()
        self._msg(f"Train started (Logdir: {logdir})")

        if pretrained_agent_path is not None:
            if self._load_agent(Path(pretrained_agent_path)) is None:
                self.abort(f"{pretrained_agent_path} does not exist!")
                return
            else:
                self._msg(f"Loaded {pretrained_agent_path} before training")

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
    ) -> List[EpisodeResult]:
        n = self.config.eval_times
        self.ag.set_mode(train=False)
        if self._has_eval_parallel and not (render or replay) and n > 1:
            res = self.ag.eval_parallel(n)  # type: ignore
        else:
            res = [self.ag.eval_episode(render=render, pause=pause) for _ in range(n)]
        self.ag.set_mode(train=True)
        return res

    def _load_agent(self, logdir_or_file: Path) -> Path:
        if not logdir_or_file.exists():
            return None
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
        if agent_file is None:
            self.abort(f"{logdir_or_file} does not exist!")
            return
        self._msg(f"Loaded {agent_file} for re-training")
        total_steps, episodes = self.logger.retrive(agent_file.parent)
        self.ag.total_steps = total_steps
        self.episode_offset = episodes
        self._retrain_impl(eval_render)

    def evaluate(
        self,
        render: bool = False,
        replay: bool = False,
        pause: bool = False,
    ) -> None:
        result = self._eval_impl(render, replay, pause)
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
        if agent_file is None:
            self.abort(f"{logdir_or_file} does not exist!")
            return
        self._msg(f"Loaded {agent_file} for evaluation")
        self.evaluate(render=render, replay=replay, pause=pause)

    def random(
        self,
        render: bool = False,
        replay: bool = False,
        pause: bool = False,
    ) -> None:
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
