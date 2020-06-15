import atexit
import datetime as dt
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Tuple, Union

import click
import pandas as pd
from pandas import DataFrame


class LogStore:
    """A temporal object for storing logs.
    Since Pandas DataFrame/Series's append is not efficient, we store logs
    in this object before converting it into Pandas objects.
    """

    def __init__(self) -> None:
        self.inner: DefaultDict[str, List[Any]] = defaultdict(list)

    def submit(self, d: Dict[str, Any]) -> int:
        res = 0
        for key, value in d.items():
            # Convert return_ to return
            self.inner[key.rstrip("_")].append(value)
            if res == 0:
                res = len(self.inner[key])
        return res

    def into_df(self) -> DataFrame:
        df = DataFrame(self.inner)
        self.inner.clear()
        return df

    def to_df(self) -> DataFrame:
        return DataFrame(self.inner)

    def reset(self) -> None:
        self.inner.clear()

    def __getitem__(self, index: Union[int, slice]) -> "LogStore":
        res: DefaultDict[str, List[Any]] = defaultdict(list)
        for key, value in self.inner.items():
            res[key] = value[index]
        log_store = LogStore()
        log_store.inner = res
        return log_store

    def __len__(self) -> int:
        return len(self.inner)

    def __repr__(self) -> str:
        return "LogStore({})".format(dict.__repr__(self.inner))


class SummarySetting(NamedTuple):
    indices: List[str]
    interval: int
    color: str
    describe_range: bool


class ExperimentLogger:
    """
    - stores experiment log
    - exposes logs as pandas.DataFrame
    - prints summaries of logs to stdout
    """

    FINGERPRINT = "fingerprint.txt"
    LOG_CAPACITY = int(1e5)

    def __init__(self, show_summary: bool = True) -> None:
        self.logdir = Path("Results/Temp")
        self.exp_start = dt.datetime.now()
        self.exp_name = "No name"

        self._store: DefaultDict[str, LogStore] = defaultdict(LogStore)
        self._summary_setting: DefaultDict[str, SummarySetting] = defaultdict(
            lambda: SummarySetting(["sec"], 1, "black", True)
        )
        self._show_summary = show_summary
        self._closed = False
        self._time_offset = dt.timedelta(0)
        self.ready = False
        atexit.register(self.close)

    def setup_from_script_path(
        self,
        script_path_: str,
        dirname: Optional[str] = None,
        fingerprint: Optional[Dict[str, str]] = None,
    ) -> None:
        script_path = Path(script_path_)
        logdir_top = "Results"
        self.exp_name = script_path.stem
        if dirname is None:
            dirname = self.exp_start.strftime("%y%m%d-%H%M%S")
        self.logdir = Path(logdir_top).joinpath(self.exp_name).joinpath(dirname)
        if self.logdir.exists():
            warnings.warn(f"{self.logdir} already exists and can be overridden.")
        if fingerprint is None:
            fingerprint = {}
        fingerprint.update(self.git_metadata(script_path))
        self.setup_logdir(fingerprint=fingerprint)

    def retrive(self, logdir: Path) -> Tuple[int, int]:
        self.logdir = logdir
        sec, total_steps, episodes = 0, 0, 0
        for csvlog in self.logdir.glob("*.csv"):
            df = pd.read_csv(csvlog.as_posix())
            last = df.iloc[-1]
            sec = max(sec, last["sec"])
            if "total_steps" in last:
                total_steps = max(total_steps, last["total_steps"])
            if "episodes" in last:
                episodes = max(episodes, last["episodes"])
        self._time_offset = dt.timedelta(seconds=sec)
        return total_steps, episodes

    @staticmethod
    def git_metadata(script_path: Path) -> dict:
        res: Dict["str", "str"] = {}
        try:
            import git
        except ImportError:
            warnings.warn("GitPython is not Installed")
            return res
        try:
            repo = git.Repo(script_path, search_parent_directories=True)
            head = repo.head.commit
            res["git-head"] = head.hexsha
            res["git-diff"] = repo.git.diff()
        except git.exc.InvalidGitRepositoryError:
            warnings.warn("{} is not in a git repository".format(script_path))
        return res

    def setup_logdir(self, fingerprint: Dict[str, str] = {}) -> None:
        if not self.logdir.exists():
            self.logdir.mkdir(parents=True)
        finger = self.logdir.joinpath(self.FINGERPRINT)
        with finger.open(mode="w") as f:
            f.write("name: {}\nstarttime: {}\n".format(self.exp_name, self.exp_start))
            for fkey in fingerprint.keys():
                f.write("{}: {}\n".format(fkey, fingerprint[fkey]))
        self.ready = True

    def submit(self, name: str, **kwargs) -> None:
        """Stores log.
        """
        kwargs["sec"] = (dt.datetime.now() - self.exp_start).total_seconds()
        current_length = self._store[name].submit(kwargs)
        if self._show_summary:
            interval = self._summary_setting[name].interval
            if current_length % interval == 0:
                self.show_summary(name)
        if current_length >= self.LOG_CAPACITY:
            self._truncate_and_dump(name)

    def summary_setting(
        self,
        name: str,
        indices: List[str],
        interval: int = 1,
        color: str = "black",
        describe_range: bool = False,
    ) -> None:
        if "sec" not in indices:
            indices.append("sec")
        self._summary_setting[name] = SummarySetting(
            indices, interval, color, describe_range
        )

    def show_summary(self, name: str) -> None:
        indices, interval, color, describe_range = self._summary_setting[name]
        df = self._store[name][-interval:].to_df()
        indices_df = df[indices]
        click.secho(
            f"=========={name.upper()} LOG===========", bg=color, fg="white", bold=True
        )
        min_, max_ = indices_df.iloc[0], indices_df.iloc[-1]
        range_str = "\n".join([f"{idx}: {min_[idx]}-{max_[idx]}" for idx in indices])
        click.secho(range_str, bg="black", fg="white")
        df.drop(columns=indices, inplace=True)
        if describe_range:
            min_, max_ = df.iloc[0], df.iloc[-1]
            click.echo("\n".join([f"{i}: {min_[i]}-{max_[i]}" for i in df.columns]))
        else:
            describe = df.describe()
            describe.drop(labels="count", inplace=True)
            click.echo(describe)

    def close(self) -> None:
        if self._closed:
            return
        for key in self._store.keys():
            self._truncate_and_dump(key)
        self._closed = True

    def _truncate_and_dump(self, name: str) -> None:
        df = self._store[name].into_df()
        path = self.logdir.joinpath(name + ".csv")
        include_header = not path.exists()
        mode = "w" if include_header else "a"
        df.to_csv(path.as_posix(), mode=mode, header=include_header, index=False)
