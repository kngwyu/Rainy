import atexit
import click
from collections import defaultdict
from datetime import datetime
from pandas import DataFrame
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional, Union
import warnings


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
            self.inner[key].append(value)
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
        res = {}
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


class ExperimentLogger:
    """
    - stores experiment log
    - exposes logs as pandas.DataFrame
    - prints summaries of logs to stdout
    """
    FINGERPRINT = 'fingerprint.txt'
    LOG_CAPACITY = int(1e6)

    def __init__(self, show_summary: bool = True) -> None:
        self._logdir: Optional[Path] = None
        self._store: DefaultDict[str, LogStore] = defaultdict(LogStore)
        self._summary_setting: DefaultDict[str, SummarySetting] = \
            defaultdict(lambda: SummarySetting(["time"], 1, "black"))
        self.exp_start = datetime.now()
        self.exp_name = "No name"
        self.time_offset = 0.0
        self._show_summary = show_summary
        self._closed = False
        atexit.register(self.close)

    def set_dir_from_script_path(
            self,
            script_path_: str,
            prefix: str = '',
            fingerprint: Dict[str, str] = {},
    ) -> None:
        script_path = Path(script_path_)
        logdir_top = "Results"
        if prefix:
            logdir_top = prefix + '/' + logdir
        self.exp_name = script_path.stem
        time = self.exp_start.strftime("%y%m%d-%H%M%S")
        metadata = self.git_metadata(script_path)
        logdir = Path(logdir_top).joinpath(self.exp_name).joinpath(time)
        logdir.mkdir(parents=True)
        fingerprint.update(metadata)
        self.set_dir(logdir, fingerprint=fingerprint)

    @staticmethod
    def git_metadata(script_path: Path) -> dict:
        try:
            import git
        except ImportError:
            warnings.warn('GitPython is not Installed')
            return
        res = {}
        try:
            repo = git.Repo(script_path, search_parent_directories=True)
            head = repo.head.commit
            res["git-head"] = head.hexsha
            res["git-diff"] = repo.git.diff()
        except git.exc.InvalidGitRepositoryError:
            warnings.warn('{} is not in a git repository'.format(script_path))
        return res

    def set_dir(self, logdir: Path, fingerprint: Dict[str, str] = {}) -> None:
        self.logdir = logdir
        finger = logdir.joinpath(self.FINGERPRINT)
        with finger.open(mode="w") as f:
            f.write('name: {}\nstarttime: {}\n'.format(self.exp_name, self.exp_start))
            for fkey in fingerprint.keys():
                f.write('{}: {}\n'.format(fkey, fingerprint[fkey]))

    def submit(self, name: str, **kwargs) -> None:
        """Stores log.
        """
        delta = datetime.now() - self.exp_start
        kwargs['time'] = delta.total_seconds()
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
    ) -> None:
        if "time" not in indices:
            indices.append("time")
        self._summary_setting[name] = SummarySetting(indices, interval, color)

    def show_summary(self, name: str) -> None:
        indices, interval, color = self._summary_setting[name]
        df = self._store[name][-interval:].to_df()
        indices_df = df[indices]
        df.drop(columns=indices, inplace=True)
        click.secho(f"=========={name.upper()} LOG===========", bg=color, fg="white", bold=True)
        min_, max_ = indices_df.iloc[0], indices_df.iloc[-1]
        range_str = ".".join([f"{idx}: {min_[idx]}-{max_[idx]} " for idx in indices])
        click.secho(range_str, underline=True, fg=color, bg="white")
        describe = df.describe()
        describe.drop(labels="count", inplace=True)
        click.echo(df.describe())

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
