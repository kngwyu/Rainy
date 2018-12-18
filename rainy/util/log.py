from datetime import datetime
import json
import logging
import git
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set
NORMAL_FORMATTER = logging.Formatter('%(levelname)s %(asctime)s: %(name)s: %(message)s')
JSON_FORMATTER = logging.Formatter('%(levelname)s::%(message)s')
FINGERPRINT = 'fingerprint.txt'
LOGFILE = 'log.txt'
EXP = 5
logging.addLevelName(EXP, 'EXP')


class Logger(logging.Logger):
    def __init__(self) -> None:
        # set log level to debug
        super().__init__('rainy', EXP)
        self._log_dir: Optional[Path] = None
        self.exp_start = datetime.now()

    def set_dir_from_script_path(self, script_path_: str, comment: Optional[str] = None) -> None:
        script_path = Path(script_path_)
        log_dir = script_path.stem + '-' + self.exp_start.strftime("%y%m%d-%H%M%S")
        try:
            repo = git.Repo(script_path, search_parent_directories=True)
            head = repo.head.commit
            log_dir += '-' + head.hexsha[:8]
        finally:
            pass
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            log_dir_path.mkdir()
        self.set_dir(log_dir_path, comment=comment)

    def set_dir(self, log_dir: Path, comment: Optional[str] = None) -> None:
        self._log_dir = log_dir

        def make_handler(log_path: Path, level: int) -> logging.Handler:
            if not log_path.exists():
                log_path.touch()
            handler = logging.FileHandler(log_path.as_posix())
            handler.setFormatter(JSON_FORMATTER)
            handler.setLevel(level)
            return handler
        finger = log_dir.joinpath(FINGERPRINT)
        with open(finger.as_posix(), 'w') as f:
            f.write('{}\n'.format(self.exp_start))
            if comment:
                f.write(comment)
        handler = make_handler(Path(log_dir).joinpath(LOGFILE), EXP)
        self.addHandler(handler)

    def set_stderr(self, level: int = EXP) -> None:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(NORMAL_FORMATTER)
        handler.setLevel(level)
        self.addHandler(handler)

    @property
    def log_dir(self) -> Optional[Path]:
        return self._log_dir

    def exp(self, name: str, msg: dict, *args, **kwargs) -> None:
        """
        For experiment logging. Only dict is enabled as argument
        """
        if not self.isEnabledFor(EXP):
            return
        delta = datetime.now() - self.exp_start
        msg['elapsed-time'] = delta.total_seconds()
        msg['name'] = name
        self._log(EXP, json.dumps(msg), args, **kwargs)  # type: ignore


def _load_log_file(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path.as_posix()) as f:
        lines = f.readlines()
    log = []
    for line in filter(lambda s: s.startswith('EXP::'), lines):
        log.append(json.loads(line[5:]))
    return log


class LogWrapper:
    """Wrapper of filterd log.
    """
    def __init__(self, name: str, inner: List[Dict[str, Any]]) -> None:
        self.name = name
        self.inner = inner
        self._available_keys = set()

    @property
    def unwrapped(self) -> List[Dict[str, Any]]:
        return self.inner

    def keys(self) -> Set[str]:
        if not self._available_keys:
            for log in self.inner:
                for key in log:
                    self._available_keys.add(key)
        return self._available_keys

    def get_values(self, key: str) -> List[Any]:
        if key not in self.inner[0]:
            raise ValueError('LogWrapper({}) has no logging key {}'.format(self.name, key))
        return list(map(lambda d: d[key], self.inner))

    def update_steps(self) -> List[int]:
        return self.get_values('update-steps')

    def episodes(self) -> List[int]:
        return self.get_values('episodes')

    def elapsed_time(self) -> List[int]:
        return self.get_values('elapsed-time')


class ExperimentLog:
    """Structured log file.
       Used to get graphs or else from rainy log files.
    """
    def __init__(self, file_or_dir_name: str) -> None:
        path = Path(file_or_dir_name)
        if path.is_dir():
            log_path = path.joinpath(LOGFILE)
            self.fingerprint = path.joinpath(FINGERPRINT).read_text()
        else:
            log_path = path
            self.fingerprint = ''
        self.log = _load_log_file(log_path)
        self._available_keys = set()
        self.log_path = log_path

    def keys(self) -> Set[str]:
        if not self._available_keys:
            for log in self.log:
                self._available_keys.add(log['name'])
        return self._available_keys

    def get_log(self, key: str) -> LogWrapper:
        return LogWrapper(key, list(filter(lambda log: log['name'] == key, self.log)))
