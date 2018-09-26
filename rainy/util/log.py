import datetime
import logging
import git
from pathlib import Path
import sys
from typing import Optional
DEFAULT_FORMATTER = logging.Formatter('%(levelname)s %(asctime)s: %(name)s: %(message)s')
EXP = 5
logging.addLevelName(5, 'EXP')


class Logger(logging.Logger):
    def __init__(self) -> None:
        # set log level to debug
        super().__init__('rainy', EXP)
        self._log_dir = None

    def set_dir_from_script_path(self, script_path: str) -> None:
        path = Path(script_path)
        filename = path.stem
        log_dir = filename + '-log-'
        try:
            repo = git.Repo(script_path, search_parent_directories=True)
            head = repo.head.commit
            log_dir += head.hexsha[:8] + '-'
        finally:
            pass
        log_dir += datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            log_dir_path.mkdir()
        self.set_dir(log_dir_path)

    def set_dir(self, log_dir: Path) -> None:
        self._log_dir = log_dir

        def make_handler(log_path: Path, level: int) -> logging.Handler:
            if not log_path.exists():
                log_path.touch()
            handler = logging.FileHandler(log_path)
            handler.setFormatter(DEFAULT_FORMATTER)
            handler.setLevel(level)
            return handler
        handler = make_handler(Path(log_dir).joinpath('log.txt'), EXP)
        self.addHandler(handler)

    def set_stderr(self, level: int = EXP) -> None:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setLevel(level)
        self.addHandler(handler)

    def log_dir(self) -> Optional[Path]:
        return self._log_dir

    def exp(self, msg, *arg, **kwargs) -> None:
        if self.isEnabledFor(EXP):
            self._log(EXP, msg, arg, **kwargs)
