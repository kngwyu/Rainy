from datetime import datetime
import json
import logging
import git
from pathlib import Path
import sys
from typing import Optional
NORMAL_FORMATTER = logging.Formatter('%(levelname)s %(asctime)s: %(name)s: %(message)s')
JSON_FORMATTER = logging.Formatter('{"%(levelname)s": %(message)s},')
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
        log_dir = script_path.stem + '-log-'
        try:
            repo = git.Repo(script_path, search_parent_directories=True)
            head = repo.head.commit
            log_dir += head.hexsha[:8] + '-'
        finally:
            pass
        log_dir += self.exp_start.strftime("%y%m%d-%H%M%S")
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
        finger = log_dir.joinpath('fingerprint.txt')
        with open(finger.as_posix(), 'w') as f:
            f.write('{}\n'.format(self.exp_start))
            if comment:
                f.write(comment)
        handler = make_handler(Path(log_dir).joinpath('log.txt'), EXP)
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
        For structured logging, only dict is enabled as argument
        """
        if self.isEnabledFor(EXP):
            delta = datetime.now() - self.exp_start
            msg['elapsed-time'] = delta.total_seconds()
            msg['name'] = name
            self._log(EXP, json.dumps(msg), args, **kwargs)
