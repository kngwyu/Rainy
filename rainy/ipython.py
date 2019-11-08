import click
from typing import Optional
from .utils.log import ExperimentLog


def _open_ipython(log_dir: Optional[str]) -> None:
    if log_dir is not None:
        log = ExperimentLog(log_dir)  # noqa
    open_log = ExperimentLog  # noqa
    import rainy # noqa
    del log_dir
    try:
        import IPython
        IPython.embed(colors="Linux")
    except ImportError:
        print("To use ipython mode, please install IPython first.")


@click.command()
@click.option('--log-dir', type=str)
def ipython(log_dir: Optional[str]) -> None:
    _open_ipython(log_dir)


if __name__ == '__main__':
    ipython()
