import click
from typing import Optional
from .utils.log import ExperimentLog


@click.command()
@click.option('--log-dir', type=str)
@click.option('--vi-mode', is_flag=True)
def ipython(log_dir: Optional[str], vi_mode: bool) -> None:
    if log_dir is not None:
        log = ExperimentLog(log_dir)  # noqa
    else:
        open_log = ExperimentLog  # noqa
    try:
        from ptpython.ipython import embed
        import matplotlib as mpl
        mpl.use('TkAgg')
        from matplotlib import pyplot as plt  # noqa
        import rainy  # noqa
        del log_dir
        embed(vi_mode=vi_mode)
    except ImportError:
        print("To use ipython mode, install ipython and ptpython first.")


if __name__ == '__main__':
    ipython()
