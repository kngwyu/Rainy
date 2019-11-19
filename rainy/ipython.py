import click
from typing import Optional


def _open_ipython(logdir: Optional[str]) -> None:
    import rainy # noqa
    del logdir
    try:
        import IPython
        IPython.embed(colors="Linux")
    except ImportError:
        print("To use ipython mode, please install IPython first.")


@click.command()
@click.option('--log-dir', type=str)
def ipython(logdir: Optional[str]) -> None:
    _open_ipython(logdir)


if __name__ == '__main__':
    ipython()
