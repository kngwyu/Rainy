import traceback
from typing import Callable


def pretty_loop(worker_id: int, loop_fn: Callable[[], None]) -> None:
    try:
        loop_fn()
    except KeyboardInterrupt:
        print(f"Worker {worker_id} ended by KeyboardInterruption.")
    except Exception as e:
        traceback.print_exc()
        print("Exception in worker process %i", worker_id)
        raise e
