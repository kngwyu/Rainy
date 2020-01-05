import logging
import traceback
from typing import Callable


def pretty_loop(worker_id: int, loop_fn: Callable[[], None]) -> None:
    try:
        loop_fn()
    except KeyboardInterrupt:
        print(f"Worker {worker_id} ended by KeyboardInterruption.")
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e
