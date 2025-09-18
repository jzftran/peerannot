import time
from contextlib import contextmanager

from loguru import logger


class OnlineMongoLoggingMixin:
    """Provides monitoring utilities for EM iterations, MongoDB I/O, and batches."""

    @contextmanager
    def mongo_timer(self, op_name: str):
        """Context manager for timing MongoDB operations."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        logger.debug(f"[Mongo] {op_name} took {elapsed:.3f}s")

    def log_em_iter(
        self,
        i: int,
        likeli: float,
        delta: float,
        iter_time: float,
    ):
        """Log EM iteration diagnostics."""
        logger.debug(
            f"[EM] Iter {i:03d} | L={likeli:.6f} | delta={delta:.2e} | {iter_time:.3f}s",
        )

    def log_batch_summary(
        self,
        t: int,
        n_tasks: int,
        n_workers: int,
        n_classes: int,
        n_iters: int,
        batch_time: float,
        final_ll: float,
    ):
        """Log summary after processing a batch."""
        logger.info(
            f"[Batch {t}] | tasks={n_tasks} | workers={n_workers} | classes={n_classes} "
            f"| iters={n_iters} | time={batch_time:.2f}s | final L={final_ll:.6f}",
        )
