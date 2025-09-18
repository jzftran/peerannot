import numpy as np
import sparse as sp
from line_profiler import profile
from pydantic import validate_call
from pymongo import UpdateOne

from peerannot.models.aggregation.mongo_online_helpers import (
    SparseMongoOnlineAlgorithm,
)
from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class MultinomialBinaryOnline(OnlineAlgorithm):
    @validate_call
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        """Expand the pi array if the number of workers or classes increases."""
        if new_n_workers > self.n_workers:
            self.pi = self._expand_array(
                self.pi,
                (new_n_workers, 1),
                fill_value=0.0,
            )

    def _initialize_pi(self) -> None:
        self.pi = np.zeros((self.n_workers, 1))

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        # Update only workers present in the batch
        for worker, batch_worker_idx in worker_mapping.items():
            worker_idx = self.worker_mapping[worker]
            if self.pi[worker_idx] == 0:
                self.pi[worker_idx] = batch_pi[batch_worker_idx,]
            else:
                self.pi[worker_idx] = (1 - self.gamma) * self.pi[
                    worker_idx,
                ] + self.gamma * batch_pi[batch_worker_idx,]

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)
        batch_n_workers = batch_matrix.shape[1]

        batch_pi = np.zeros(batch_n_workers)
        for j in range(batch_n_workers):
            labeled_count = batch_matrix[:, j, :].sum()
            weighted_sum = (batch_T * batch_matrix[:, j, :]).sum()
            alpha = np.divide(
                weighted_sum,
                labeled_count,
                out=np.zeros_like(weighted_sum),
                where=labeled_count > 0,
            )
            batch_pi[j] = alpha
        return batch_rho, batch_pi

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_n_tasks = batch_matrix.shape[0]
        batch_n_classes = batch_matrix.shape[2]

        off_diag_alpha = (np.ones_like(batch_pi) - batch_pi) / (
            batch_n_classes - 1
        )
        T = np.zeros((batch_n_tasks, batch_n_classes))
        for i in range(batch_n_tasks):
            for j in range(batch_n_classes):
                diag_contrib = np.prod(
                    np.power(
                        batch_pi,
                        batch_matrix[i, :, j],
                    ),
                )  # shape (n_workers,)

                mask = np.ones(batch_n_classes, dtype=bool)
                mask[j] = False
                off_diag_labels = batch_matrix[i, :, mask]

                off_diag_contrib = np.prod(
                    np.power(off_diag_alpha, off_diag_labels),
                )

                T[i, j] = (
                    np.prod(diag_contrib * off_diag_contrib) * batch_rho[j]
                )

        batch_denom_e_step = T.sum(1, keepdims=True)

        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)
        return batch_T, batch_denom_e_step


# %%
class VectorizedMultinomialBinaryOnlineMongo(
    SparseMongoOnlineAlgorithm,
):
    """Vectorized pooled diagonal multinomial binary online algorithm
    using sparse matrices and mongo."""

    @validate_call
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @profile
    def _m_step(
        self,
        batch_matrix: np.ndarray,  # shape: (n_samples, n_workers, n_classes)
        batch_T: np.ndarray,  # shape: (n_samples, n_classes)
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        # shape: (n_samples, n_workers)

        weighted = np.sum(
            np.multiply(batch_T[:, None, :], batch_matrix),
            axis=2,
        )

        # Sum over samples to get total weighted contribution per worker
        weighted_sums = np.sum(
            weighted,
            axis=0,
        )  # shape: (n_workers,)

        labeled_counts = np.sum(
            batch_matrix,
            axis=(0, 2),
        ).todense()  # shape: (n_workers,)

        # weighted_sums = weighted_sums.todense()
        batch_pi = np.where(
            labeled_counts > 0,
            weighted_sums / labeled_counts,
            weighted_sums,
        )

        return batch_rho, batch_pi

    @profile
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        # Fetch all existing confusion matrices for these workers
        worker_ids = list(worker_mapping.keys())
        worker_docs = self.db.worker_confusion_matrices.find(
            {"_id": {"$in": worker_ids}},
        )
        worker_confusions = {
            doc["_id"]: doc.get("confusion_matrix", None)
            for doc in worker_docs
        }

        updates = []

        for worker_id, batch_worker_idx in worker_mapping.items():
            pi_new_value = batch_pi[batch_worker_idx]
            existing_entry = worker_confusions.get(worker_id)
            if existing_entry is None:
                # No previous pi — store it directly
                updates.append(
                    UpdateOne(
                        {"_id": worker_id},
                        {"$set": {"confusion_matrix": {"prob": pi_new_value}}},
                        upsert=True,
                    ),
                )
            else:
                pi_old_value = existing_entry["prob"]

                # Online update
                pi_updated = (
                    1 - self.gamma
                ) * pi_old_value + self.gamma * pi_new_value
                updates.append(
                    UpdateOne(
                        {"_id": worker_id},
                        {"$set": {"confusion_matrix": {"prob": pi_updated}}},
                        upsert=True,
                    ),
                )

        if updates:
            self.db.worker_confusion_matrices.bulk_write(updates)

    @profile
    def _e_step(
        self,
        batch_matrix: sp.COO,
        batch_pi: sp.COO,
        batch_rho: sp.COO,
    ) -> tuple[sp.COO, np.ndarray]:
        n_tasks, _, n_classes = batch_matrix.shape

        # Compute per-worker off-diagonal probabilities
        off_diag_alpha = (1.0 - batch_pi) / (
            n_classes - 1
        )  # shape: (n_workers,)

        # Extract COO coords & counts
        tasks, workers, assigned_classes = batch_matrix.coords
        counts = batch_matrix.data

        # Shape: (nnz, n_classes),  probability for each possible true_class
        match_mask = assigned_classes[:, None] == np.arange(n_classes)[None, :]

        probs_nnz = (
            np.where(
                match_mask,
                batch_pi[workers][:, None],
                off_diag_alpha[workers][:, None],
            )
            ** counts[:, None]
        )

        likelihood = np.ones((n_tasks, n_classes), dtype=np.float64)

        np.multiply.at(
            likelihood,
            (tasks[:, None], np.arange(n_classes)[None, :]),
            probs_nnz,
        )
        T = likelihood * batch_rho[None, :]
        denom = T.sum(axis=1, keepdims=True).todense()

        batch_T = np.where(denom > 0, T / denom, T)

        return batch_T, denom

    @property
    def pi(self) -> np.ndarray:
        raise NotImplementedError
