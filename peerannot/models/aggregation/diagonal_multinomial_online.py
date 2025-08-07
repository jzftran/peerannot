# %%
from typing import Annotated

import numpy as np
import sparse as sp
from annotated_types import Ge, Gt
from line_profiler import profile
from pydantic import validate_call
from pymongo import UpdateOne

from peerannot.models.aggregation.mongo_online_helpers import (
    SparseMongoOnlineAlgorithm,
)
from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class DiagonalMultinomialOnline(OnlineAlgorithm):
    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        """Expand the pi array if the number of workers or classes increases."""
        if new_n_workers > self.n_workers or new_n_classes > self.n_classes:
            self.pi = self._expand_array(
                self.pi,
                (new_n_workers, new_n_classes),
                fill_value=0.0,
            )

    def _initialize_pi(self) -> None:
        self.pi = np.zeros((self.n_workers, self.n_classes))

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        # For each class in the batch, map batch class idx to global class idx
        batch_to_global = {
            batch_class_idx: self.class_mapping[class_name]
            for class_name, batch_class_idx in class_mapping.items()
        }
        # Update only workers present in the batch
        for worker, batch_worker_idx in worker_mapping.items():
            worker_idx = self.worker_mapping[worker]

            for i_batch, i_global in batch_to_global.items():
                # for j_batch, j_global in batch_to_global.items():
                self.pi[worker_idx, i_global] = (1 - self.gamma) * self.pi[
                    worker_idx,
                    i_global,
                ] + self.gamma * batch_pi[
                    batch_worker_idx,
                    i_batch,
                ]

    @profile
    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        batch_n_workers = batch_matrix.shape[1]
        batch_n_classes = batch_matrix.shape[2]

        batch_pi = np.zeros((batch_n_workers, batch_n_classes))

        for j in range(batch_n_classes):
            # TODO @jzftran: change this to calculate diagonal faster
            pij = batch_T[:, j] @ batch_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)

            diag_values = pij[:, j] / np.where(denom > 0, denom, 1e-9)
            batch_pi[:, j] = diag_values

        # pi shape (n_workers, n_class), reresents how sure worker is
        # sure that the label j is true
        return batch_rho, batch_pi

    @profile
    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_n_tasks = batch_matrix.shape[0]
        batch_n_classes = batch_matrix.shape[2]

        T = np.zeros((batch_n_tasks, batch_n_classes))

        batch_pi_non_diag_values = (np.ones_like(batch_pi) - batch_pi) / (
            batch_n_classes - 1
        )

        for i in range(batch_n_tasks):
            for j in range(batch_n_classes):
                # Get all worker labels for task i (n_workers, n_classes)
                worker_labels = batch_matrix[i]  # shape (n_workers, n_classes)

                # Vectorized computation for all workers simultaneously
                # Diagonal contributions: pi[k,j]^worker_labels[k,j]
                diag_contrib = np.power(
                    batch_pi[:, j],
                    worker_labels[:, j],
                )  # shape (n_workers,)

                # Off-diagonal contributions: product over l≠j of
                # pi_non_diag[k,j]^worker_labels[k,l]
                mask = np.ones(batch_n_classes, dtype=bool)
                mask[j] = False  # exclude current class j
                off_diag_labels = worker_labels[
                    :,
                    mask,
                ]  # shape (n_workers, n_classes-1)

                off_diag_contrib = np.prod(
                    np.power(
                        batch_pi_non_diag_values[:, j][:, np.newaxis],
                        off_diag_labels,
                    ),
                    axis=1,
                )  # shape (n_workers,)

                worker_probs = (
                    diag_contrib * off_diag_contrib
                )  # shape (n_workers,)
                T[i, j] = np.prod(worker_probs) * batch_rho[j]
        batch_denom_e_step = T.sum(1, keepdims=True)
        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)

        return batch_T, batch_denom_e_step


class VectorizedDiagonalMultinomialOnlineMongo(
    SparseMongoOnlineAlgorithm,
):
    """Vectorized pooled diagonal multinomial binary online algorithm
    using sparse matrices and mongo."""

    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

    @profile
    def _m_step(
        self,
        batch_matrix: sp.COO,  # shape: (n_tasks, n_workers, n_classes)
        batch_T: np.ndarray,  # shape: (n_tasks, n_classes)
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        batch_n_classes = batch_matrix.shape[2]

        pij_all = np.einsum(
            "jb,wbc->jwc",
            batch_T.T,
            batch_matrix.transpose((1, 0, 2)),
        )
        denom_all = pij_all.sum(axis=2)
        denom_all_safe = np.where(denom_all > 0, denom_all, 1e-9)
        indices = np.arange(batch_n_classes)
        batch_pi = (pij_all[indices, :, indices] / denom_all_safe).T
        # pi shape (n_workers, n_class), reresents how sure worker is
        # sure that the label j is true
        return batch_rho, batch_pi

    @profile
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        gamma = self.gamma
        worker_ids = list(worker_mapping.keys())

        batch_to_global = {
            batch_idx: class_mapping[class_name]
            for class_name, batch_idx in class_mapping.items()
        }

        worker_docs = self.db.worker_confusion_matrices.find(
            {"_id": {"$in": worker_ids}},
        )
        worker_confusions = {
            doc["_id"]: doc.get("confusion_matrix", []) for doc in worker_docs
        }

        updates = []

        for worker_id, batch_worker_idx in worker_mapping.items():
            existing_matrix = worker_confusions.get(worker_id, [])

            entry_map = {
                entry["class_id"]: entry
                for entry in existing_matrix
                if "class_id" in entry
            }

            updated_entries = {}

            for batch_class_idx, global_class_id in batch_to_global.items():
                prob = float(batch_pi[batch_worker_idx, batch_class_idx])
                if prob == 0:
                    continue

                if global_class_id in entry_map:
                    old_prob = entry_map[global_class_id]["prob"]
                    new_prob = (1 - gamma) * old_prob + gamma * prob
                else:
                    new_prob = gamma * prob

                updated_entries[global_class_id] = new_prob

            if not updated_entries:
                continue

            new_confusion_matrix = []

            seen = set(updated_entries.keys())
            for class_id, prob in updated_entries.items():
                new_confusion_matrix.append(
                    {
                        "class_id": class_id,
                        "prob": prob,
                    },
                )

            for entry in existing_matrix:
                cid = entry["class_id"]
                if cid not in seen:
                    new_confusion_matrix.append(entry)

            updates.append(
                UpdateOne(
                    {"_id": worker_id},
                    {"$set": {"confusion_matrix": new_confusion_matrix}},
                    upsert=True,
                ),
            )

        if updates:
            self.db.worker_confusion_matrices.bulk_write(updates)

    @profile
    def _e_step(
        self,
        batch_matrix: sp.COO,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_n_classes = batch_matrix.shape[2]

        batch_pi_non_diag_values = (np.ones_like(batch_pi) - batch_pi) / (
            batch_n_classes - 1
        )

        # Expand batch_pi and batch_pi_non_diag_values to include the tasks dimension
        batch_pi_expanded = batch_pi[
            np.newaxis,
            :,
            :,
        ]  # shape (1, n_workers, n_classes) -> broadcast to (n_tasks, n_workers, n_classes)
        batch_pi_non_diag_expanded = batch_pi_non_diag_values[
            np.newaxis,
            :,
            :,
        ]

        # batch_pi_expanded[i,k,j] ** batch_matrix[i,k,j]
        diag_contrib = np.power(
            batch_pi_expanded,
            batch_matrix,
        )  # shape (n_tasks, n_workers, n_classes)

        # Compute sum_off_diag: sum over classes l ≠ j of batch_matrix[i,k,l]
        sum_over_classes = batch_matrix.sum(
            axis=2,
        )  # shape (n_tasks, n_workers)
        sum_off_diag = (
            sum_over_classes[:, :, np.newaxis] - batch_matrix
        )  # shape (n_tasks, n_workers, n_classes)

        # batch_pi_non_diag_expanded[i,k,j] ** sum_off_diag[i,k,j]
        off_diag_contrib = np.power(
            batch_pi_non_diag_expanded,
            sum_off_diag,
        )  # shape (n_tasks, n_workers, n_classes)

        # product of diag and off_diag contributions
        worker_contrib = (
            diag_contrib * off_diag_contrib
        )  # shape (n_tasks, n_workers, n_classes)

        # Compute product over workers for each (i,j)
        products_over_workers = worker_contrib.prod(
            axis=1,
        )

        T = products_over_workers * batch_rho[np.newaxis, :]

        batch_denom_e_step = T.sum(1, keepdims=True).todense()
        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)

        return batch_T, batch_denom_e_step
