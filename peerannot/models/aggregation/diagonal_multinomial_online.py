from __future__ import annotations

import numpy as np
import sparse as sp
from line_profiler import profile
from pydantic import validate_call
from pymongo import UpdateOne

from peerannot.models.aggregation.mongo_online_helpers import (
    EStepResult,
    MStepResult,
    SparseMongoOnlineAlgorithm,
)
from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class DiagonalMultinomialOnline(OnlineAlgorithm):
    @validate_call
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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
    using sparse matrices and mongo.

    Assumptions:

    - workers are independente

    - each worker is only characterized by their reliability in recognizing the correct class
        (stores only diagonal of the full confusion matrix for each worker)

    - all errors (misclassifications) are uniformly distributed among the incorrect classes


    """

    @validate_call
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @profile
    def _m_step(
        self,
        batch_matrix: sp.COO,  # shape: (n_tasks, n_workers, n_classes)
        batch_T: sp.COO,  # shape: (n_tasks, n_classes)
    ) -> MStepResult:
        batch_rho = batch_T.mean(axis=0)
        batch_n_classes = batch_matrix.shape[2]

        pij_all = np.tensordot(
            batch_T.T,
            batch_matrix.transpose((1, 0, 2)),
            axes=([1], [1]),
        )

        denom_all = pij_all.sum(axis=2)
        denom_all_safe = np.where(denom_all > 0, denom_all, 1e-9)

        indices = np.arange(batch_n_classes)
        batch_pi = (pij_all[indices, :, indices] / denom_all_safe).T

        return MStepResult(batch_rho, batch_pi)

    @profile
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: sp.COO,
    ) -> None:
        """
        vectorized update of worker confusion matrices using sparse COO arrays.

        Arguments:
            worker_mapping: dict mapping worker_id -> batch row index
            class_mapping: dict mapping class_name -> global class_id
            batch_pi: sparse.COO of shape (n_workers, n_classes)
        """
        worker_ids = list(worker_mapping.keys())
        n_workers, n_classes = batch_pi.shape

        worker_idx_map = {wid: i for i, wid in enumerate(worker_ids)}
        global_class_ids = np.fromiter(
            self._reverse_class_mapping.keys(),
            dtype=int,
        )
        batch_class_indices = np.arange(len(global_class_ids), dtype=int)

        batch_worker_indices = np.fromiter(
            worker_mapping.values(),
            dtype=int,
        )

        conf_matrix = np.zeros(
            (n_workers, len(self._reverse_class_mapping)),
            dtype=np.float64,
        )

        proj = {f"confusion_matrix.{cls}": 1 for cls in class_mapping.keys()}

        worker_docs = self.db.worker_confusion_matrices.find(
            {"_id": {"$in": worker_ids}},
            projection=proj,
        )
        for doc in worker_docs:
            wid = doc["_id"]
            i = worker_idx_map[wid]
            conf_dict = doc.get("confusion_matrix", {})
            for cls_name, prob in conf_dict.items():
                cid = class_mapping.get(cls_name)
                conf_matrix[i, cid] = prob

        coords = batch_pi.coords  # shape (2, nnz)
        data = batch_pi.data

        row_map = {w: i for i, w in enumerate(batch_worker_indices)}
        col_map = {c: j for j, c in enumerate(batch_class_indices)}

        mask = np.isin(coords[0], batch_worker_indices) & np.isin(
            coords[1],
            batch_class_indices,
        )
        sub_rows = coords[0, mask]
        sub_cols = coords[1, mask]
        sub_data = data[mask]

        rows = np.fromiter(
            (row_map[r] for r in sub_rows),
            dtype=int,
            count=len(sub_rows),
        )
        cols = np.fromiter(
            (col_map[c] for c in sub_cols),
            dtype=int,
            count=len(sub_cols),
        )

        batch_probs = np.zeros(
            (n_workers, len(batch_class_indices)),
            dtype=sub_data.dtype,
        )
        batch_probs[rows, cols] = sub_data

        conf_matrix[:, global_class_ids] = (1 - self.gamma) * conf_matrix[
            :,
            global_class_ids,
        ] + self.gamma * batch_probs

        row_idx, col_idx = np.nonzero(conf_matrix)
        probs = conf_matrix[row_idx, col_idx]

        order = np.argsort(row_idx)
        row_idx, col_idx, probs = row_idx[order], col_idx[order], probs[order]

        unique_rows, start_idx, counts = np.unique(
            row_idx,
            return_index=True,
            return_counts=True,
        )

        updates = []
        for r, start, count in zip(unique_rows, start_idx, counts):
            wid = worker_ids[r]
            cids = col_idx[start : start + count]
            vals = probs[start : start + count]

            new_conf = {
                self._reverse_class_mapping[int(cid)]: float(p)
                for cid, p in zip(cids, vals, strict=True)
            }

            updates.append(
                UpdateOne(
                    {"_id": wid},
                    {"$set": {"confusion_matrix": new_conf}},
                    upsert=True,
                ),
            )

        if updates:
            self.db.worker_confusion_matrices.bulk_write(updates)

    @profile
    def _e_step(
        self,
        batch_matrix: sp.COO,
        batch_pi: sp.COO | np.array,
        batch_rho: sp.COO,
    ) -> EStepResult:
        batch_n_classes = batch_matrix.shape[2]

        denom = np.maximum(batch_n_classes - 1, 1)
        batch_pi_non_diag_values = np.where(
            batch_n_classes > 1,
            (np.ones_like(batch_pi) - batch_pi) / denom,
            0.0,
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

        # Compute sum_off_diag: sum over classes l != j of batch_matrix[i,k,l]
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

        return EStepResult(batch_T, batch_denom_e_step)

    @property
    def pi(self) -> np.ndarray:
        pi = np.zeros((self.n_workers, self.n_classes), dtype=np.float64)

        projection = {"_id": 1, "confusion_matrix": 1}

        cursor = self.db.worker_confusion_matrices.find({}, projection)
        for doc in cursor:
            worker_id = doc["_id"]
            conf = doc.get("confusion_matrix", {})
            worker_doc = self.worker_mapping.find_one({"_id": worker_id})
            if worker_doc is None:
                continue
            worker_idx = worker_doc.get("index")
            for class_name, prob in conf.items():
                class_doc = self.class_mapping.find_one(
                    {"_id": class_name},
                )
                if class_doc is None:
                    continue
                class_idx = class_doc.get("index")
                pi[worker_idx, class_idx] = float(prob)

        return pi

    def build_full_pi_tensor(self) -> np.ndarray:
        pi = self.pi
        n_workers, n_classes = pi.shape

        # Off-diagonal probability for each worker and class (row-based)
        off_diag = (1.0 - pi) / (n_classes - 1)  # (n_workers, n_classes)

        # Broadcast  full_pi[w, i, j] = off_diag[w, i] for all j
        full_pi = np.broadcast_to(
            off_diag[:, :, None],
            (n_workers, n_classes, n_classes),
        ).copy()

        # Replace diagonals with pi
        idx = np.arange(n_classes)
        full_pi[:, idx, idx] = pi

        return full_pi

    def build_batch_pi_tensor(
        self,
        batch_pi: np.ndarray,
        class_mapping: ClassMapping,
        worker_mapping: WorkerMapping,
    ) -> np.ndarray:
        pi = batch_pi
        n_workers, _ = pi.shape
        n_classes = len(class_mapping)

        off_diag = (1.0 - pi) / (n_classes - 1)

        pi_batch = np.broadcast_to(
            off_diag[:, :, None],
            (n_workers, n_classes, n_classes),
        ).copy()

        idx = np.arange(n_classes)
        pi_batch[:, idx, idx] = pi

        return pi_batch
