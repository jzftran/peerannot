from typing import Annotated

import numpy as np
import sparse as sp
from annotated_types import Ge, Gt
from line_profiler import profile
from pydantic import validate_call

from peerannot.models.aggregation.mongo_online_helpers import (
    SparseMongoOnlineAlgorithm,
)
from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class PooledMultinomialOnline(OnlineAlgorithm):
    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        if new_n_classes > self.n_classes:
            self.pi = self._expand_array(
                self.pi,
                (new_n_classes, new_n_classes),
                fill_value=0.0,
            )

    def _initialize_pi(self) -> None:
        self.pi = np.zeros((self.n_classes, self.n_classes))

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        aggregated_votes = np.einsum(
            "tq, tij -> qj",
            batch_T,
            batch_matrix,
        )  # shape (n_classes, n_classes)
        denom = aggregated_votes.sum(axis=1, keepdims=True)

        batch_pi = np.divide(
            aggregated_votes,
            denom,
            out=np.zeros_like(aggregated_votes),
            where=denom != 0,
        )

        return batch_rho, batch_pi

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_n_tasks = batch_matrix.shape[0]
        batch_n_classes = batch_matrix.shape[2]

        T = np.zeros((batch_n_tasks, batch_n_classes))

        # use mask instead of power
        for i in range(batch_n_tasks):
            for j in range(batch_n_classes):
                num = (
                    np.prod(
                        np.power(
                            batch_pi[j, :],
                            batch_matrix[i, :, :],
                        ),
                    )
                    * batch_rho[j]
                )
                T[i, j] = num

        batch_denom_e_step = T.sum(axis=1, keepdims=True)
        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)
        return batch_T, batch_denom_e_step

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ):
        batch_to_global = {
            batch_class_idx: self.class_mapping[class_name]
            for class_name, batch_class_idx in class_mapping.items()
        }

        for i_batch, i_global in batch_to_global.items():
            for j_batch, j_global in batch_to_global.items():
                self.pi[i_global, j_global] = (1 - self.gamma) * self.pi[
                    i_global,
                    j_global,
                ] + self.gamma * batch_pi[
                    i_batch,
                    j_batch,
                ]

            row_sum = self.pi[i_global, :].sum()
            if row_sum > 0:
                self.pi[i_global, :] /= row_sum


class VectorizedPooledMultinomialOnlineMongo(SparseMongoOnlineAlgorithm):
    @validate_call
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @profile
    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: sp.COO,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        aggregated_votes = np.einsum(
            "tq, tij -> qj",
            batch_T,
            batch_matrix,
        )  # shape (n_classes, n_classes)
        denom = aggregated_votes.sum(axis=1, keepdims=True).todense()

        batch_pi = np.where(
            denom > 0,
            aggregated_votes / denom,
            aggregated_votes,
        )

        return batch_rho, batch_pi

    @profile
    def _e_step(self, batch_matrix, batch_pi, batch_rho):
        T = batch_matrix.sum(axis=1)
        pows = np.power(batch_pi[None, :, :], T[:, None, :])

        prods = pows.prod(axis=2)

        batch_T = prods * batch_rho[None, :]
        denom = batch_T.sum(axis=1, keepdims=True).todense()
        batch_T = np.where(denom > 0, batch_T / denom, batch_T)
        return batch_T, denom

    @profile
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        doc = self.db.worker_confusion_matrices.find_one(
            {"_id": self.__class__.__name__},
        )
        confusion_matrix = doc.get("confusion_matrix", []) if doc else []

        # Get all class names (from batch index -> name)
        batch_to_name = self._reverse_class_mapping

        n_classes = len(batch_to_name)

        # Reconstruct dense global matrix from sparse DB
        global_pi = np.zeros((n_classes, n_classes), dtype=np.float64)
        for entry in confusion_matrix:
            i_name, j_name = entry["from_class"], entry["to_class"]
            if i_name in class_mapping and j_name in class_mapping:
                i = class_mapping[i_name]
                j = class_mapping[j_name]
                global_pi[i, j] = float(entry["prob"])

        # Prepare indices for sparse update
        batch_indices = np.array(list(class_mapping.values()))

        gi, gj = np.meshgrid(batch_indices, batch_indices, indexing="ij")
        bi, bj = np.meshgrid(batch_indices, batch_indices, indexing="ij")

        gi_flat, gj_flat = gi.ravel(), gj.ravel()
        bi_flat, bj_flat = bi.ravel(), bj.ravel()

        batch_values = batch_pi[bi_flat, bj_flat].todense().ravel()

        mask = batch_values != 0
        gi_flat, gj_flat, batch_values = (
            gi_flat[mask],
            gj_flat[mask],
            batch_values[mask],
        )

        global_pi[gi_flat, gj_flat] = (1 - self.gamma) * global_pi[
            gi_flat,
            gj_flat,
        ] + self.gamma * batch_values

        # Normalize rows
        row_sums = global_pi.sum(axis=1, keepdims=True)
        mask_rows = row_sums > 0
        global_pi[mask_rows[:, 0]] /= row_sums[mask_rows]

        # Convert back to sparse format using class names
        new_confusion_matrix = [
            {
                "from_class": batch_to_name[i],
                "to_class": batch_to_name[j],
                "prob": float(global_pi[i, j]),
            }
            for i, j in zip(*np.nonzero(global_pi))
        ]

        with self.mongo_timer("online update global confusion_matrix"):
            self.db.worker_confusion_matrices.update_one(
                {"_id": self.__class__.__name__},
                {"$set": {"confusion_matrix": new_confusion_matrix}},
                upsert=True,
            )

    @property
    def pi(self) -> np.ndarray:
        raise NotImplementedError
