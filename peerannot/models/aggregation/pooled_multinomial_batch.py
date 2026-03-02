from typing import Annotated

import numpy as np
import sparse as sp
from annotated_types import Ge, Gt
from line_profiler import profile
from pydantic import validate_call
from pymongo import UpdateOne

from peerannot.models.aggregation.mongo_online_helpers import (
    SparseMongoBatchAlgorithm,
)
from peerannot.models.aggregation.online_helpers import BatchAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class PooledMultinomialBatch(BatchAlgorithm):
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


class VectorizedPooledMultinomialBatchMongo(SparseMongoBatchAlgorithm):
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
        # is buggy:
        # aggregated_votes = np.einsum(
        #     "tq, tij -> qj",
        #     batch_T,
        #     batch_matrix,
        # )  # shape (n_classes, n_classes)
        aggregated_votes = sp.tensordot(
            batch_T,
            batch_matrix,
            axes=(0, 0),
        ).sum(axis=1)

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

        prod = prods * batch_rho[None, :]

        batch_denom_e_step = prod.sum(axis=1, keepdims=True)

        if not np.any(batch_denom_e_step == 0):
            batch_denom_e_step = batch_denom_e_step.todense()

        batch_T = np.where(
            batch_denom_e_step > 0,
            prod / batch_denom_e_step,
            prod,
        )

        return batch_T, batch_denom_e_step

    @profile
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        model_name = self.__class__.__name__
        gamma = self.gamma

        batch_names = list(class_mapping.keys())
        n_classes = len(class_mapping)

        cursor = self.db.worker_confusion_matrices.find(
            {
                "_id": {
                    "$in": [
                        {"model": model_name, "from_class": name}
                        for name in batch_names
                    ],
                },
            },
        )
        docs = list(cursor)

        old_matrix = np.zeros((n_classes, n_classes), dtype=float)

        for doc in docs:
            _id = doc["_id"]
            from_name = _id["from_class"]
            if from_name not in class_mapping:
                continue

            r = class_mapping[from_name]
            probs = doc.get("probs", {}) or {}

            for to_name, prob in probs.items():
                if to_name in class_mapping:
                    c = class_mapping[to_name]
                    old_matrix[r, c] = float(prob)

        batch_matrix = np.zeros((n_classes, n_classes), dtype=float)

        if hasattr(batch_pi, "coords"):
            rows, cols = batch_pi.coords
            batch_matrix[rows, cols] = batch_pi.data
        else:
            batch_matrix = np.asarray(batch_pi, dtype=float)

        new_matrix = (1.0 - gamma) * old_matrix + gamma * batch_matrix

        row_sums = new_matrix.sum(axis=1, keepdims=True)
        nonzero_mask = row_sums.squeeze() > 0
        new_matrix[nonzero_mask] /= row_sums[nonzero_mask]

        updates = []

        for from_name in batch_names:
            r = class_mapping[from_name]
            row = new_matrix[r]

            probs_dict = {
                self._reverse_class_mapping[c]: float(p)
                for c, p in enumerate(row)
                if p > 0.0
            }

            updates.append(
                UpdateOne(
                    {"_id": {"model": model_name, "from_class": from_name}},
                    {"$set": {"probs": probs_dict}},
                    upsert=True,
                ),
            )

        if updates:
            with self.mongo_timer("online update worker confusion matrices"):
                self.db.worker_confusion_matrices.bulk_write(
                    updates,
                    ordered=False,
                )

    @property
    def pi(self) -> np.ndarray:
        class_mapping_docs = list(
            self.class_mapping.find({}, {"_id": 1, "index": 1}),
        )
        name_to_idx = {doc["_id"]: doc["index"] for doc in class_mapping_docs}

        n_classes = len(name_to_idx)
        pi_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)

        cursor = self.db.worker_confusion_matrices.find(
            {"_id.model": self.__class__.__name__},
        )
        for doc in cursor:
            from_class = doc["_id"]["from_class"]
            from_idx = name_to_idx.get(from_class)
            if from_idx is None:
                continue

            probs = doc.get("probs", {})
            for to_class, prob in probs.items():
                to_idx = name_to_idx.get(to_class)
                if to_idx is not None:
                    pi_matrix[from_idx, to_idx] = float(prob)

        return pi_matrix

    def build_full_pi_tensor(self) -> np.ndarray:
        full_pi = np.broadcast_to(
            self.pi,
            (self.n_workers, self.n_classes, self.n_classes),
        ).copy()

        return full_pi

    def build_batch_pi_tensor(
        self,
        batch_pi: np.ndarray,
        class_mapping: ClassMapping,
        worker_mapping: WorkerMapping,
    ) -> np.ndarray:
        n_workers = len(worker_mapping)
        n_classes = len(class_mapping)
        full_pi = np.broadcast_to(
            batch_pi,
            (n_workers, n_classes, n_classes),
        ).copy()

        return full_pi


m = VectorizedPooledMultinomialBatchMongo()
m.drop()
batch1 = {0: {0: 0}, 1: {1: 1}, 2: {2: 0}}
batch2 = {0: {3: 1, 4: 1}, 3: {2: 1, 4: 0}, 4: {2: 1, 4: 2}}

m.process_batch(batch1)
m.process_batch(batch2)
m.pi
