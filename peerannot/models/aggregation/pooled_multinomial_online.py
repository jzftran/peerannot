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
        model_name = self.__class__.__name__
        batch_names = list(class_mapping.keys())

        for from_name in batch_names:
            from_idx = class_mapping[from_name]

            doc = self.db.worker_confusion_rows.find_one(
                {"_id": {"model": model_name, "from_class": from_name}},
            )
            probs = doc.get("probs", {}) if doc else {}

            updated = {}
            for to_name, to_idx in class_mapping.items():
                old_prob = probs.get(to_name, 0.0)
                batch_val = float(batch_pi[from_idx, to_idx])
                new_prob = (1 - self.gamma) * old_prob + self.gamma * batch_val
                updated[to_name] = new_prob

            merged = {**probs, **updated}

            # Normalize row
            total = sum(merged.values())
            if total > 0:
                merged = {k: v / total for k, v in merged.items()}

            self.db.worker_confusion_rows.update_one(
                {"_id": {"model": model_name, "from_class": from_name}},
                {"$set": {"probs": merged}},
                upsert=True,
            )

    @property
    def pi(self) -> np.ndarray:
        class_mapping_docs = list(
            self.class_mapping.find({}, {"_id": 1, "index": 1}),
        )
        name_to_idx = {doc["_id"]: doc["index"] for doc in class_mapping_docs}

        n_classes = len(name_to_idx)
        pi_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)

        cursor = self.db.worker_confusion_rows.find(
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
