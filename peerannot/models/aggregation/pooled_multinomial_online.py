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

        batch_names = list(class_mapping.keys())

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

        old_rows = []
        old_cols = []
        old_data = []

        existing_from_set = set()
        for doc in docs:
            _id = doc["_id"]

            from_name = _id.get("from_class") if isinstance(_id, dict) else _id
            if from_name not in class_mapping:
                continue
            r = class_mapping[from_name]
            existing_from_set.add(from_name)
            probs = doc.get("probs", {}) or {}
            for to_name, prob in probs.items():
                if to_name not in class_mapping:
                    continue
                c = class_mapping[to_name]
                old_rows.append(r)
                old_cols.append(c)
                old_data.append(float(prob))

        # old_rows = np.asarray(old_rows, dtype=int)
        # old_cols = np.asarray(old_cols, dtype=int)
        # old_data = np.asarray(old_data, dtype=float)

        bp_rows, bp_cols = batch_pi.coords

        rows_concat = np.concatenate([old_rows, bp_rows])
        cols_concat = np.concatenate([old_cols, bp_cols])
        data_concat = np.concatenate([old_data, batch_pi.data])

        # sum duplicates by (row, col) pairs
        # Create structured keys for lexsort: sort by (row, col)
        order = np.lexsort((cols_concat, rows_concat))
        rows_sorted = rows_concat[order]
        cols_sorted = cols_concat[order]
        data_sorted = data_concat[order]

        # compute mask of new groups
        change_mask = np.empty(rows_sorted.shape, dtype=bool)
        change_mask[0] = True
        if rows_sorted.size > 1:
            change_mask[1:] = (rows_sorted[1:] != rows_sorted[:-1]) | (
                cols_sorted[1:] != cols_sorted[:-1]
            )
        # indices where groups start
        starts = np.nonzero(change_mask)[0]
        # ends are next start or len
        ends = np.concatenate([starts[1:], np.array([rows_sorted.size])])

        uniq_rows = rows_sorted[starts]
        uniq_cols = cols_sorted[starts]
        # sum data within runs
        uniq_data = np.empty(starts.size, dtype=float)
        for i, (s, e) in enumerate(zip(starts, ends)):
            uniq_data[i] = data_sorted[s:e].sum()

        per_row = {}
        for r, c, v in zip(uniq_rows, uniq_cols, uniq_data):
            per_row.setdefault(int(r), []).append((int(c), float(v)))

        updates = []
        for from_name in batch_names:
            r = class_mapping[from_name]
            entries = per_row.get(r, [])
            probs_dict = (
                {list(class_mapping.keys())[c]: p for c, p in entries}
                if entries
                else {}
            )
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
