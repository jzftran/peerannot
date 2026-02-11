# %%
from __future__ import annotations

from typing import Annotated

import numpy as np
import sparse as sp
from annotated_types import Ge, Gt
from line_profiler import profile
from pymongo import UpdateOne

from peerannot.models.aggregation.mongo_online_helpers import (
    MongoOnlineAlgorithm,
    SparseMongoOnlineAlgorithm,
    WeightedOnlineAlgorithm,
    sparse_topk_fast,
)
from peerannot.models.aggregation.online_helpers import (
    OnlineAlgorithm,
)
from peerannot.models.aggregation.types import (
    ClassMapping,
    TaskMapping,
    WorkerMapping,
)


class VectorizedDawidSkeneOnlineMongo(SparseMongoOnlineAlgorithm):
    def _e_step(self, batch_matrix, batch_pi, batch_rho):
        powered = np.power(
            batch_pi[np.newaxis, :, :, :],
            batch_matrix[:, :, np.newaxis, :],
        )

        likelihood = powered.prod(axis=(1, 3))

        batch_T = likelihood * batch_rho[np.newaxis, :]

        batch_denom_e_step = batch_T.sum(axis=1, keepdims=True)
        nonzero = batch_denom_e_step > 0

        if not np.any(batch_denom_e_step == 0):
            batch_denom_e_step = batch_denom_e_step.todense()

        batch_T = np.where(nonzero, batch_T / batch_denom_e_step, batch_T)

        return batch_T, batch_denom_e_step

    def _m_step(self, batch_matrix, batch_T):
        batch_rho = batch_T.mean(axis=0)

        weighted = batch_T[:, None, :, None] * batch_matrix[:, :, None, :]

        pij = weighted.sum(axis=0)

        denom = pij.sum(axis=2)

        # maybe better remove safe denom?
        safe = np.where(denom <= 0, -1e9, denom)[..., None]

        batch_pi = pij / safe

        return batch_rho, batch_pi

    def _load_pi_for_worker(self, worker_id: str) -> np.ndarray:
        """
        Load worker confusion matrix stored with class-name keys.
        Convert class names to indices using the class_mapping collection.
        """
        doc = self.db.worker_confusion_matrices.find_one({"_id": worker_id})
        n_classes = self.n_classes

        # Output dense result
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=float)

        if not doc or "confusion_matrix" not in doc:
            return confusion_matrix

        used_class_names = set()
        for entry in doc["confusion_matrix"]:
            used_class_names.add(entry["from_class"])
            used_class_names.add(entry["to_class"])

        cursor = self.db.class_mapping.find(
            {"_id": {"$in": list(used_class_names)}},
            {"index": 1},
        )

        name_to_idx = {doc["_id"]: doc["index"] for doc in cursor}

        for entry in doc["confusion_matrix"]:
            from_name = entry["from_class"]
            to_name = entry["to_class"]
            prob = float(entry["prob"])

            # Skip unknown classes (should not happen)
            if from_name not in name_to_idx or to_name not in name_to_idx:
                continue

            i = name_to_idx[from_name]
            j = name_to_idx[to_name]

            confusion_matrix[i, j] = prob

        return confusion_matrix

    @property
    def pi(self) -> np.ndarray:
        """Load the entire pi array from MongoDB into a numpy array."""
        pi = np.zeros((self.n_workers, self.n_classes, self.n_classes))
        for worker_doc in self.worker_mapping.find({}):
            pi[worker_doc["index"], :, :] = self._load_pi_for_worker(
                worker_doc["_id"],
            )
        return pi

    def build_full_pi_tensor(self) -> np.ndarray:
        return self.pi

    @profile
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        gamma = self.gamma
        scale = 1.0 - gamma
        updates = []

        worker_ids = list(worker_mapping.keys())

        worker_cursor = self.db.worker_confusion_matrices.find(
            {"_id": {"$in": worker_ids}},
            {"confusion_matrix": 1},
        )

        worker_conf = {
            doc["_id"]: doc.get("confusion_matrix", [])
            for doc in worker_cursor
        }

        for worker_name, batch_worker_idx in worker_mapping.items():
            existing_matrix = worker_conf.get(worker_name, [])

            entry_dict = {
                (e["from_class"], e["to_class"]): e for e in existing_matrix
            }

            worker_batch_pi = batch_pi[batch_worker_idx].todense()

            nz_from, nz_to = np.nonzero(worker_batch_pi > 0)

            for i, j in zip(nz_from, nz_to):
                batch_prob = worker_batch_pi[i, j]
                from_class = self._reverse_class_mapping.get(i)
                to_class = self._reverse_class_mapping.get(j)
                if from_class is None or to_class is None:
                    continue

                key = (from_class, to_class)
                if key in entry_dict:
                    # update existing
                    entry_dict[key]["prob"] = (
                        scale * entry_dict[key]["prob"] + gamma * batch_prob
                    )
                else:
                    # add new
                    entry_dict[key] = {
                        "from_class": from_class,
                        "to_class": to_class,
                        "prob": gamma * batch_prob,
                    }

            # Normalize per from_class
            updated_matrix = list(entry_dict.values())
            from_class_entries = {}
            for e in updated_matrix:
                from_class_entries.setdefault(e["from_class"], []).append(e)

            for fc, entries in from_class_entries.items():
                row_sum = sum(e["prob"] for e in entries)
                if row_sum > 0:
                    inv_sum = 1.0 / row_sum
                    for e in entries:
                        e["prob"] *= inv_sum

            updates.append(
                UpdateOne(
                    {"_id": worker_name},
                    {"$set": {"confusion_matrix": updated_matrix}},
                    upsert=True,
                ),
            )

        if updates:
            with self.mongo_timer("online update worker confusion matrices"):
                self.db.worker_confusion_matrices.bulk_write(
                    updates,
                    ordered=False,
                )


class DawidSkeneMongo(MongoOnlineAlgorithm):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform the E-step of the expectation-maximization algorithm.

        This method calculates the expected values of the latent variables given the current
        estimates of the parameters. It computes the likelihood of each class for each task
        in the provided batch matrix and normalizes these values to obtain probabilities.

        Parameters:
        ----------
        batch_matrix : np.ndarray
            A 2D array of shape (n_tasks, n_workers) where each entry indicates
            the presence (1) or absence (0) of a worker's assignment to a task.

        batch_pi : np.ndarray
            A 3D array of shape (n_workers, n_classes, n_labels) representing
            the probability of each worker assigning a label to a class.

        batch_rho : np.ndarray
            A 1D array of shape (n_classes) representing the prior
            probabilities of each class.

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - batch_T: A 2D array of shape (n_tasks, n_classes) where each entry
            represents the normalized likelihood of each class for each task.
            - batch_denom_e_step: A 2D array of shape (n_tasks, 1) containing
            the sum of the likelihoods for each task, used for normalization.

        """
        batch_T = np.zeros((batch_matrix.shape[0], batch_matrix.shape[2]))
        for t in range(batch_matrix.shape[0]):
            for c in range(batch_matrix.shape[2]):
                likelihood = (
                    np.prod(
                        np.power(batch_pi[:, c, :], batch_matrix[t, :, :]),
                    )
                    * batch_rho[c]
                )
                batch_T[t, c] = likelihood

        batch_denom_e_step = batch_T.sum(1, keepdims=True)
        batch_T = np.where(
            batch_denom_e_step > 0,
            batch_T / batch_denom_e_step,
            batch_T,
        )
        return batch_T, batch_denom_e_step

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform the M-step of the Expectation-Maximization (EM) algorithm.

        This method updates the parameters of the model based on the expected
        values calculated during the E-step.
        It computes the updated class priors and the conditional probabilities
        of labels given classes.

        Parameters:
        ----------
        batch_matrix : np.ndarray
            A 2D array of shape (n_tasks, n_workers) where each entry indicates
            the presence (1) or absence (0) of a worker's assignment to a task.

        batch_T : np.ndarray
            A 2D array of shape (n_tasks, n_classes) representing the
            normalized likelihood of each class for eachtask,
            as computed in the E-step.

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - batch_rho: A 1D array of shape (n_classes) representing the
            updated prior probabilities of each class.
            - batch_pi: A 3D array of shape (n_workers, n_classes, n_classes)
              representing the updated conditional probabilities of labels
              given classes.

        """
        batch_rho = batch_T.mean(axis=0)

        batch_pi = np.zeros(
            (
                batch_matrix.shape[1],
                batch_matrix.shape[2],
                batch_matrix.shape[2],
            ),
        )

        for q in range(batch_matrix.shape[2]):
            pij = batch_T[:, q] @ batch_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)
            batch_pi[:, q, :] = pij / np.where(
                denom <= 0,
                -1e9,
                denom,
            ).reshape(-1, 1)

        return batch_rho, batch_pi

    def _load_pi_for_worker(self, worker_id: str) -> np.ndarray:
        doc = self.db.worker_confusion_matrices.find_one({"_id": worker_id})
        n_classes = self.n_classes
        confusion_matrix = np.zeros((n_classes, n_classes))
        if doc is not None and "confusion_matrix" in doc:
            for entry in doc["confusion_matrix"]:
                from_idx = entry["from_class_id"]
                to_idx = entry["to_class_id"]
                prob = entry["prob"]
                if from_idx < n_classes and to_idx < n_classes:
                    confusion_matrix[from_idx, to_idx] = prob
        return confusion_matrix

    @property
    def pi(self) -> np.ndarray:
        """Load the entire pi array from MongoDB into a numpy array."""
        pi = np.zeros((self.n_workers, self.n_classes, self.n_classes))
        for worker_id in range(self.n_workers):
            pi[worker_id, :, :] = self._load_pi_for_worker(str(worker_id))
        return pi

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        class_docs = self.db.class_mapping.find(
            {"_id": {"$in": list(class_mapping.keys())}},
        )
        batch_to_global = {
            class_mapping[doc["_id"]]: doc["index"] for doc in class_docs
        }

        worker_confusions_cursor = self.db.worker_confusion_matrices.find(
            {"_id": {"$in": list(worker_mapping.keys())}},
        )
        worker_confusions = {
            doc["_id"]: doc.get("confusion_matrix", [])
            for doc in worker_confusions_cursor
        }
        updates = []
        for worker, batch_worker_idx in worker_mapping.items():
            confusion_matrix = worker_confusions.get(worker, [])

            entry_map = {
                (entry["from_class_id"], entry["to_class_id"]): idx
                for idx, entry in enumerate(confusion_matrix)
            }

            # Update the confusion matrix with new probabilities
            for i_batch, i_global in batch_to_global.items():
                for j_batch, j_global in batch_to_global.items():
                    batch_prob = batch_pi[batch_worker_idx, i_batch, j_batch]
                    key = (i_global, j_global)

                    if key in entry_map:
                        idx = entry_map[key]
                        confusion_matrix[idx]["prob"] = (
                            1 - self.gamma
                        ) * confusion_matrix[idx][
                            "prob"
                        ] + self.gamma * batch_prob
                    else:
                        if batch_prob == 0:
                            continue
                        confusion_matrix.append(
                            {
                                "from_class_id": i_global,
                                "to_class_id": j_global,
                                "prob": self.gamma * batch_prob,
                            },
                        )
                        entry_map[key] = len(confusion_matrix) - 1

            # Normalize the confusion matrix
            from_class_ids = set(
                entry["from_class_id"] for entry in confusion_matrix
            )

            for from_class_id in from_class_ids:
                # Filter entries with the current from_class_id
                entries = [
                    e
                    for e in confusion_matrix
                    if e["from_class_id"] == from_class_id
                ]
                if entries:
                    row_sum = sum(entry["prob"] for entry in entries)
                    if row_sum > 0:
                        for entry in entries:
                            entry["prob"] /= row_sum

            # Save the updated confusion matrix back to MongoDB
            updates.append(
                UpdateOne(
                    {"_id": worker},
                    {"$set": {"confusion_matrix": confusion_matrix}},
                    upsert=True,
                ),
            )
        if updates:
            self.db.worker_confusion_matrices.bulk_write(
                updates,
                ordered=False,
            )


class DawidSkeneOnline(OnlineAlgorithm):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform the E-step of the expectation-maximization algorithm.

        This method calculates the expected values of the latent variables given the current
        estimates of the parameters. It computes the likelihood of each class for each task
        in the provided batch matrix and normalizes these values to obtain probabilities.

        Parameters:
        ----------
        batch_matrix : np.ndarray
            A 2D array of shape (n_tasks, n_workers) where each entry indicates
            the presence (1) or absence (0) of a worker's assignment to a task.

        batch_pi : np.ndarray
            A 3D array of shape (n_workers, n_classes, n_labels) representing
            the probability of each worker assigning a label to a class.

        batch_rho : np.ndarray
            A 1D array of shape (n_classes) representing the prior
            probabilities of each class.

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - batch_T: A 2D array of shape (n_tasks, n_classes) where each entry
            represents the normalized likelihood of each class for each task.
            - batch_denom_e_step: A 2D array of shape (n_tasks, 1) containing
            the sum of the likelihoods for each task, used for normalization.

        """
        batch_T = np.zeros((batch_matrix.shape[0], batch_matrix.shape[2]))
        for t in range(batch_matrix.shape[0]):
            for c in range(batch_matrix.shape[2]):
                likelihood = (
                    np.prod(
                        np.power(batch_pi[:, c, :], batch_matrix[t, :, :]),
                    )
                    * batch_rho[c]
                )
                batch_T[t, c] = likelihood

        batch_denom_e_step = batch_T.sum(1, keepdims=True)
        batch_T = np.where(
            batch_denom_e_step > 0,
            batch_T / batch_denom_e_step,
            batch_T,
        )
        return batch_T, batch_denom_e_step

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform the M-step of the Expectation-Maximization (EM) algorithm.

        This method updates the parameters of the model based on the expected
        values calculated during the E-step.
        It computes the updated class priors and the conditional probabilities
        of labels given classes.

        Parameters:
        ----------
        batch_matrix : np.ndarray
            A 2D array of shape (n_tasks, n_workers) where each entry indicates
            the presence (1) or absence (0) of a worker's assignment to a task.

        batch_T : np.ndarray
            A 2D array of shape (n_tasks, n_classes) representing the
            normalized likelihood of each class for eachtask,
            as computed in the E-step.

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - batch_rho: A 1D array of shape (n_classes) representing the
            updated prior probabilities of each class.
            - batch_pi: A 3D array of shape (n_workers, n_classes, n_classes)
              representing the updated conditional probabilities of labels
              given classes.

        """
        batch_rho = batch_T.mean(axis=0)

        batch_pi = np.zeros(
            (
                batch_matrix.shape[1],
                batch_matrix.shape[2],
                batch_matrix.shape[2],
            ),
        )

        for q in range(batch_matrix.shape[2]):
            pij = batch_T[:, q] @ batch_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)
            batch_pi[:, q, :] = pij / np.where(
                denom <= 0,
                -1e9,
                denom,
            ).reshape(-1, 1)

        return batch_rho, batch_pi

    def _initialize_pi(self) -> None:
        self.pi = np.zeros((self.n_workers, self.n_classes, self.n_classes))

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        # Update only workers present in the batch
        for worker, batch_worker_idx in worker_mapping.items():
            worker_idx = self.worker_mapping[worker]

            # For each class in the batch, map batch class idx to global class idx
            batch_to_global = {
                batch_class_idx: self.class_mapping[class_name]
                for class_name, batch_class_idx in class_mapping.items()
            }
            for i_batch, i_global in batch_to_global.items():
                for j_batch, j_global in batch_to_global.items():
                    self.pi[worker_idx, i_global, j_global] = (
                        1 - self.gamma
                    ) * self.pi[
                        worker_idx,
                        i_global,
                        j_global,
                    ] + self.gamma * batch_pi[
                        batch_worker_idx,
                        i_batch,
                        j_batch,
                    ]

                row_sum = self.pi[worker_idx, i_global, :].sum()
                if row_sum > 0:
                    self.pi[worker_idx, i_global, :] /= row_sum


class WeightedDawidSkene(
    VectorizedDawidSkeneOnlineMongo,
    WeightedOnlineAlgorithm,
):
    """One-step Weighted Majority Voting after Dawid Skene.
    Use the mean of the diagonal of the confusion matrix as a weight for
    one-step weighted majority voting."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_workers_weights(self, worker_ids=None):
        pipeline = []

        if worker_ids is not None:
            pipeline.append({"$match": {"_id": {"$in": list(worker_ids)}}})

        pipeline += [
            {"$unwind": "$confusion_matrix"},
            {
                "$match": {
                    "$expr": {
                        "$eq": [
                            "$confusion_matrix.from_class",
                            "$confusion_matrix.to_class",
                        ],
                    },
                },
            },
            {
                "$group": {
                    "_id": "$_id",
                    "weight": {"$avg": "$confusion_matrix.prob"},
                },
            },
        ]

        return {
            doc["_id"]: doc["weight"]
            for doc in self.db.worker_confusion_matrices.aggregate(pipeline)
        }


class OnlineDawidSkene(VectorizedDawidSkeneOnlineMongo):
    """
    Cappé-style online EM:
    For each task, update sufficient statistics with a stochastic
    approximation step and immediately refresh parameters.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._online_step: int | None = None

    def _init_T(
        self,
        batch_matrix: sp.COO,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
    ) -> sp.COO:
        """Initialize T matrix based on batch data."""

        T = batch_matrix.sum(axis=1)
        tdim = T.sum(1, keepdims=True).todense()
        batch_T = np.where(tdim > 0, T / tdim, 0).todense()

        return sp.COO(batch_T)

    @profile
    def _online_update_T(
        self,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
        batch_T: sp.COO,
        top_k: int | None = None,
    ) -> None:
        """
        Update running task-class sufficient statistics using online EM:
            S_{n+1, l} = (1-gamma) S_{n, l} + gamma * T_{n+1, l}
        top_k is applied **only to limit computation**, not to forget other classes.
        """
        scale = 1 - self.gamma

        # --- optionally keep only top_k per task in batch_T ---
        if top_k is not None and top_k < batch_T.shape[1]:
            if isinstance(batch_T, sp.COO):
                batch_T = sparse_topk_fast(batch_T, top_k)
            else:
                idx = np.argpartition(batch_T, -top_k, axis=1)[:, -top_k:]
                mask = np.zeros_like(batch_T, dtype=bool)
                rows = np.arange(batch_T.shape[0])[:, None]
                mask[rows, idx] = True
                batch_T = np.where(mask, batch_T, 0.0)

        row_idx, col_idx = batch_T.coords
        data = batch_T.data * self.gamma

        uniq_tasks = np.unique(row_idx)
        uniq_classes = np.unique(col_idx)

        task_idx = np.searchsorted(uniq_tasks, row_idx)
        class_idx = np.searchsorted(uniq_classes, col_idx)
        block = np.zeros(
            (len(uniq_tasks), len(uniq_classes)),
            dtype=np.float64,
        )
        np.add.at(block, (task_idx, class_idx), data)

        task_names = list(task_mapping)
        class_names = [self._reverse_class_mapping[c] for c in uniq_classes]

        # Fetch existing probabilities from DB
        docs = self.db.task_class_probs.find(
            {"_id": {"$in": list(task_mapping)}},
            {"_id": 1, "probs": 1},
        )
        task_to_probs = {doc["_id"]: doc.get("probs", {}) for doc in docs}

        updates = []
        for i, task_name in enumerate(task_names):
            # scale existing probabilities
            current_probs = {
                cls: val * scale
                for cls, val in task_to_probs.get(task_name, {}).items()
            }

            # add batch contributions
            for j, cls in enumerate(class_names):
                current_probs[cls] = current_probs.get(cls, 0.0) + float(
                    block[i, j],
                )

            # --- optionally keep only top_k globally for insertion ---
            if top_k is not None and len(current_probs) > top_k:
                sorted_classes = sorted(
                    current_probs.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                top_classes = dict(sorted_classes[:top_k])
                current_probs = top_classes  # keep only top_k for DB update

            updates.append(
                UpdateOne(
                    {"_id": task_name},
                    {
                        "$set": {
                            f"probs.{cls}": val
                            for cls, val in current_probs.items()
                        },
                    },
                    upsert=True,
                ),
            )

        if updates:
            with self.mongo_timer("online update task class probs"):
                self.db.task_class_probs.bulk_write(updates, ordered=False)

        # normalize all probs for these tasks after insertion
        self._normalize_probs(task_names)

    def _online_update_sufficient_statistics(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_T: np.ndarray,
        batch_matrix: sp.COO | np.ndarray,
        top_k: int | None = None,
    ) -> None:
        """
        Default online update for full confusion matrices using sufficient stats:
        S_{l,k}^{(j)} <- (1-gamma) S_{l,k}^{(j)} + gamma * sum_t T_{t,l} 1[y_t^{(j)}=k]
        If top_k is None -> keep all classes.
        If top_k is set   -> keep only top_k classes per task.
        """
        gamma = self.gamma
        scale = 1.0 - gamma
        if top_k is not None and top_k < batch_T.shape[1]:
            if isinstance(batch_T, sp.COO):
                batch_T = sparse_topk_fast(batch_T, top_k)
            else:
                idx = np.argpartition(batch_T, -top_k, axis=1)[:, -top_k:]
                mask = np.zeros_like(batch_T, dtype=bool)
                rows = np.arange(batch_T.shape[0])[:, None]
                mask[rows, idx] = True
                batch_T = np.where(mask, batch_T, 0.0)

        # Update class prior sufficient statistics (counts)
        if isinstance(batch_T, sp.COO):
            class_counts = batch_T.sum(axis=0).todense().ravel()
        else:
            class_counts = batch_T.sum(axis=0)

        self.db.class_priors.update_many(
            {},
            [
                {
                    "$set": {
                        "count": {
                            "$multiply": [
                                {"$ifNull": ["$count", 0]},
                                scale,
                            ],
                        },
                    },
                },
            ],
        )

        class_ops = []
        for class_name, batch_class_idx in class_mapping.items():
            delta = float(class_counts[batch_class_idx]) * gamma
            if delta == 0.0:
                continue
            class_ops.append(
                UpdateOne(
                    {"_id": class_name},
                    {"$inc": {"count": delta}},
                    upsert=True,
                ),
            )
        if class_ops:
            with self.mongo_timer("online update class sufficient stats"):
                self.db.class_priors.bulk_write(class_ops, ordered=False)

        # --- WORKER CONFUSION UPDATE  ---
        worker_ids = list(worker_mapping.keys())
        worker_cursor = self.db.worker_sufficient_statistics.find(
            {"_id": {"$in": worker_ids}},
            {"confusion_matrix": 1},
        )
        worker_conf = {
            doc["_id"]: doc.get("confusion_matrix", [])
            for doc in worker_cursor
        }

        expected_sparse: dict[int, dict[tuple[int, int], float]] | None = None
        expected_dense = None
        if isinstance(batch_matrix, sp.COO) and isinstance(batch_T, sp.COO):
            expected_sparse = {}

            t_coords_T, l_coords_T = batch_T.coords
            data_T = batch_T.data
            task_to_classes: dict[int, list[tuple[int, float]]] = {}
            for t, l, v in zip(t_coords_T, l_coords_T, data_T):
                task_to_classes.setdefault(int(t), []).append(
                    (int(l), float(v)),
                )

            t_coords_M, w_coords_M, k_coords_M = batch_matrix.coords
            for t, w, k in zip(t_coords_M, w_coords_M, k_coords_M):
                entries = task_to_classes.get(int(t))
                if not entries:
                    continue
                worker_dict = expected_sparse.setdefault(int(w), {})
                k_int = int(k)
                for l, v in entries:
                    key = (l, k_int)
                    worker_dict[key] = worker_dict.get(key, 0.0) + v
        else:
            weighted = batch_T[:, None, :, None] * batch_matrix[:, :, None, :]
            expected_dense = weighted.sum(axis=0)

        updates = []
        for worker_name, batch_worker_idx in worker_mapping.items():
            existing_matrix = worker_conf.get(worker_name, [])
            entry_dict = {
                (e["from_class"], e["to_class"]): e for e in existing_matrix
            }

            for entry in entry_dict.values():
                entry["prob"] *= scale

            if expected_sparse is not None:
                worker_expected_sparse = expected_sparse.get(
                    batch_worker_idx,
                    {},
                )
                for (i, j), count in worker_expected_sparse.items():
                    if count <= 0:
                        continue
                    from_class = self._reverse_class_mapping.get(i)
                    to_class = self._reverse_class_mapping.get(j)
                    if from_class is None or to_class is None:
                        continue

                    key = (from_class, to_class)
                    if key in entry_dict:
                        entry_dict[key]["prob"] += gamma * count
                    else:
                        entry_dict[key] = {
                            "from_class": from_class,
                            "to_class": to_class,
                            "prob": gamma * count,
                        }
            else:
                worker_expected = expected_dense[batch_worker_idx]
                nz_from, nz_to = np.nonzero(worker_expected)
                for i, j in zip(nz_from, nz_to):
                    count = float(worker_expected[i, j])
                    if count <= 0:
                        continue
                    from_class = self._reverse_class_mapping.get(i)
                    to_class = self._reverse_class_mapping.get(j)
                    if from_class is None or to_class is None:
                        continue

                    key = (from_class, to_class)
                    if key in entry_dict:
                        entry_dict[key]["prob"] += gamma * count
                    else:
                        entry_dict[key] = {
                            "from_class": from_class,
                            "to_class": to_class,
                            "prob": gamma * count,
                        }

            updated_matrix = list(entry_dict.values())

            updates.append(
                UpdateOne(
                    {"_id": worker_name},
                    {"$set": {"confusion_matrix": updated_matrix}},
                    upsert=True,
                ),
            )

        if updates:
            with self.mongo_timer("online update worker confusion matrices"):
                self.db.worker_sufficient_statistics.bulk_write(
                    updates,
                    ordered=False,
                )

    def _load_rho_from_db(self, class_mapping: dict[str, int]) -> np.ndarray:
        n_classes = len(class_mapping)
        rho = np.zeros(n_classes, dtype=np.float64)

        class_ids = list(class_mapping.keys())
        cursor = self.db.class_priors.find(
            {"_id": {"$in": class_ids}},
            {"_id": 1, "prob": 1},
        )
        for doc in cursor:
            cls_id = doc["_id"]
            idx = class_mapping.get(cls_id)
            if idx is not None:
                rho[idx] = float(doc.get("prob", 0.0))

        total = rho.sum()
        if total > 0:
            missing = rho == 0
            if np.any(missing):
                rho[missing] = 1e-12
                total = rho.sum()
            rho /= total
        else:
            rho[:] = 1.0 / n_classes

        return sp.COO(rho)

    def _load_pi_from_db(
        self,
        worker_ids: list[str],
        class_mapping: dict[str, int],
    ) -> np.ndarray:
        n_workers = len(worker_ids)
        n_classes = len(class_mapping)

        pi = np.zeros((n_workers, n_classes, n_classes), dtype=np.float64)
        if n_workers == 0:
            return pi

        worker_to_idx = {w: i for i, w in enumerate(worker_ids)}

        cursor = self.db.worker_confusion_matrices.find(
            {"_id": {"$in": worker_ids}},
            {"_id": 1, "confusion_matrix": 1},
        )
        for doc in cursor:
            w_name = doc["_id"]
            w_idx = worker_to_idx.get(w_name)
            if w_idx is None:
                continue

            for e in doc.get("confusion_matrix", []):
                from_cls = e.get("from_class")
                to_cls = e.get("to_class")
                l_idx = class_mapping.get(from_cls)
                k_idx = class_mapping.get(to_cls)
                if l_idx is None or k_idx is None:
                    continue
                pi[w_idx, l_idx, k_idx] = float(e.get("prob", 0.0))

        row_sums = pi.sum(axis=2, keepdims=True)
        zero_rows = row_sums == 0
        pi = np.divide(pi, row_sums, out=np.zeros_like(pi), where=row_sums > 0)
        if n_classes > 0:
            pi[zero_rows.repeat(n_classes, axis=2)] = 1.0 / n_classes

        return sp.COO(pi)

    def _online_update_pi_from_sufficient_statistics(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
    ) -> None:
        worker_ids = list(worker_mapping.keys())
        worker_cursor = self.db.worker_sufficient_statistics.find(
            {"_id": {"$in": worker_ids}},
            {"confusion_matrix": 1},
        )
        worker_conf = {
            doc["_id"]: doc.get("confusion_matrix", [])
            for doc in worker_cursor
        }

        updates = []
        for worker_name in worker_ids:
            counts = worker_conf.get(worker_name, [])
            if not counts:
                updates.append(
                    UpdateOne(
                        {"_id": worker_name},
                        {"$set": {"confusion_matrix": []}},
                        upsert=True,
                    ),
                )
                continue

            by_from: dict[str, list[dict]] = {}
            for entry in counts:
                prob = float(entry.get("prob", 0.0))
                if prob <= 0:
                    continue
                by_from.setdefault(entry["from_class"], []).append(entry)

            normalized = []
            for from_class, entries in by_from.items():
                row_sum = sum(float(e["prob"]) for e in entries)
                if row_sum <= 0:
                    continue
                inv_sum = 1.0 / row_sum
                for e in entries:
                    p = float(e["prob"]) * inv_sum
                    if p <= 0:
                        continue
                    normalized.append(
                        {
                            "from_class": e["from_class"],
                            "to_class": e["to_class"],
                            "prob": p,
                        },
                    )

            updates.append(
                UpdateOne(
                    {"_id": worker_name},
                    {"$set": {"confusion_matrix": normalized}},
                    upsert=True,
                ),
            )

        if updates:
            with self.mongo_timer("online update worker confusion matrices"):
                self.db.worker_confusion_matrices.bulk_write(
                    updates,
                    ordered=False,
                )

    def _online_update_rho_from_sufficient_statistics(self) -> None:
        total_doc = list(
            self.db.class_priors.aggregate(
                [{"$group": {"_id": None, "total": {"$sum": "$count"}}}],
            ),
        )
        if not total_doc:
            return
        total = float(total_doc[0].get("total", 0.0))
        if total <= 0.0:
            return

        with self.mongo_timer("online update class priors"):
            self.db.class_priors.update_many(
                {},
                [
                    {
                        "$set": {
                            "prob": {
                                "$cond": {
                                    "if": {"$gt": ["$count", 0]},
                                    "then": {"$divide": ["$count", total]},
                                    "else": 0.0,
                                },
                            },
                        },
                    },
                ],
            )

    @profile
    def _online_update(
        self,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_T: np.ndarray,
        batch_rho: np.ndarray,
        batch_pi: np.ndarray,
        batch_matrix: sp.COO | np.ndarray,
    ) -> None:
        self._online_update_T(task_mapping, class_mapping, batch_T, self.top_k)
        self._online_update_rho(class_mapping, batch_rho)
        self._online_update_pi(worker_mapping, class_mapping, batch_pi)
        self._online_update_sufficient_statistics(
            worker_mapping,
            class_mapping,
            batch_T,
            batch_matrix,
            top_k=self.top_k,
        )

    def _em_loop_on_batch(
        self,
        batch_matrix: np.ndarray | sp.COO,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        epsilon: Annotated[float, Ge(0)] = 1e-6,
        maxiter: Annotated[int, Gt(0)] = 50,
    ) -> list[float]:
        """
        Cappé-style Online EM step on a batch.
        Interface preserved, but EM iterations are disabled by design.
        """
        _ = epsilon, maxiter

        self._batch_size = 1

        ll: list[float] = []

        # init one step MV, then get pi and rho
        global_T = self._init_T(
            batch_matrix,
            task_mapping,
            class_mapping,
        )
        global_rho, global_pi = self._m_step(batch_matrix, global_T)

        self._online_update(
            task_mapping,
            worker_mapping,
            class_mapping,
            global_T,
            global_rho,
            global_pi,
            batch_matrix,
        )

        # TODO add avg_pi
        for batch_task_idx in task_mapping.values():
            # --- extract task-specific data ---
            if isinstance(batch_matrix, sp.COO):
                mask = batch_matrix.coords[0] == batch_task_idx
                task_w = batch_matrix.coords[1][mask]
                task_k = batch_matrix.coords[2][mask]

                uniq_workers = np.unique(task_w)
                worker_ids_task = [
                    self._reverse_worker_mapping[int(w)] for w in uniq_workers
                ]
                worker_to_local_task = {
                    int(w): i for i, w in enumerate(uniq_workers)
                }

                task_classes = np.unique(task_k)
                class_mapping_task = {
                    self._reverse_class_mapping[int(c)]: i
                    for i, c in enumerate(task_classes)
                }

                task_matrix = np.zeros(
                    (1, len(uniq_workers), len(task_classes)),
                    dtype=bool,
                )
                task_matrix[
                    0,
                    [worker_to_local_task[int(w)] for w in task_w],
                    [
                        class_mapping_task[self._reverse_class_mapping[int(k)]]
                        for k in task_k
                    ],
                ] = True

            else:
                row = batch_matrix[batch_task_idx]
                worker_mask = row.sum(axis=1) > 0
                worker_idxs = np.where(worker_mask)[0]
                worker_ids_task = [
                    self._reverse_worker_mapping[int(w)] for w in worker_idxs
                ]

                task_classes = np.nonzero(row.sum(axis=0))[0]
                class_mapping_task = {
                    self._reverse_class_mapping[int(c)]: i
                    for i, c in enumerate(task_classes)
                }

                task_matrix = row[worker_idxs][:, task_classes][
                    None,
                    :,
                    :,
                ]  # add batch dim

            task_matrix = sp.COO(task_matrix)

            # --- load global pi/rho but restricted to task-specific classes/workers ---
            batch_rho = self._load_rho_from_db(class_mapping_task)
            batch_pi = self._load_pi_from_db(
                worker_ids_task,
                class_mapping_task,
            )

            # e-step
            batch_T, batch_denom_e_step = self._e_step(
                task_matrix,
                batch_pi,
                batch_rho,
            )

            # update T
            task_name = self._reverse_task_mapping[batch_task_idx]
            task_mapping_task_single = {task_name: batch_task_idx}
            batch_T_update = np.zeros(
                (len(task_mapping), len(class_mapping)),
                dtype=np.float64,
            )
            for cls_name, idx in class_mapping_task.items():
                global_idx = class_mapping[
                    cls_name
                ]  # map task-local -> global index
                batch_T_update[batch_task_idx, global_idx] = batch_T[0, idx]

            self._online_update_T(
                task_mapping_task_single,
                class_mapping,
                sp.COO(batch_T_update),
                self.top_k,
            )

            # m-step: update only task-specific workers
            worker_mapping_task = {
                worker_id: i for i, worker_id in enumerate(worker_ids_task)
            }
            self._online_update_sufficient_statistics(
                worker_mapping_task,
                class_mapping_task,
                batch_T,
                task_matrix,
                top_k=self.top_k,
            )
            self._online_update_rho_from_sufficient_statistics()
            self._online_update_pi_from_sufficient_statistics(
                worker_mapping_task,
                class_mapping_task,
            )

            likeli = float(np.sum(np.log(batch_denom_e_step + 1e-12)))
            ll.append(likeli)
        return ll


batch1 = {
    "task_1": {
        "user_1": "pine",
        "user_2": "pine",
        "user_3": "pine",
        "user_4": "pine",
    },
    "task_2": {"user_5": "oak", "user_6": "maple", "user_7": "maple"},
    "task_3": {
        "user_0": "oak",
        "user_2": "maple",
        "user_5": "cedar",
    },
}

batch2 = {
    "task_1": {"user_0": "oak"},
    "task_4": {"user_1": "pine"},
    "task_5": {"user_2": "maple"},
}

m = OnlineDawidSkene(top_k=5)
m.drop()
m.process_batch(batch1)
m.process_batch(batch2)

for x in m.db.worker_sufficient_statistics.find({}):
    print(x)

m.get_answers()
m.pi

m.db.task_class_probs.find_one({"_id": "task_1"})
# %%
