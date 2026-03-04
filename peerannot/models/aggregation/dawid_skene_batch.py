# %%
from __future__ import annotations

import numpy as np
from line_profiler import profile
from pymongo import UpdateOne

from peerannot.models.aggregation.mongo_online_helpers import (
    MongoBatchAlgorithm,
    SparseMongoBatchAlgorithm,
    WeightedBatchAlgorithm,
)
from peerannot.models.aggregation.online_helpers import (
    BatchAlgorithm,
)
from peerannot.models.aggregation.types import (
    ClassMapping,
    WorkerMapping,
)


class VectorizedDawidSkeneBatchMongo(SparseMongoBatchAlgorithm):
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
        batch_pi: np.ndarray,
    ) -> None:
        gamma = self.gamma
        scale = 1.0 - gamma
        n_classes = len(self._batch_class_to_idx)

        if n_classes == 0:
            return

        # Build reverse class mapping (deterministic)
        idx_to_class = np.empty(n_classes, dtype=object)
        for cls, idx in self._batch_class_to_idx.items():
            idx_to_class[idx] = cls

        updates = []

        # Extract sparse batch entries
        if hasattr(batch_pi, "coords") and hasattr(batch_pi, "data"):
            w_b, i_b, j_b = batch_pi.coords
            p_b = batch_pi.data
        else:
            dense = np.asarray(batch_pi)
            w_b, i_b, j_b = np.nonzero(dense > 0)
            p_b = dense[w_b, i_b, j_b]

        # Group batch entries per worker
        batch_per_worker = {}
        for w, i, j, p in zip(w_b, i_b, j_b, p_b):
            batch_per_worker.setdefault(w, []).append((i, j, p))

        # Fetch existing matrices
        worker_ids = list(self._batch_worker_to_idx.keys())
        existing_docs = {
            doc["_id"]: doc.get("confusion_matrix", [])
            for doc in self.db.worker_confusion_matrices.find(
                {"_id": {"$in": worker_ids}},
                {"confusion_matrix": 1},
            )
        }

        # Process each worker independently
        for worker_name, worker_idx in self._batch_worker_to_idx.items():
            existing = existing_docs.get(worker_name, [])

            # Build dict[(from_idx, to_idx)] -> prob
            matrix = {}

            for entry in existing:
                from_idx = self._batch_class_to_idx.get(entry["from_class"])
                to_idx = self._batch_class_to_idx.get(entry["to_class"])

                if from_idx is None or to_idx is None:
                    continue

                matrix[(from_idx, to_idx)] = scale * float(entry["prob"])

            for i, j, p in batch_per_worker.get(worker_idx, []):
                key = (i, j)
                matrix[key] = matrix.get(key, 0.0) + gamma * float(p)

            if not matrix:
                updates.append(
                    UpdateOne(
                        {"_id": worker_name},
                        {"$set": {"confusion_matrix": []}},
                        upsert=True,
                    ),
                )
                continue

            rows = {}
            for (i, j), p in matrix.items():
                rows.setdefault(i, []).append((j, p))

            updated_matrix = []

            for i, entries in rows.items():
                total = sum(p for _, p in entries)
                if total <= 0:
                    continue

                from_class = idx_to_class[i]

                for j, p in entries:
                    updated_matrix.append(
                        {
                            "from_class": from_class,
                            "to_class": idx_to_class[j],
                            "prob": float(p / total),
                        },
                    )

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


class DawidSkeneMongo(MongoBatchAlgorithm):
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
        batch_pi: np.ndarray,
    ) -> None:
        class_docs = self.db.class_mapping.find(
            {"_id": {"$in": list(self._batch_class_to_idx.keys())}},
        )
        batch_to_global = {
            self._batch_class_to_idx[doc["_id"]]: doc["index"]
            for doc in class_docs
        }

        worker_confusions_cursor = self.db.worker_confusion_matrices.find(
            {"_id": {"$in": list(self._batch_worker_to_idx.keys())}},
        )
        worker_confusions = {
            doc["_id"]: doc.get("confusion_matrix", [])
            for doc in worker_confusions_cursor
        }
        updates = []
        for worker, batch_worker_idx in self._batch_worker_to_idx.items():
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


class DawidSkeneBatch(BatchAlgorithm):
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
    VectorizedDawidSkeneBatchMongo,
    WeightedBatchAlgorithm,
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
