from __future__ import annotations

import numpy as np

from peerannot.models.aggregation.mongo_online_helpers import (
    MongoOnlineAlgorithm,
)
from peerannot.models.aggregation.online_helpers import (
    OnlineAlgorithm,
)
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


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
