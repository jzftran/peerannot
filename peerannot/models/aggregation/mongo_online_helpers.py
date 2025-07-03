from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Annotated,
)

import numpy as np
from annotated_types import Ge, Gt
from pymongo import MongoClient, UpdateOne

from peerannot.models.aggregation.warnings_errors import (
    NotInitialized,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from peerannot.models.aggregation.types import (
        ClassMapping,
        TaskMapping,
        WorkerMapping,
    )
    from peerannot.models.template import AnswersDict


class MongoOnlineAlgorithm(ABC):
    """Batch EM is the same as in OnlineAlgorithm.
    Global rho, pi and T are stored in MongoDB.
    Online updates of rho, pi,and T reconstruct full rho, and T by rows and pi by user.
    This could be done better by using sparse representations for"""

    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
        mongo_uri: str = "mongodb://localhost:27017/",
        mongo_db_name: str = "online_algorithm",
    ) -> None:
        self.gamma0 = gamma0
        self.decay = decay
        self.t = 0

        # Initialize MongoDB connection
        self.client = MongoClient(mongo_uri)
        self.db = self.client[mongo_db_name]

        # Initialize collections if they don't exist
        collections = [
            "task_class_probs",
            "class_priors",
            "worker_confusion_matrices",
            "task_mapping",
            "worker_mapping",
            "class_mapping",
        ]
        for coll_name in collections:
            if coll_name not in self.db.list_collection_names():
                self.db.create_collection(coll_name)

        self.task_mapping = self.db.get_collection("task_mapping")
        self.worker_mapping = self.db.get_collection("worker_mapping")
        self.class_mapping = self.db.get_collection("class_mapping")

    @property
    def n_classes(self) -> int:
        return self.class_mapping.count_documents({})

    @property
    def n_workers(self) -> int:
        return self.worker_mapping.count_documents({})

    @property
    def n_task(self) -> int:
        return self.task_mapping.count_documents({})

    @property
    def gamma(self) -> float:
        """Compute current step size"""
        return self.gamma0 / (self.t) ** self.decay

    def drop(self):
        self.client.drop_database("online_algorithm")

    def _update_rho_in_mongodb(self, rho_array: np.ndarray) -> None:
        doc = {
            "_id": "rho",
            "probs": rho_array.tolist(),
            "n_classes": self.n_classes,
        }
        self.db.class_priors.replace_one({"_id": "rho"}, doc, upsert=True)

    def _load_T_row(self, task_id: int) -> np.ndarray:
        doc = self.db.task_class_probs.find_one({"_id": task_id})
        if doc is None:
            # Initialize with uniform probabilities if task not found
            probs = np.ones(self.n_classes) / self.n_classes
        else:
            probs = np.array(doc["probs"])
            doc_n_classes = doc.get("n_classes", 0)
            if doc_n_classes < self.n_classes:
                # Expand with uniform probabilities for new classes
                new_probs = np.ones(self.n_classes) / self.n_classes
                new_probs[:doc_n_classes] = probs
                probs = new_probs
        return probs

    def _update_T_row_in_mongodb(
        self,
        task_id: int,
        row_array: np.ndarray,
        current_answer,
    ) -> None:
        doc = {
            "_id": task_id,
            "probs": row_array.tolist(),
            "n_classes": self.n_classes,
            "current_answer": current_answer,
        }
        self.db.task_class_probs.replace_one(
            {"_id": task_id},
            doc,
            upsert=True,
        )

    def _load_pi_for_worker(self, worker_id: int) -> np.ndarray:
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

    def _update_pi_for_worker_in_mongodb(
        self,
        worker_id: int,
        confusion_matrix: np.ndarray,
    ) -> None:
        # Convert the full matrix to sparse list of non-zero entries
        n_classes = self.n_classes
        sparse_entries = []
        for i in range(n_classes):
            for j in range(n_classes):
                prob = confusion_matrix[i, j]
                if prob != 0:  # Only store non-zero entries
                    sparse_entries.append(
                        {
                            "from_class_id": i,
                            "to_class_id": j,
                            "prob": prob,
                        },
                    )
        doc = {
            "_id": worker_id,
            "confusion_matrix": sparse_entries,
            "n_classes": self.n_classes,
        }
        self.db.worker_confusion_matrices.replace_one(
            {"_id": worker_id},
            doc,
            upsert=True,
        )

    @property
    def T(self) -> np.ndarray:
        """Load the entire T matrix from MongoDB into a numpy array."""
        T = np.zeros((self.n_task, self.n_classes))
        for task_id in range(self.n_task):
            T[task_id, :] = self._load_T_row(task_id)
        return T

    @property
    def rho(self) -> np.ndarray:
        """Load the rho array from MongoDB into a numpy array."""
        # TODO@jzftran

    @property
    def pi(self) -> np.ndarray:
        """Load the entire pi array from MongoDB into a numpy array."""
        pi = np.zeros((self.n_workers, self.n_classes, self.n_classes))
        for worker_id in range(self.n_workers):
            pi[worker_id, :, :] = self._load_pi_for_worker(worker_id)
        return pi

    def get_or_create_indices(self, collection, keys: list[Hashable]) -> dict:
        existing_docs = collection.find({"_id": {"$in": keys}})
        existing_map = {doc["_id"]: doc["index"] for doc in existing_docs}

        new_keys = [k for k in keys if k not in existing_map]

        if new_keys:
            max_index = collection.estimated_document_count()
            new_docs = [
                {"_id": key, "index": max_index + i}
                for i, key in enumerate(new_keys)
            ]
            collection.insert_many(new_docs)

            for i, key in enumerate(new_keys):
                existing_map[key] = max_index + i

        return existing_map

    def _prepare_mapping(
        self,
        batch: AnswersDict,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
    ) -> None:
        """Updates the provided mappings in-place."""

        for task_id, worker_class in batch.items():
            if task_id not in task_mapping:
                task_mapping[task_id] = len(task_mapping)

            for worker_id, class_id in worker_class.items():
                if worker_id not in worker_mapping:
                    worker_mapping[worker_id] = len(worker_mapping)
                if class_id not in class_mapping:
                    class_mapping[class_id] = len(class_mapping)

    def _process_batch_to_matrix(
        self,
        batch: AnswersDict,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
    ) -> np.ndarray:
        """Convert a batch of task assignments to a matrix format."""

        num_tasks = len(task_mapping)
        num_users = len(worker_mapping)
        num_labels = len(class_mapping)

        batch_matrix = np.zeros(
            (num_tasks, num_users, num_labels),
            dtype=bool,
        )

        for task_id, worker_class in batch.items():
            for worker_id, class_id in worker_class.items():
                task_index = task_mapping[task_id]
                user_index = worker_mapping[worker_id]
                label_index = class_mapping[class_id]
                batch_matrix[task_index, user_index, label_index] = True
        return batch_matrix

    def _init_T(
        self,
        batch_matrix: np.ndarray,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
    ) -> np.ndarray:
        """Initialize T matrix based on batch data."""

        T = batch_matrix.sum(axis=1)
        tdim = T.sum(1, keepdims=True)
        batch_T = np.where(tdim > 0, T / tdim, 0)

        updated_batch_T = batch_T.copy()

        for g_task, batch_task_idx in task_mapping.items():
            for g_class, batch_class_idx in class_mapping.items():
                global_task_pos = self.task_mapping.find_one({"_id": g_task})[
                    "index"
                ]
                global_class_pos = self.class_mapping.find_one(
                    {"_id": g_class},
                )["index"]

                task_classes = self._load_T_row(global_task_pos)

                if not np.all(
                    np.isclose(
                        task_classes,
                        task_classes[0],
                    ),
                ):  # check if not uniform
                    updated_batch_T[batch_task_idx, batch_class_idx] = (
                        1 - self.gamma
                    ) * batch_T[
                        batch_task_idx,
                        batch_class_idx,
                    ] + self.gamma * task_classes[global_class_pos]

        return batch_T

    def get_probas(self) -> np.ndarray:
        """Get current estimates of task-class probabilities"""
        return self.T

    def get_answers(self) -> np.ndarray:
        """Get current most likely class for each task. Shouldn't
        be used."""
        if self.n_task == 0:
            raise NotInitialized(self.__class__.__name__)
        T = self.T

        rev_class = {
            doc["index"]: doc["_id"]
            for doc in self.db.get_collection("class_mapping").find({})
        }
        map_back = np.vectorize(lambda x: rev_class[x])
        return map_back(np.argmax(T, axis=1))

    def get_answer(self, task_id):
        """Get current most likely class for each task. Shouldn't
        be used."""

        return self.db.task_class_probs.find_one({"_id": task_id})[
            "current_answer"
        ]

    def process_batch(
        self,
        batch: AnswersDict,
        maxiter: int = 50,
        epsilon: float = 1e-6,
    ) -> list[float]:
        """Process a batch with per-batch EM until local convergence."""
        self.t += 1

        task_mapping: TaskMapping = {}
        worker_mapping: WorkerMapping = {}
        class_mapping: ClassMapping = {}

        self._prepare_mapping(
            batch,
            task_mapping,
            worker_mapping,
            class_mapping,
        )

        batch_matrix = self._process_batch_to_matrix(
            batch,
            task_mapping,
            worker_mapping,
            class_mapping,
        )
        return self.process_batch_matrix(
            batch_matrix,
            task_mapping,
            worker_mapping,
            class_mapping,
            maxiter,
            epsilon,
        )

    def process_batch_matrix(
        self,
        batch_matrix: np.ndarray,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        maxiter: int = 50,
        epsilon: float = 1e-6,
    ) -> list[float]:
        self.get_or_create_indices(self.task_mapping, list(task_mapping))
        self.get_or_create_indices(self.worker_mapping, list(worker_mapping))
        self.get_or_create_indices(self.class_mapping, list(class_mapping))

        return self._em_loop_on_batch(
            batch_matrix,
            task_mapping,
            worker_mapping,
            class_mapping,
            epsilon,
            maxiter,
        )

    def _em_loop_on_batch(
        self,
        batch_matrix: np.ndarray,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        epsilon: Annotated[float, Ge(0)] = 1e-6,
        maxiter: Annotated[int, Gt(0)] = 50,
    ) -> list[float]:
        i = 0
        eps = np.inf
        ll: list[float] = []
        batch_T = self._init_T(
            batch_matrix,
            task_mapping,
            class_mapping,
        )

        while i < maxiter and eps > epsilon:
            batch_rho, batch_pi = self._m_step(batch_matrix, batch_T)
            batch_T, batch_denom_e_step = self._e_step(
                batch_matrix,
                batch_pi,
                batch_rho,
            )

            likeli = np.log(np.sum(batch_denom_e_step))
            ll.append(likeli)
            if i > 0:
                eps = np.abs(ll[-1] - ll[-2])
            i += 1

        # Online updates
        self._online_update(
            task_mapping,
            worker_mapping,
            class_mapping,
            batch_T,
            batch_rho,
            batch_pi,
        )

        return ll

    def _online_update(
        self,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_T: np.ndarray,
        batch_rho: np.ndarray,
        batch_pi: np.ndarray,
    ) -> None:
        self._online_update_T(task_mapping, class_mapping, batch_T)
        self._online_update_rho(class_mapping, batch_rho)
        self._online_update_pi(worker_mapping, class_mapping, batch_pi)

    def _online_update_T(
        self,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
        batch_T: np.ndarray,
    ) -> None:
        scale = 1 - self.gamma
        updates = []

        for task_name, batch_task_idx in task_mapping.items():
            set_stage = {}
            for class_name, batch_class_idx in class_mapping.items():
                delta = batch_T[batch_task_idx, batch_class_idx] * self.gamma
                set_stage[f"probs.{class_name}"] = {
                    "$cond": {
                        "if": {
                            "$ne": [
                                {
                                    "$add": [
                                        {
                                            "$multiply": [
                                                {
                                                    "$ifNull": [
                                                        f"$probs.{class_name}",
                                                        0,
                                                    ],
                                                },
                                                scale,
                                            ],
                                        },
                                        delta,
                                    ],
                                },
                                0,
                            ],
                        },
                        "then": {
                            "$add": [
                                {
                                    "$multiply": [
                                        {
                                            "$ifNull": [
                                                f"$probs.{class_name}",
                                                0,
                                            ],
                                        },
                                        scale,
                                    ],
                                },
                                delta,
                            ],
                        },
                        "else": "$$REMOVE",
                    },
                }

            updates.append(
                UpdateOne(
                    {"_id": task_name},
                    [{"$set": set_stage}],
                    upsert=True,
                ),
            )

        if updates:
            self.db.task_class_probs.bulk_write(updates)
            self._normalize_probs(list(task_mapping.keys()))

    def _normalize_probs(self, updated_task_ids: list[str]) -> None:
        pipeline = [
            {
                "$set": {
                    "probs": {
                        "$let": {
                            "vars": {
                                "probs_array": {"$objectToArray": "$probs"},
                                "total": {
                                    "$sum": {
                                        "$map": {
                                            "input": {
                                                "$objectToArray": "$probs",
                                            },
                                            "as": "item",
                                            "in": "$$item.v",
                                        },
                                    },
                                },
                            },
                            "in": {
                                "$arrayToObject": {
                                    "$map": {
                                        "input": "$$probs_array",
                                        "as": "item",
                                        "in": {
                                            "k": "$$item.k",
                                            "v": {
                                                "$cond": {
                                                    "if": {
                                                        "$eq": ["$$total", 0],
                                                    },
                                                    "then": 0,
                                                    "else": {
                                                        "$divide": [
                                                            "$$item.v",
                                                            "$$total",
                                                        ],
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        ]

        self.db.task_class_probs.update_many(
            {"_id": {"$in": updated_task_ids}},
            pipeline,
        )

    def _online_update_rho(
        self,
        class_mapping: ClassMapping,
        batch_rho: np.ndarray,
    ) -> None:
        scale = 1 - self.gamma

        self.db.class_priors.update_many(
            {},  # filter: update all documents
            [
                {
                    "$set": {
                        "prob": {"$multiply": ["$prob", scale]},
                    },
                },
            ],
        )

        ops = []
        gamma = self.gamma

        for class_name, batch_class_idx in class_mapping.items():
            delta = batch_rho[batch_class_idx] * gamma

            ops.append(
                UpdateOne(
                    {"_id": class_name},
                    {"$inc": {"prob": delta}},  # increment
                    upsert=True,
                ),
            )

        # Bulk write to rho collection
        if ops:
            self.db.class_priors.bulk_write(ops)

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        # Update only workers present in the batch
        for worker, batch_worker_idx in worker_mapping.items():
            worker_idx = self.worker_mapping.find_one({"_id": worker})["index"]

            # Load the worker's confusion matrix from MongoDB
            confusion_matrix = self._load_pi_for_worker(worker_idx)

            # For each class in the batch, map batch class idx to global class idx
            batch_to_global = {
                batch_class_idx: self.class_mapping.find_one(
                    {"_id": class_name},
                )["index"]
                for class_name, batch_class_idx in class_mapping.items()
            }

            # Perform updates in memory
            for i_batch, i_global in batch_to_global.items():
                for j_batch, j_global in batch_to_global.items():
                    confusion_matrix[i_global, j_global] = (
                        1 - self.gamma
                    ) * confusion_matrix[
                        i_global,
                        j_global,
                    ] + self.gamma * batch_pi[
                        batch_worker_idx,
                        i_batch,
                        j_batch,
                    ]

                    # Normalize the row
                    row_sum = confusion_matrix[i_global, :].sum()
                    if row_sum > 0:
                        confusion_matrix[i_global, :] /= row_sum

            # Save the updated confusion matrix back to MongoDB
            self._update_pi_for_worker_in_mongodb(worker_idx, confusion_matrix)

    @abstractmethod
    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass
