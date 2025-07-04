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
    Online updates of  pi reconstruct full confusion matrix for each user.
    This could be done better by using sparse representations for pi, or by the approach
    used in class SparseMongoOnlineAlgorithm"""

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

        task_idx_map = {
            doc["_id"]: doc["index"]
            for doc in self.task_mapping.find({}, {"_id": 1, "index": 1})
        }
        class_idx_map = {
            str(doc["_id"]): doc["index"]
            for doc in self.class_mapping.find({}, {"_id": 1, "index": 1})
        }

        task_probs_cursor = self.db.task_class_probs.find(
            {},
            {"_id": 1, "probs": 1},
        )

        for task in task_probs_cursor:
            task_idx = task_idx_map.get(task["_id"])
            if task_idx is None:
                continue

            for cls_key, prob in task.get("probs", {}).items():
                class_idx = class_idx_map.get(str(cls_key))
                if class_idx is not None:
                    T[task_idx, class_idx] = prob

        return T

    @property
    def rho(self) -> np.ndarray:
        """Load the rho array from MongoDB into a numpy array."""
        rho = np.zeros(self.n_classes)

        class_idx_map = {
            str(doc["_id"]): doc["index"]
            for doc in self.class_mapping.find({}, {"_id": 1, "index": 1})
        }

        for doc in self.db.class_priors.find({}, {"_id": 1, "prob": 1}):
            class_id = str(doc["_id"])
            idx = class_idx_map.get(class_id)
            if idx is not None:
                rho[idx] = doc["prob"]

        return rho

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

        task_probs_cursor = self.db.task_class_probs.find(
            {"_id": {"$in": list(task_mapping)}},
            {"_id": 1, **{f"probs.{cls}": 1 for cls in class_mapping}},
        )

        for doc in task_probs_cursor:
            task_name = doc["_id"]
            if task_name not in task_mapping:
                continue

            batch_task_idx = task_mapping[task_name]
            probs = doc.get("probs", {})

            for class_name, prob in probs.items():
                if class_name not in class_mapping:
                    continue
                batch_class_idx = class_mapping[class_name]

                batch_T[batch_task_idx, batch_class_idx] = (
                    1 - self.gamma
                ) * batch_T[
                    batch_task_idx,
                    batch_class_idx,
                ] + self.gamma * prob

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
            {
                "$set": {
                    "current_answer": {
                        "$let": {
                            "vars": {
                                "probs_array": {"$objectToArray": "$probs"},
                            },
                            "in": {
                                "$arrayElemAt": [
                                    {
                                        "$map": {
                                            "input": {
                                                "$filter": {
                                                    "input": "$$probs_array",
                                                    "as": "item",
                                                    "cond": {
                                                        "$eq": [
                                                            "$$item.v",
                                                            {
                                                                "$max": "$$probs_array.v",
                                                            },
                                                        ],
                                                    },
                                                },
                                            },
                                            "as": "item",
                                            "in": "$$item.k",
                                        },
                                    },
                                    0,
                                ],
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


class SparseMongoOnlineAlgorithm(MongoOnlineAlgorithm):
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        # Update only workers present in the batch
        scale = 1 - self.gamma

        updates = []

        for worker, batch_worker_idx in worker_mapping.items():
            # retrieve the current confusion matrix for the worker
            doc = self.db.worker_confusion_matrices.find_one({"_id": worker})
            current_confusion_matrix = {}
            if doc and "confusion_matrix" in doc:
                for entry in doc["confusion_matrix"]:
                    key = (entry["from_class_id"], entry["to_class_id"])
                    current_confusion_matrix[key] = entry["prob"]

            # Iterate over batch classes to update
            for i_class_name, i_batch_idx in class_mapping.items():
                row_total = 0.0
                row_entries = []
                # collect all entries for the current row (i_class_name)
                for j_class_name, j_batch_idx in class_mapping.items():
                    key = (i_class_name, j_class_name)
                    current_prob = current_confusion_matrix.get(key, 0)
                    new_prob = (
                        scale * current_prob
                        + self.gamma
                        * batch_pi[batch_worker_idx, i_batch_idx, j_batch_idx]
                    )

                    if new_prob != 0:
                        row_total += new_prob
                        row_entries.append(
                            (i_class_name, j_class_name, new_prob),
                        )

                # normalize the row if it has entries
                if row_entries:
                    for i_class_name, j_class_name, new_prob in row_entries:
                        normalized_prob = new_prob / row_total
                        existing_entry = None
                        if doc and "confusion_matrix" in doc:
                            for entry in doc["confusion_matrix"]:
                                if (
                                    entry["from_class_id"] == i_class_name
                                    and entry["to_class_id"] == j_class_name
                                ):
                                    existing_entry = entry
                                    break

                        if existing_entry:
                            # update existing entry with normalized probability
                            updates.append(
                                UpdateOne(
                                    {
                                        "_id": worker,
                                        "confusion_matrix": {
                                            "$elemMatch": {
                                                "from_class_id": i_class_name,
                                                "to_class_id": j_class_name,
                                            },
                                        },
                                    },
                                    {
                                        "$set": {
                                            "confusion_matrix.$.prob": normalized_prob,
                                        },
                                    },
                                ),
                            )
                        else:
                            # Add new entry with normalized probability
                            updates.append(
                                UpdateOne(
                                    {"_id": worker},
                                    {
                                        "$push": {
                                            "confusion_matrix": {
                                                "from_class_id": i_class_name,
                                                "to_class_id": j_class_name,
                                                "prob": normalized_prob,
                                            },
                                        },
                                    },
                                    upsert=True,
                                ),
                            )

                # handle zero entries for the current row
                for j_class_name, j_batch_idx in class_mapping.items():
                    key = (i_class_name, j_class_name)
                    current_prob = current_confusion_matrix.get(key, 0)
                    new_prob = (
                        scale * current_prob
                        + self.gamma
                        * batch_pi[batch_worker_idx, i_batch_idx, j_batch_idx]
                    )

                    if new_prob == 0 and key in current_confusion_matrix:
                        # if the new probability is zero and the entry exists, remove it
                        updates.append(
                            UpdateOne(
                                {"_id": worker},
                                {
                                    "$pull": {
                                        "confusion_matrix": {
                                            "from_class_id": i_class_name,
                                            "to_class_id": j_class_name,
                                        },
                                    },
                                },
                            ),
                        )

            # Apply updates in bulk
        if updates:
            self.db.worker_confusion_matrices.bulk_write(updates)
