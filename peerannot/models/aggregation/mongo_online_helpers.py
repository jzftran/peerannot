from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Annotated,
)

import numpy as np
import sparse as sp
from annotated_types import Ge, Gt
from line_profiler import profile
from pymongo import MongoClient, UpdateOne

from peerannot.models.aggregation.warnings_errors import (
    DidNotConverge,
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

from collections import defaultdict


class MongoOnlineAlgorithm(ABC):
    """Batch EM is the same as in OnlineAlgorithm.
    Global rho, pi and T are stored sparsly in MongoDB.
    used in class SparseMongoOnlineAlgorithm"""

    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
        mongo_uri: str = "mongodb://localhost:27017/",
        # mongo_db_name: str = "online_algorithm",
    ) -> None:
        self.gamma0 = gamma0
        self.decay = decay
        self.t = 0

        # Initialize MongoDB connection
        self.client = MongoClient(mongo_uri)
        self.db = self.client[self.__class__.__name__]

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
        self.client.drop_database(self.__class__.__name__)

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
        T_array = np.zeros((self.n_task, self.n_classes))

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
                    T_array[task_idx, class_idx] = prob

        return T_array

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
            pi[worker_id, :, :] = self._load_pi_for_worker(str(worker_id))
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
            if str(task_id) not in task_mapping:
                task_mapping[str(task_id)] = len(task_mapping)

            for worker_id, class_id in worker_class.items():
                if str(worker_id) not in worker_mapping:
                    worker_mapping[str(worker_id)] = len(worker_mapping)
                if str(class_id) not in class_mapping:
                    class_mapping[str(class_id)] = len(class_mapping)

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
                task_index = task_mapping[str(task_id)]
                user_index = worker_mapping[str(worker_id)]
                label_index = class_mapping[str(class_id)]
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
            batch_T[batch_task_idx, :] *= 1 - self.gamma
            probs = doc.get("probs", {})

            for class_name, prob in probs.items():
                if class_name not in class_mapping:
                    continue
                batch_class_idx = class_mapping[class_name]
                batch_T[batch_task_idx, batch_class_idx] += self.gamma * prob

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

    @profile
    def _normalize_probs(self, updated_task_ids: list[str]) -> None:
        docs = list(
            self.db.task_class_probs.find(
                {"_id": {"$in": updated_task_ids}},
                {"_id": 1, "probs": 1},
            ),
        )

        updates = []
        for doc in docs:
            probs = doc["probs"]
            total = sum(probs.values())
            if total == 0:
                normalized = dict.fromkeys(probs, 0)
            else:
                normalized = {k: v / total for k, v in probs.items()}

            current_answer = max(normalized, key=normalized.get)

            updates.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {
                        "$set": {
                            "probs": normalized,
                            "current_answer": current_answer,
                        },
                    },
                ),
            )

        if updates:
            self.db.task_class_probs.bulk_write(updates)

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

        for class_name, batch_class_idx in class_mapping.items():
            delta = batch_rho[batch_class_idx] * self.gamma

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
    """Batch EM is the same as in OnlineAlgorithm.
    Global rho, pi and T are stored sparsly in MongoDB.
    used in class SparseMongoOnlineAlgorithm"""

    @profile
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
        mongo_uri: str = "mongodb://localhost:27017/",
    ) -> None:
        self.gamma0 = gamma0
        self.decay = decay
        self.t = 0

        # Initialize MongoDB connection
        self.client = MongoClient(mongo_uri)
        self.db = self.client[self.__class__.__name__]

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
        self.client.drop_database(self.__class__.__name__)

    @profile
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

    @profile
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
        T_array = np.zeros((self.n_task, self.n_classes))

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
                    T_array[task_idx, class_idx] = prob

        return T_array

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
            pi[worker_id, :, :] = self._load_pi_for_worker(str(worker_id))
        return pi

    @profile
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

    @profile
    def _prepare_mapping(
        self,
        batch: AnswersDict,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
    ) -> None:
        """Updates the provided mappings in-place."""

        for task_id, worker_class in batch.items():
            if str(task_id) not in task_mapping:
                task_mapping[str(task_id)] = len(task_mapping)

            for worker_id, class_id in worker_class.items():
                if str(worker_id) not in worker_mapping:
                    worker_mapping[str(worker_id)] = len(worker_mapping)
                if str(class_id) not in class_mapping:
                    class_mapping[str(class_id)] = len(class_mapping)

        self._reverse_task_mapping = {v: k for k, v in task_mapping.items()}
        self._reverse_worker_mapping = {
            v: k for k, v in worker_mapping.items()
        }
        self._reverse_class_mapping = {v: k for k, v in class_mapping.items()}

    @profile
    def _process_batch_to_matrix(
        self,
        batch: AnswersDict,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
    ) -> sp.COO:
        """Convert a batch of task assignments to a matrix format."""

        coords = [[], [], []]  # [task_indices, worker_indices, class_indices]

        for task_id, worker_class in batch.items():
            for worker_id, class_id in worker_class.items():
                task_index = task_mapping[str(task_id)]
                worker_index = worker_mapping[str(worker_id)]
                class_index = class_mapping[str(class_id)]

                coords[0].append(task_index)
                coords[1].append(worker_index)
                coords[2].append(class_index)

        shape = (
            len(task_mapping),
            len(worker_mapping),
            len(class_mapping),
        )

        return sp.COO(coords, data=True, shape=shape)

    @profile
    def _init_T(
        self,
        batch_matrix: np.ndarray,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
    ) -> np.ndarray:
        """Initialize T matrix based on batch data."""

        T = batch_matrix.sum(axis=1)
        tdim = T.sum(1, keepdims=True).todense()
        batch_T = np.where(tdim > 0, T / tdim, 0).todense()

        task_probs_cursor = self.db.task_class_probs.find(
            {"_id": {"$in": list(task_mapping)}},
            {"_id": 1, **{f"probs.{cls}": 1 for cls in class_mapping}},
        )
        for doc in task_probs_cursor:
            task_name = doc["_id"]
            if task_name not in task_mapping:
                continue

            batch_task_idx = task_mapping[task_name]
            batch_T[batch_task_idx, :] *= 1 - self.gamma
            probs = doc.get("probs", {})

            for class_name, prob in probs.items():
                if class_name not in class_mapping:
                    continue
                batch_class_idx = class_mapping[class_name]
                batch_T[batch_task_idx, batch_class_idx] += self.gamma * prob

        return sp.COO(batch_T)

    @profile
    def get_probas(self) -> np.ndarray:
        """Get current estimates of task-class probabilities"""
        return self.T

    @profile
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

    @profile
    def get_answer(self, task_id):
        """Get current most likely class for each task. Shouldn't
        be used."""

        return self.db.task_class_probs.find_one({"_id": task_id})[
            "current_answer"
        ]

    @profile
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

    @profile
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

    @profile
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

        if eps > epsilon:
            warnings.warn(
                DidNotConverge(self.__class__.__name__, eps, epsilon),
                stacklevel=2,
            )

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

    @profile
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

    @profile
    def _online_update_T(
        self,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
        batch_T: sp.COO,
    ) -> None:
        scale = 1 - self.gamma

        # Precompute fast integer→name lookups
        id_to_task = {v: k for k, v in task_mapping.items()}
        id_to_class = {v: k for k, v in class_mapping.items()}

        # Extract coords & apply gamma once
        row_idx, col_idx = batch_T.coords
        data = batch_T.data * self.gamma

        # Aggregate by (task_id, class_id) using vectorized add
        pairs = np.stack([row_idx, col_idx], axis=1)
        uniq_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        delta_sums = np.zeros(len(uniq_pairs), dtype=float)
        np.add.at(delta_sums, inverse, data)

        # Unique task IDs for DB fetch
        uniq_task_ids = np.unique(uniq_pairs[:, 0])
        task_list = [id_to_task[t] for t in uniq_task_ids]

        # Fetch existing probs for these tasks
        docs = self.db.task_class_probs.find(
            {"_id": {"$in": task_list}},
            {"_id": 1, "probs": 1},
        )
        task_to_probs = {doc["_id"]: doc.get("probs", {}) for doc in docs}

        # Prepare updates (integer → string mapping only once)
        updates_by_task = defaultdict(dict)
        for (task_id, class_id), delta in zip(uniq_pairs, delta_sums):
            task_name = id_to_task[task_id]
            class_name = id_to_class[class_id]
            old_val = task_to_probs.get(task_name, {}).get(class_name, 0.0)
            new_val = old_val * scale + delta
            field_key = f"probs.{class_name}"
            if new_val != 0.0:
                updates_by_task[task_name][field_key] = new_val
            else:
                updates_by_task[task_name][field_key] = None

        # Build bulk update operations
        updates = []
        for task_name, fields in updates_by_task.items():
            set_fields = {k: v for k, v in fields.items() if v is not None}
            unset_fields = {k: "" for k, v in fields.items() if v is None}
            pipeline = []
            if set_fields:
                pipeline.append({"$set": set_fields})
            if unset_fields:
                pipeline.append({"$unset": unset_fields})
            updates.append(
                UpdateOne({"_id": task_name}, pipeline, upsert=True),
            )

        # Write + normalize
        if updates:
            self.db.task_class_probs.bulk_write(updates, ordered=False)
            self._normalize_probs(task_list)

    @profile
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

        for class_name, batch_class_idx in class_mapping.items():
            delta = batch_rho[batch_class_idx] * self.gamma

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

    @profile
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


class SparseMongoOnlineAlgorithmTUpdate(SparseMongoOnlineAlgorithm):
    @profile
    def _online_update_T(
        self,
        task_mapping: dict[str, int],
        class_mapping: dict[str, int],
        batch_T: sp.COO,
    ) -> None:
        """Fast online update of T with sparse input."""
        if not task_mapping or not class_mapping:
            return

        scale = 1 - self.gamma
        # Extract coords & apply gamma once
        row_idx, col_idx = batch_T.coords
        data = batch_T.data * self.gamma

        # Aggregate by (task_id, class_id) using vectorized add
        pairs = np.stack([row_idx, col_idx], axis=1)
        uniq_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        delta_sums = np.zeros(len(uniq_pairs), dtype=float)
        np.add.at(delta_sums, inverse, data)

        if len(uniq_pairs) == 0:
            return  # no updates needed

        task_list = list(task_mapping.keys())

        # Fetch existing probs for these tasks
        docs = list(
            self.db.task_class_probs.find(
                {"_id": {"$in": task_list}},
                {"_id": 1, "probs": 1},
            ),
        )

        # Precompute field keys by class_id
        field_keys = {}
        reverse_class_map = self._reverse_class_mapping
        for class_id, class_name in reverse_class_map.items():
            field_keys[class_id] = f"probs.{class_name}"

        # Determine if we can use the matrix approach
        max_task_id = max(task_mapping.values(), default=0)
        max_class_id = max(class_mapping.values(), default=0)
        matrix_size = (max_task_id + 1) * (max_class_id + 1)
        is_matrix_feasible = (
            matrix_size < 1e6
        )  # arbitrary threshold of 1 million elements

        if is_matrix_feasible:
            # Use matrix approach
            old_vals_matrix = np.zeros(
                (max_task_id + 1, max_class_id + 1),
                dtype=float,
            )
            for doc in docs:
                task_name = doc["_id"]
                task_id = task_mapping.get(task_name, -1)
                if task_id == -1:
                    continue
                probs = doc.get("probs", {})
                for class_name, prob in probs.items():
                    class_id = class_mapping.get(class_name, -1)
                    if class_id != -1:
                        old_vals_matrix[task_id, class_id] = prob

            task_ids = uniq_pairs[:, 0]
            class_ids = uniq_pairs[:, 1]
            old_vals = old_vals_matrix[task_ids, class_ids]
            new_vals = old_vals * scale + delta_sums
        else:
            # Use dictionary approach with id-based keys
            old_probs_by_ids = {}
            for doc in docs:
                task_name = doc["_id"]
                task_id = task_mapping.get(task_name, -1)
                if task_id == -1:
                    continue
                probs = doc.get("probs", {})
                for class_name, prob in probs.items():
                    class_id = class_mapping.get(class_name, -1)
                    if class_id != -1:
                        old_probs_by_ids[(task_id, class_id)] = prob

        # Prepare updates using a list of tuples for efficiency
        update_tuples = []
        reverse_task_map = self._reverse_task_mapping

        if is_matrix_feasible:
            for idx in range(len(uniq_pairs)):
                task_id, class_id = uniq_pairs[idx]
                new_val = new_vals[idx]
                task_name = reverse_task_map[task_id]
                field_key = field_keys[class_id]
                if new_val != 0.0:
                    update_tuples.append((task_name, field_key, new_val))
                else:
                    update_tuples.append((task_name, field_key, None))
        else:
            for idx, (task_id, class_id) in enumerate(uniq_pairs):
                old_val = old_probs_by_ids.get((task_id, class_id), 0.0)
                new_val = old_val * scale + delta_sums[idx]
                task_name = reverse_task_map[task_id]
                field_key = field_keys[class_id]
                if new_val != 0.0:
                    update_tuples.append((task_name, field_key, new_val))
                else:
                    update_tuples.append((task_name, field_key, None))

        # Build updates_by_task from update_tuples
        updates_by_task = defaultdict(dict)
        for task_name, field_key, val in update_tuples:
            updates_by_task[task_name][field_key] = val

        # Build bulk update operations
        updates = []
        for task_name, fields in updates_by_task.items():
            set_fields = {k: v for k, v in fields.items() if v is not None}
            unset_fields = {k: "" for k, v in fields.items() if v is None}
            update_ops = []
            if set_fields:
                update_ops.append({"$set": set_fields})
            if unset_fields:
                update_ops.append({"$unset": unset_fields})
            if update_ops:
                updates.append(
                    UpdateOne({"_id": task_name}, update_ops, upsert=True),
                )

        # Write + normalize
        if updates:
            self.db.task_class_probs.bulk_write(updates, ordered=False)
            self._normalize_probs(task_list)
