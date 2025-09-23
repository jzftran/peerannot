from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import (
    TYPE_CHECKING,
    Annotated,
)

import numpy as np
import sparse as sp
from annotated_types import Ge, Gt
from line_profiler import profile
from pymongo import MongoClient, UpdateOne

from peerannot.helpers.logging import OnlineMongoLoggingMixin
from peerannot.models.aggregation.online_helpers import (
    validate_recursion_limit,
)
from peerannot.models.aggregation.warnings_errors import (
    DidNotConverge,
    NotInitialized,
    TaskNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from peerannot.models.aggregation.types import (
        ClassMapping,
        TaskMapping,
        WorkerMapping,
    )
    from peerannot.models.template import AnswersDict


def apply_func_recursive(d, func):
    if isinstance(d, dict):
        return {func(k): apply_func_recursive(v, func) for k, v in d.items()}
    return func(d)


class MongoOnlineAlgorithm(ABC, OnlineMongoLoggingMixin):
    """Batch EM is the same as in OnlineAlgorithm.
    Global rho, pi and T are stored sparsly in MongoDB.
    used in class SparseMongoOnlineAlgorithm"""

    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
        mongo_client: MongoClient | None = None,
        top_k: Annotated[int, Gt(0)] | None = None,
    ) -> None:
        self.gamma0 = gamma0
        self.decay = decay
        self.t = 0
        self.top_k = top_k

        # Initialize MongoDB connection
        self.client = mongo_client or MongoClient("mongodb://localhost:27017/")
        self.db = self.client[self.__class__.__name__]

        # Initialize collections if they don't exist
        collections = [
            "task_class_probs",
            "class_priors",
            "worker_confusion_matrices",
            "task_mapping",
            "worker_mapping",
            "class_mapping",
            "user_votes",
        ]
        for coll_name in collections:
            if coll_name not in self.db.list_collection_names():
                self.db.create_collection(coll_name)

        self.task_mapping = self.db.get_collection("task_mapping")
        self.worker_mapping = self.db.get_collection("worker_mapping")
        self.class_mapping = self.db.get_collection("class_mapping")

    @profile
    def insert_batch(self, batch: AnswersDict):
        """
        Insert or update a batch of votes using bulk_write.
        """
        operations = []

        for task_id, user_votes in batch.items():
            update_fields = {
                f"votes.{user_id}": vote
                for user_id, vote in user_votes.items()
            }
            operations.append(
                UpdateOne(
                    {"_id": task_id},
                    {"$set": update_fields},
                    upsert=True,
                ),
            )

        if operations:  # only write if there are updates
            self.db.user_votes.bulk_write(operations, ordered=False)

    @staticmethod
    def _escape_id(id_str: str) -> str:
        """Escape dots in ID strings to avoid MongoDB field path issues."""
        if isinstance(id_str, str):
            return id_str.replace(".", "__DOT__")
        return str(id_str)

    @staticmethod
    def _unescape_id(escaped_id: str) -> str:
        """Unescape dots in ID strings."""
        if isinstance(escaped_id, str):
            return escaped_id.replace("__DOT__", ".")
        return str(escaped_id)

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
    @abstractmethod
    def pi(self) -> np.ndarray:
        """Load the entire pi array from MongoDB into a numpy array."""

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

    def get_answer(self, task_id: Hashable) -> str:
        doc = self.db.task_class_probs.find_one({"_id": task_id})
        if doc is None:
            raise TaskNotFoundError(task_id)
        return str(doc["current_answer"])

    @profile
    def process_batch(
        self,
        batch: AnswersDict,
        maxiter: int = 50,
        epsilon: float = 1e-6,
    ) -> list[float]:
        """Process a batch with per-batch EM until local convergence."""
        self.t += 1
        batch = apply_func_recursive(batch, self._escape_id)

        self.insert_batch(batch)
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
        batch_matrix: np.ndarray | sp.COO,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        maxiter: int = 50,
        epsilon: float = 1e-6,
    ) -> list[float]:
        batch_start = time.perf_counter()

        self.get_or_create_indices(self.task_mapping, list(task_mapping))
        self.get_or_create_indices(self.worker_mapping, list(worker_mapping))
        self.get_or_create_indices(self.class_mapping, list(class_mapping))

        ll = self._em_loop_on_batch(
            batch_matrix,
            task_mapping,
            worker_mapping,
            class_mapping,
            epsilon,
            maxiter,
        )
        batch_time = time.perf_counter() - batch_start

        self.log_batch_summary(
            self.t,
            len(task_mapping),
            len(worker_mapping),
            len(class_mapping),
            len(ll),
            batch_time,
            ll[-1],
        )

        return ll

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
            iter_start = time.perf_counter()

            batch_rho, batch_pi = self._m_step(batch_matrix, batch_T)

            batch_T, batch_denom_e_step = self._e_step(
                batch_matrix,
                batch_pi,
                batch_rho,
            )
            likeli = np.log(np.sum(batch_denom_e_step))
            ll.append(likeli)
            if i > 0:
                eps = np.abs((ll[-1] - ll[-2]) / (np.abs(ll[-2]) + 1e-12))

            iter_time = time.perf_counter() - iter_start

            self.log_em_iter(i, likeli, eps, iter_time)

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
        self._online_update_T(task_mapping, class_mapping, batch_T, self.top_k)
        self._online_update_rho(class_mapping, batch_rho)
        self._online_update_pi(worker_mapping, class_mapping, batch_pi)

    def _online_update_T(
        self,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
        batch_T: np.ndarray,
        top_k: int | None = None,
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
            self.db.task_class_probs.bulk_write(updates, ordered=False)
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
            self.db.task_class_probs.bulk_write(updates, ordered=False)

    def _online_update_rho(
        self,
        class_mapping: ClassMapping,
        batch_rho: np.ndarray,
    ) -> None:
        self.db.class_priors.update_many(
            {},  # filter: update all documents
            [
                {
                    "$set": {
                        "prob": {"$multiply": ["$prob", (1 - self.gamma)]},
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
            self.db.class_priors.bulk_write(ops, ordered=False)

    @abstractmethod
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        "Online update_pi"

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


class SparseMongoOnlineAlgorithm(
    MongoOnlineAlgorithm,
):
    """Batch EM is the same as in OnlineAlgorithm.
    Global rho, pi and T are stored sparsly in MongoDB.
    used in class SparseMongoOnlineAlgorithm"""

    @profile
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
                task_index = task_mapping[task_id]
                worker_index = worker_mapping[worker_id]
                class_index = class_mapping[class_id]

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
        batch_matrix: sp.COO,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
    ) -> sp.COO:
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
    def _online_update_T(
        self,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
        batch_T: sp.COO,
        top_k: int | None = None,
    ) -> None:
        """
        If top_k is None -> keep all classes (old behavior).
        If top_k is set   -> keep only top_k classes per task.
        """
        scale = 1 - self.gamma

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

        task_names = np.array(
            [self._reverse_task_mapping[t] for t in uniq_tasks],
        )
        class_names = np.array(
            [self._reverse_class_mapping[c] for c in uniq_classes],
        )
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        docs = self.db.task_class_probs.find(
            {"_id": {"$in": task_names.tolist()}},
            {"_id": 1, "probs": 1},
        )
        task_to_probs = {doc["_id"]: doc.get("probs", {}) for doc in docs}

        existing = np.zeros_like(block)
        for i, task_name in enumerate(task_names):
            current_probs = task_to_probs.get(task_name, {})
            for cls, val in current_probs.items():
                j = class_to_idx.get(cls)
                if j is not None:
                    existing[i, j] = val * scale

        probs_matrix = existing + block

        if top_k is not None and top_k < probs_matrix.shape[1]:
            idx = np.argpartition(probs_matrix, -top_k, axis=1)[:, -top_k:]
            mask = np.zeros_like(probs_matrix, dtype=bool)
            rows = np.arange(len(uniq_tasks))[:, None]
            mask[rows, idx] = True
            probs_matrix[~mask] = 0.0

        updates = []
        for i, task_name in enumerate(task_names):
            row = probs_matrix[i]
            nz = row.nonzero()[0]
            if nz.size == 0:
                continue

            merged = {class_names[j]: float(row[j]) for j in nz}
            update_doc = {
                "$set": {f"probs.{cls}": val for cls, val in merged.items()},
            }

            if top_k is not None:
                existing_names = set(task_to_probs.get(task_name, {}).keys())
                touched_names = set(class_names[block[i, :] > 0])
                unset_names = (existing_names | touched_names) - merged.keys()
                if unset_names:
                    update_doc["$unset"] = {
                        f"probs.{cls}": "" for cls in unset_names
                    }

            updates.append(
                UpdateOne({"_id": task_name}, update_doc, upsert=True),
            )

        if updates:
            with self.mongo_timer("online update class probs"):
                self.db.task_class_probs.bulk_write(updates, ordered=False)
            self._normalize_probs(task_names.tolist())

    @abstractmethod
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        "Online update pi"

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


class RetroactiveSparseMongoOnlineAlgorithm(SparseMongoOnlineAlgorithm):
    def __init__(
        self,
        recursion_limit: Annotated[int, Ge(0)] = 5,
        *args,
        **kwargs,
    ) -> None:
        self.recursion_limit = validate_recursion_limit(recursion_limit)
        super().__init__(*args, **kwargs)

    @profile
    def _normalize_probs(self, updated_task_ids: list[str]) -> None:
        current_time = datetime.now(tz=UTC)
        docs = list(
            self.db.task_class_probs.find(
                {"_id": {"$in": updated_task_ids}},
                {
                    "_id": 1,
                    "probs": 1,
                    "current_answer": 1,
                    "answer_history": 1,
                },
            ),
        )
        updates = []
        for doc in docs:
            probs = doc.get("probs", {})
            total = sum(probs.values())
            if total == 0:
                normalized = dict.fromkeys(probs, 0)
                current_answer = None
            else:
                normalized = {k: v / total for k, v in probs.items()}
                current_answer = max(normalized, key=normalized.get)

            # Initialize answer_history if it doesn't exist
            if "answer_history" not in doc:
                answer_history = []
            else:
                answer_history = doc["answer_history"]

            # Store the previous answer in history if it exists and is different from the new one
            if (
                "current_answer" in doc
                and doc["current_answer"] != current_answer
            ):
                answer_history.append(
                    {
                        "timestamp": current_time,
                        "answer": doc["current_answer"],
                    },
                )

            # Prepare the update operation with history
            update_fields = {
                "probs": normalized,
                "current_answer": current_answer,
                "answer_history": answer_history,
                "last_reviewed_answer": doc.get(
                    "last_reviewed_answer",
                ),  # preserve existing
            }

            updates.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": update_fields},
                ),
            )
        if updates:
            self.db.task_class_probs.bulk_write(updates, ordered=False)

    @profile
    def process_batch(
        self,
        batch: AnswersDict,
        maxiter: int = 50,
        epsilon: float = 1e-6,
        _depth: int = 0,
    ) -> list[float]:
        ll = super().process_batch(batch, maxiter, epsilon)

        self._perform_retroactive_updates(_depth=_depth + 1)
        return ll

    @profile
    def _perform_retroactive_updates(self, _depth: int = 0) -> None:
        raise NotImplementedError
        if _depth > self.recursion_limit:
            msg = f"Max recursion depth {self.recursion_limit} exceeded"
            print(msg)
            return

        # Check if there are any task_class_probs documents
        if not self.db.task_class_probs.count_documents({}, limit=1):
            return

        try:
            # Get all tasks that have answer_history with at least one entry
            # TODO and id should be in current tasks ,

            current_tasks = list(self._reverse_task_mapping.values())

            cursor = self.db.task_class_probs.find(
                {
                    "_id": {"$in": current_tasks},
                    "answer_history": {"$exists": True, "$not": {"$size": 0}},
                },
                {
                    "_id": 1,
                    "current_answer": 1,
                    "answer_history": 1,
                    "last_reviewed_answer": 1,
                },
            )
            docs = list(cursor)

            changed_tasks = set()
            # For each task, check if current_answer differs from the last answer in history
            for doc in docs:
                print(f"{doc=}")
                if "current_answer" not in doc:
                    continue

                last_answer = doc["answer_history"][-1]["answer"]
                last_reviewed = doc.get("last_reviewed_answer")

                # Only trigger if current_answer != last_answer AND we haven’t already reviewed it
                if (
                    doc["current_answer"] != last_answer
                    and doc["current_answer"] != last_reviewed
                ):
                    changed_tasks.add(doc["_id"])

            if not changed_tasks:
                return

            # Get all votes for changed tasks in one query
            task_docs = list(
                self.db.user_votes.find(
                    {"_id": {"$in": list(changed_tasks)}},
                    {"_id": 1, "votes": 1},
                ),
            )

            retro_batch = {
                doc["_id"]: doc.get("votes", {})
                for doc in task_docs
                if doc.get("votes")
            }

            if retro_batch:
                print(
                    f"Performing retroactive update for {len(retro_batch)} tasks",
                )
                self.process_batch(retro_batch, maxiter=3, _depth=_depth + 1)

                # Mark tasks as reviewed (store the current answer we reviewed against)
                self.db.task_class_probs.update_many(
                    {"_id": {"$in": list(retro_batch.keys())}},
                    {
                        "$set": {"last_reviewed_answer": None},
                    },  # will overwrite below
                )

                for doc in task_docs:
                    self.db.task_class_probs.update_one(
                        {"_id": doc["_id"]},
                        {
                            "$set": {
                                "last_reviewed_answer": retro_batch[
                                    doc["_id"]
                                ],
                            },
                        },
                    )

        except Exception as e:
            print(f"Error in retroactive updates: {e!s}")
            raise
