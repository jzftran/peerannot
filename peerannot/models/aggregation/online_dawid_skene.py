from __future__ import annotations

from typing import Annotated

import numpy as np
import sparse as sp
from annotated_types import Ge, Gt
from line_profiler import profile

from peerannot.models.aggregation.dawid_skene_batch import (
    VectorizedDawidSkeneBatchMongo,
    WeightedDawidSkene,
)
from peerannot.models.aggregation.mongo_online_helpers import (
    SparseMongoOnlineAlgorithm,
)
from peerannot.models.aggregation.types import (
    ClassMapping,
    TaskMapping,
    WorkerMapping,
)


class MinibatchOnlineDawidSkene(
    VectorizedDawidSkeneBatchMongo,
    SparseMongoOnlineAlgorithm,
):
    @profile
    def _load_pi_from_db(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
    ) -> sp.COO:
        """
        Load worker confusion matrices from DB and project into batch tensor.

        Output shape:
            (n_workers, n_batch_classes, n_batch_classes)

        Behavior:
            - If worker has no DB matrix -> prior.
            - If row at least partially specified the rest is set to 1e-12.
        """

        n_workers = len(worker_mapping)
        n_batch_classes = len(class_mapping)

        # TODO move to init
        if self.n_classes <= 1:
            default_diag = 1.0
            default_off_diag = 0.0
        else:
            default_diag = 0.7
            default_off_diag = (1.0 - default_diag) / (self.n_classes - 1)
        ####

        pi = np.full(
            (n_workers, n_batch_classes, n_batch_classes),
            default_off_diag,
        )

        diag_idx = np.arange(n_batch_classes)

        pi[:, diag_idx, diag_idx] = default_diag
        for w in range(n_workers):
            np.fill_diagonal(pi[w], default_diag)

        cursor = self.db.worker_confusion_matrices.find(
            {"_id": {"$in": list(worker_mapping.keys())}},
            {"_id": 1, "confusion_matrix": 1},
        )

        for worker_doc in cursor:
            worker_id = worker_doc["_id"]
            w_idx = worker_mapping.get(worker_id)

            if w_idx is None:
                continue

            entries = worker_doc.get("confusion_matrix", [])
            if not entries:
                continue  # keep prior

            # Collect row-wise updates (batch index space)
            row_updates: dict[int, dict[int, float]] = {}

            for entry in entries:
                from_name = entry.get("from_class")
                to_name = entry.get("to_class")

                l_idx = class_mapping.get(from_name)
                k_idx = class_mapping.get(to_name)

                if l_idx is None or k_idx is None:
                    continue  # class not in batch

                prob = float(entry.get("prob", 0.0))
                row_updates.setdefault(l_idx, {})[k_idx] = prob

            for l_idx, col_probs in row_updates.items():
                unspecified_cols = [
                    k for k in range(n_batch_classes) if k not in col_probs
                ]

                # set DB probabilities
                for k_idx, prob in col_probs.items():
                    pi[w_idx, l_idx, k_idx] = prob

                for k_idx in unspecified_cols:
                    pi[w_idx, l_idx, k_idx] = (
                        1e-12  # clips the rest to small value
                    )
        return pi


class MajorityVotingOnlineDawidSkene(
    VectorizedDawidSkeneBatchMongo,
    SparseMongoOnlineAlgorithm,
):
    """
    Cappé-style online EM:
    For each task, update sufficient statistics with a stochastic
    approximation step and immediately refresh parameters.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_processed = 0
        self._total_task_count = 0

    @profile
    def _load_rho_from_db(self, class_mapping: dict[str, int]) -> sp.COO:
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

    @profile
    def _load_pi_from_db(
        self,
        worker_mapping: dict[str, int],
        class_mapping: dict[str, int],
    ) -> sp.COO:
        n_workers = len(worker_mapping)
        n_classes = len(class_mapping)

        pi = np.zeros((n_workers, n_classes, n_classes), dtype=np.float64)

        if n_workers == 0 or n_classes == 0:
            return sp.COO(pi)

        worker_ids = list(worker_mapping.keys())

        cursor = self.db.worker_confusion_matrices.find(
            {"_id": {"$in": worker_ids}},
            {"_id": 1, "confusion_matrix": 1},
        )

        for doc in cursor:
            worker_id = doc["_id"]

            w_idx = worker_mapping.get(worker_id)
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

        # Normalize missing rows
        row_sums = pi.sum(axis=2, keepdims=True)
        zero_rows = row_sums == 0

        if np.any(zero_rows):
            pi[zero_rows.repeat(n_classes, axis=2)] = 1.0 / n_classes

        return sp.COO(pi)

    @property
    def total_task_count(self) -> int:
        """
        Returns the total task count. If manually set, use that value.
        Otherwise, use estimated_document_count().
        """
        return (
            self._total_task_count
            if self._total_task_count is not None
            else self.db.task_mapping.estimated_document_count()
        )

    @total_task_count.setter
    def total_task_count(self, value: int):
        """
        Allows manual setting of the total task count.
        """
        self._total_task_count = value

    @property
    def gamma(self):
        n = self.total_task_count
        tau = 1000
        kappa = 0.6
        return (n + tau) ** -kappa

    @profile
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

        global_reverse_task_mapping = self._reverse_task_mapping.copy()
        global_reverse_worker_mapping = self._reverse_worker_mapping.copy()
        global_reverse_class_mapping = self._reverse_class_mapping.copy()

        # TODO add avg_pi

        for batch_task_idx in task_mapping.values():
            self.total_task_count += 1

            mask = batch_matrix.coords[0] == batch_task_idx
            task_w = batch_matrix.coords[1][mask]
            task_k = batch_matrix.coords[2][mask]
            task_name = global_reverse_task_mapping[batch_task_idx]
            task_votes = {
                global_reverse_worker_mapping[
                    int(w)
                ]: global_reverse_class_mapping[int(k)]
                for w, k in zip(task_w, task_k)
            }
            task_batch = {task_name: task_votes}
            task_mapping_task: TaskMapping = {}
            worker_mapping_task: WorkerMapping = {}
            class_mapping_task: ClassMapping = {}
            self._prepare_mapping(
                task_batch,
                task_mapping_task,
                worker_mapping_task,
                class_mapping_task,
            )
            task_matrix = self._process_batch_to_matrix(
                task_batch,
                task_mapping_task,
                worker_mapping_task,
                class_mapping_task,
            )

            # load global pi/rho but restricted to task-specific classes/workers
            batch_rho = self._load_rho_from_db(class_mapping_task)
            batch_pi = self._load_pi_from_db(
                worker_mapping_task,
                class_mapping_task,
            )

            # e-step
            batch_T, batch_denom_e_step = self._e_step(
                task_matrix,
                batch_pi,
                batch_rho,
            )

            self._online_update(
                task_mapping_task,
                worker_mapping_task,
                class_mapping_task,
                sp.COO(batch_T),
                global_rho,
                global_pi,
                task_matrix,
            )

            likeli = float(np.sum(np.log(batch_denom_e_step + 1e-12)))
            ll.append(likeli)

        self._reverse_task_mapping = global_reverse_task_mapping
        self._reverse_worker_mapping = global_reverse_worker_mapping
        self._reverse_class_mapping = global_reverse_class_mapping

        return ll


class WeightedMinibatchOnlineDawidSkene(
    MinibatchOnlineDawidSkene,
    WeightedDawidSkene,
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class WeightedMajorityVotingOnlineDawidSkene(
    MajorityVotingOnlineDawidSkene,
    WeightedDawidSkene,
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
