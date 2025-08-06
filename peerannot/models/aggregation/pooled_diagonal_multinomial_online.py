from typing import Annotated

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.mongo_online_helpers import (
    SparseMongoOnlineAlgorithm,
)
from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class PooledDiagonalMultinomialOnline(OnlineAlgorithm):
    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(gamma0=gamma0, decay=decay, *args, **kwargs)

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        """Expand the pi array if the number of workers or classes increases."""
        if new_n_classes > self.n_classes:
            self.pi = self._expand_array(
                self.pi,
                (new_n_classes,),
                fill_value=0.0,
            )

    def _initialize_pi(self) -> None:
        self.pi = np.zeros(self.n_classes)

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        # For each class in the batch, map batch class idx to global class idx
        batch_to_global = {
            batch_class_idx: self.class_mapping[class_name]
            for class_name, batch_class_idx in class_mapping.items()
        }
        for i_batch, i_global in batch_to_global.items():
            # for j_batch, j_global in batch_to_global.items():
            self.pi[i_global] = (1 - self.gamma) * self.pi[
                i_global
            ] + self.gamma * batch_pi[i_batch]

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        diag_votes = np.einsum("tq, tiq -> q", batch_T, batch_matrix)
        denom = np.einsum("tq, tij -> q", batch_T, batch_matrix)

        batch_pi = np.divide(
            diag_votes,
            denom,
            out=np.zeros_like(diag_votes),
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

        batch_pi_non_diag_values = (np.ones_like(batch_pi) - batch_pi) / (
            batch_n_tasks
        )

        T = np.zeros((batch_n_tasks, batch_n_classes))

        for i in range(batch_n_tasks):
            for j in range(batch_n_classes):
                worker_labels = batch_matrix[i]
                diag_contrib = np.prod(
                    np.power(batch_pi[j], worker_labels[:, j]),
                )
                mask = np.ones(batch_n_classes, dtype=bool)
                mask[j] = False
                off_diag_contrib = np.prod(
                    np.power(
                        batch_pi_non_diag_values[mask],
                        worker_labels[:, mask],
                    ),
                )

                T[i, j] = diag_contrib * off_diag_contrib * batch_rho[j]

        batch_denom_e_step = T.sum(1, keepdims=True)
        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)

        return batch_T, batch_denom_e_step


class VectorizedPooledDiagonalMultinomialOnlineMongo(
    SparseMongoOnlineAlgorithm,
):
    """Vectorized pooled diagonal multinomial binary online algorithm
    using sparse matrices and mongo."""

    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(0)

        self.batch_T = batch_T
        self.batch_matrix = batch_matrix

        # Calculate diag_votes
        # diag_votes = np.einsum("tq, tiq -> q", batch_T, batch_matrix).todense()

        expanded_T = batch_T[:, None, :]
        # Multiply elementwiseResult shape: (t, i, q)
        product = expanded_T * batch_matrix
        # Sum over t and i -> axis=(0,1)
        diag_votes = product.sum(axis=(0, 1)).todense()  # shape: (q,)

        # calculate denom
        # denom = np.einsum("tq, tij -> q", batch_T, batch_matrix).todense()
        batch_T_reshaped = batch_T[:, :, None, None]  # shape: (t, q, 1, 1)

        # Reshape batch_matrix to (t, 1, i, j)
        batch_matrix_reshaped = batch_matrix[
            :,
            None,
            :,
            :,
        ]  # shape: (t, 1, i, j)

        # Elementwise multiplication, result: shape (t, q, i, j)
        product = batch_T_reshaped * batch_matrix_reshaped

        # Now sum over t, i, and j to get shape (q,)
        denom = product.sum(axis=(0, 2, 3)).todense()  # axis 0=t, 2=i, 3=j

        batch_pi = np.divide(
            diag_votes,
            denom,
            where=denom != 0,
        )
        self.batch_pi = batch_pi

        return batch_rho, batch_pi

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,  # shape: (n_batch_classes,)
    ) -> None:
        class_docs = list(
            self.db.class_mapping.find(
                {"_id": {"$in": list(class_mapping.keys())}},
            ),
        )
        batch_to_global = {
            class_mapping[doc["_id"]]: doc["index"] for doc in class_docs
        }
        pooled_doc = self.db.worker_confusion_matrices.find_one(
            {"_id": "pooled"},
        )
        confusion_matrix = (
            pooled_doc.get("confusion_matrix", []) if pooled_doc else []
        )

        entry_map = {
            entry["class_id"]: entry
            for entry in confusion_matrix
            if "class_id" in entry
        }
        print(f"{entry_map=}")
        for i_batch, i_global in batch_to_global.items():
            batch_prob = batch_pi[i_batch]

            if i_global in entry_map:
                entry = entry_map[i_global]
                entry["prob"] = (1 - self.gamma) * entry[
                    "prob"
                ] + self.gamma * batch_prob
            else:
                if batch_prob == 0:
                    continue
                entry = {
                    "class_id": i_global,
                    "prob": self.gamma * batch_prob,
                }
                confusion_matrix.append(entry)
                entry_map[i_global] = entry

        self.db.worker_confusion_matrices.update_one(
            {"_id": "pooledDiagonalMultinomial"},
            {"$set": {"confusion_matrix": confusion_matrix}},
            upsert=True,
        )

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ):
        n_classes = len(batch_pi)

        diag_terms = np.power(batch_pi[None, None, :], batch_matrix)

        batch_pi_non_diag_values = (1.0 - batch_pi) / (n_classes - 1)

        # shape (n_task, n_worker, n_classes)
        all_terms = np.power(
            batch_pi_non_diag_values[None, None, :],
            batch_matrix,
        )

        # Build mask (shape: n_classes x n_classes, False on diagonal)
        mask = np.ones((n_classes, n_classes), dtype=bool)
        np.fill_diagonal(mask, False)

        # Broadcast all_terms to (n_task, n_worker, n_classes, n_classes)
        all_terms_expanded = np.broadcast_to(
            all_terms[:, :, None, :],
            (*all_terms.shape[:2], n_classes, n_classes),
        )

        # Apply broadcasted mask (shape: (1, 1, n_classes, n_classes))
        masked_terms = np.where(mask[None, None, :, :], all_terms_expanded, 1)

        # Take product over last axis -> shape (n_task, n_worker, n_classes)
        off_diag_prod = np.prod(masked_terms, axis=3)
        combined_prod = diag_terms * off_diag_prod

        worker_prod = np.prod(combined_prod, axis=1)  # (n_task, n_classes)

        T = batch_rho[None, :] * worker_prod

        batch_denom_e_step = np.sum(T, axis=1, keepdims=True)

        self.Tmatrix = T
        self.batch_denom_e_step = batch_denom_e_step

        batch_T = np.where(
            batch_denom_e_step > 0,
            T / batch_denom_e_step.todense(),
            T,
        )

        return batch_T, batch_denom_e_step
