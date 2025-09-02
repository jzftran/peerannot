# %%
import numpy as np
import sparse as sp
from pydantic import validate_call

from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.pooled_multinomial_binary_online import (
    VectorizedPooledMultinomialBinaryOnlineMongo,
)
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class PooledFlatSingleBinomialOnline(OnlineAlgorithm):
    @validate_call
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.pi = np.array(0)  # 1 elem np.array

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        # no need to expand_pi
        pass

    def _initialize_pi(self) -> None:
        self.pi = np.array(0.5)

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        sum_diag_votes = np.einsum(
            "tq, tiq ->",
            batch_T,
            batch_matrix,
        )  # trace(T.T @ crowd_matrix)

        batch_total_votes = batch_matrix.sum()
        batch_pi = np.divide(
            sum_diag_votes,
            batch_total_votes,
            out=np.zeros_like(sum_diag_votes),
            where=(batch_total_votes != 0),
        )

        return batch_rho, batch_pi

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_n_il = np.sum(
            batch_matrix,
            axis=1,
        )  # n_tasks, n_classes: sum of votes given by each worker

        batch_n_task, batch_n_classes = batch_n_il.shape
        batch_T = np.zeros((batch_n_task, batch_n_classes))
        print(f"{batch_pi=}")
        for i in range(batch_n_task):
            n_i_k = batch_n_il[i]  # shape (K,)

            for l in range(batch_n_classes):
                p_lk = ((1 - batch_pi) * batch_rho) / (1 - batch_rho[l])
                p_lk[l] = batch_pi

                batch_T[i, l] = np.prod(np.power(p_lk, n_i_k)) * batch_rho[l]

        print(f"{p_lk=}")
        # Normalize
        T = batch_T
        print(f"{T=}")
        batch_denom_e_step = batch_T.sum(axis=1, keepdims=True)
        batch_T = np.where(
            batch_denom_e_step > 0,
            batch_T / batch_denom_e_step,
            batch_T,
        )

        print(f"{batch_T=}")
        return batch_T, batch_denom_e_step

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ):
        self.pi = self.pi + self.gamma * (batch_pi - self.pi)


class VectorizedPooledFlatSingleBinomialOnlineMongo(
    VectorizedPooledMultinomialBinaryOnlineMongo,
):
    @validate_call
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _e_step(
        self,
        batch_matrix: sp.COO,
        batch_pi: sp.COO,
        batch_rho: sp.COO,
    ):
        batch_n_il = np.sum(batch_matrix, axis=1)  # shape (n_tasks, n_classes)

        denom = 1 - batch_rho
        p_lk = ((1 - batch_pi) * batch_rho[None, :]) / denom[
            :,
            None,
        ]  # shape (K, K)
        p_lk = p_lk.todense()
        np.fill_diagonal(p_lk, batch_pi)

        powers = np.power(
            p_lk[None, :, :],
            batch_n_il[:, None, :],
        )  # shape (n_task, K, K)

        T = np.prod(powers, axis=2)  # shape (n_task, K)

        T *= batch_rho[None, :]  # shape (n_task, K)

        batch_denom_e_step = T.sum(axis=1, keepdims=True).todense()
        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)
        return batch_T, batch_denom_e_step
