from __future__ import annotations

import numpy as np

from peerannot.models.aggregation.mongo_online_helpers import (
    MongoOnlineAlgorithm,
)
from peerannot.models.aggregation.online_helpers import (
    OnlineAlgorithm,
)


class DawidSkeneMongo(MongoOnlineAlgorithm):
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


class DawidSkeneOnline(OnlineAlgorithm):
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
