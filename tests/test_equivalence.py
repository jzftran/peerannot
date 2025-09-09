"""Testing the equivalence of the implementations of two methods: dense and sparse."""

import numpy as np
import pytest
import sparse as sp

from peerannot.models.aggregation.flat_single_binomial_online import (
    FlatSingleBinomialOnline,
    VectorizedFlatSingleBinomialOnlineMongo,
)


@pytest.fixture
def loop_model():
    return FlatSingleBinomialOnline()


@pytest.fixture
def vectorized_model():
    return VectorizedFlatSingleBinomialOnlineMongo()


@pytest.fixture
def random_data():
    rng = np.random.default_rng(42)
    n_tasks = 5
    n_workers = 4
    n_classes = 3

    # Random integer counts (votes)
    batch_matrix = rng.integers(0, 2, size=(n_tasks, n_workers, n_classes))

    # Random T matrix, normalized per task
    batch_T = rng.uniform(0.1, 1.0, size=(n_tasks, n_classes))
    batch_T /= batch_T.sum(axis=1, keepdims=True)

    return batch_matrix, batch_T


def test_m_step_equivalence(loop_model, vectorized_model, random_data):
    batch_matrix, batch_T = random_data

    rho_loop, pi_loop = loop_model._m_step(batch_matrix, batch_T)
    rho_vec, pi_vec = vectorized_model._m_step(
        sp.COO(batch_matrix),
        sp.COO(batch_T),
    )

    assert np.allclose(rho_loop, rho_vec.todense(), rtol=1e-12, atol=1e-12), (
        f"rho mismatch:\nloop:\n{rho_loop}\nvectorized:\n{rho_vec}"
    )

    assert np.allclose(pi_loop, pi_vec.todense(), rtol=1e-12, atol=1e-12), (
        f"pi mismatch:\nloop:\n{pi_loop}\nvectorized:\n{pi_vec}"
    )


def test_e_step_equivalence(loop_model, vectorized_model, random_data):
    batch_matrix, batch_T = random_data

    rho_loop, pi_loop = loop_model._m_step(batch_matrix, batch_T)
    rho_vec, pi_vec = vectorized_model._m_step(
        sp.COO(batch_matrix),
        sp.COO(batch_T),
    )

    T_loop, denom_loop = loop_model._e_step(batch_matrix, pi_loop, rho_loop)

    T_vec, denom_vec = vectorized_model._e_step(
        sp.COO(batch_matrix),
        pi_vec,
        rho_vec,
    )

    assert np.allclose(T_loop, T_vec.todense(), rtol=1e-12, atol=1e-12), (
        f"T mismatch:\nloop:\n{T_loop}\nvectorized:\n{T_vec.todense()}"
    )

    assert np.allclose(
        denom_loop,
        denom_vec,
        rtol=1e-12,
        atol=1e-12,
    ), f"denom mismatch:\nloop:\n{denom_loop}\nvectorized:\n{denom_vec}"
