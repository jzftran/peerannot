import numpy as np
import pytest
import sparse as sp

from peerannot.models.aggregation.dawid_skene_batch import (
    DawidSkeneMongo,
    VectorizedDawidSkeneBatchMongo,
)
from peerannot.models.aggregation.diagonal_multinomial_batch import (
    DiagonalMultinomialBatch,
    VectorizedDiagonalMultinomialBatchMongo,
)
from peerannot.models.aggregation.flat_single_binomial_online import (
    FlatSingleBinomialBatch,
    VectorizedFlatSingleBinomialBatchMongo,
)
from peerannot.models.aggregation.multinomial_binary_batch import (
    MultinomialBinaryBatch,
    MultinomialBinaryBatchLogSpace,
    VectorizedMultinomialBinaryBatchMongo,
    VectorizedMultinomialBinaryBatchMongoLogSpace,
)
from peerannot.models.aggregation.pooled_diagonal_multinomial_batch import (
    PooledDiagonalMultinomialBatch,
    VectorizedPooledDiagonalMultinomialBatchMongo,
)
from peerannot.models.aggregation.pooled_flat_diagonal_batch import (
    VectorizedPooledFlatDiagonalBatch,
    VectorizedPooledFlatDiagonalBatchMongo,
)
from peerannot.models.aggregation.pooled_flat_single_binomial_batch import (
    PooledFlatSingleBinomialBatch,
    VectorizedPooledFlatSingleBinomialBatchMongo,
)
from peerannot.models.aggregation.pooled_multinomial_batch import (
    PooledMultinomialBatch,
    VectorizedPooledMultinomialBatchMongo,
)
from peerannot.models.aggregation.pooled_multinomial_binary_batch import (
    PooledMultinomialBinaryBatch,
    VectorizedPooledMultinomialBinaryBatchMongo,
)


@pytest.fixture
def random_data():
    rng = np.random.default_rng(42)
    n_tasks = 5
    n_workers = 4
    n_classes = 3

    batch_matrix = rng.integers(0, 2, size=(n_tasks, n_workers, n_classes))
    batch_T = rng.uniform(0.1, 1.0, size=(n_tasks, n_classes))
    batch_T /= batch_T.sum(axis=1, keepdims=True)

    return batch_matrix, batch_T


# pairs of equivalent models
MODEL_PAIRS = [
    (PooledMultinomialBatch, VectorizedPooledMultinomialBatchMongo),
    (FlatSingleBinomialBatch, VectorizedFlatSingleBinomialBatchMongo),
    (DiagonalMultinomialBatch, VectorizedDiagonalMultinomialBatchMongo),
    (
        MultinomialBinaryBatch,
        VectorizedMultinomialBinaryBatchMongo,
    ),
    (
        VectorizedPooledFlatDiagonalBatch,
        VectorizedPooledFlatDiagonalBatchMongo,
    ),
    (
        PooledDiagonalMultinomialBatch,
        VectorizedPooledDiagonalMultinomialBatchMongo,
    ),
    (
        PooledFlatSingleBinomialBatch,
        VectorizedPooledFlatSingleBinomialBatchMongo,
    ),
    (
        PooledMultinomialBinaryBatch,
        VectorizedPooledMultinomialBinaryBatchMongo,
    ),
    (
        DawidSkeneMongo,
        VectorizedDawidSkeneBatchMongo,
    ),
    (
        MultinomialBinaryBatchLogSpace,
        VectorizedMultinomialBinaryBatchMongoLogSpace,
    ),
]


@pytest.mark.parametrize("loop_cls, vec_cls", MODEL_PAIRS)
def test_m_step_equivalence(loop_cls, vec_cls, random_data):
    batch_matrix, batch_T = random_data

    # run dense
    rho_loop, pi_loop = loop_cls()._m_step(batch_matrix, batch_T)

    # run sparse
    rho_vec, pi_vec = vec_cls()._m_step(sp.COO(batch_matrix), sp.COO(batch_T))

    # compare
    if type(pi_vec) is not sp.COO:
        pi_vec = sp.COO(pi_vec)
    assert np.allclose(rho_loop, rho_vec.todense(), rtol=1e-12, atol=1e-12)
    assert np.allclose(pi_loop, pi_vec.todense(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("loop_cls, vec_cls", MODEL_PAIRS)
def test_e_step_equivalence(loop_cls, vec_cls, random_data):
    batch_matrix, batch_T = random_data

    loop_model, vec_model = loop_cls(), vec_cls()

    rho_loop, pi_loop = loop_model._m_step(batch_matrix, batch_T)
    rho_vec, pi_vec = vec_model._m_step(sp.COO(batch_matrix), sp.COO(batch_T))

    T_loop, denom_loop = loop_model._e_step(batch_matrix, pi_loop, rho_loop)
    T_vec, denom_vec = vec_model._e_step(sp.COO(batch_matrix), pi_vec, rho_vec)

    if type(denom_vec) is sp.COO:
        denom_vec = denom_vec.todense()
    assert np.allclose(T_loop, T_vec.todense(), rtol=1e-12, atol=1e-12)
    assert np.allclose(denom_loop, denom_vec, rtol=1e-12, atol=1e-12)
