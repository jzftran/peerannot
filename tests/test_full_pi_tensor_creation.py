import numpy as np
import pytest
from pymongo import MongoClient
from testcontainers.mongodb import MongoDbContainer

from peerannot.models.aggregation.dawid_skene_batch import (
    VectorizedDawidSkeneBatchMongo,
)
from peerannot.models.aggregation.diagonal_multinomial_batch import (
    VectorizedDiagonalMultinomialBatchMongo,
)
from peerannot.models.aggregation.flat_single_binomial_online import (
    VectorizedFlatSingleBinomialBatchMongo,
)
from peerannot.models.aggregation.multinomial_binary_batch import (
    VectorizedMultinomialBinaryBatchMongo,
)
from peerannot.models.aggregation.pooled_diagonal_multinomial_batch import (
    VectorizedPooledDiagonalMultinomialBatchMongo,
)
from peerannot.models.aggregation.pooled_flat_diagonal_batch import (
    VectorizedPooledFlatDiagonalBatchMongo,
)
from peerannot.models.aggregation.pooled_flat_single_binomial_batch import (
    VectorizedPooledFlatSingleBinomialBatchMongo,
)
from peerannot.models.aggregation.pooled_multinomial_batch import (
    VectorizedPooledMultinomialBatchMongo,
)
from peerannot.models.aggregation.pooled_multinomial_binary_batch import (
    VectorizedPooledMultinomialBinaryBatchMongo,
)

MODELS = [
    VectorizedDiagonalMultinomialBatchMongo,
    VectorizedFlatSingleBinomialBatchMongo,
    VectorizedMultinomialBinaryBatchMongo,
    VectorizedPooledFlatDiagonalBatchMongo,
    VectorizedPooledDiagonalMultinomialBatchMongo,
    VectorizedPooledFlatSingleBinomialBatchMongo,
    VectorizedPooledMultinomialBinaryBatchMongo,
    VectorizedPooledMultinomialBatchMongo,
    VectorizedDawidSkeneBatchMongo,
]

# Save the original implementation


@pytest.fixture(scope="session")
def mongo_client():
    with MongoDbContainer("mongo:7.0") as mongo:
        client = MongoClient(mongo.get_connection_url())
        yield client
        client.close()


@pytest.fixture
def clean_mongo_client(mongo_client):
    db_name = "test_peerannot"
    mongo_client.drop_database(db_name)
    yield mongo_client
    mongo_client.drop_database(db_name)


@pytest.mark.parametrize("ModelCls", MODELS)
def test_build_full_pi_tensor_row_normalization(ModelCls, clean_mongo_client):
    """
    For each model class, verify that build_full_pi_tensor()
    returns a row-normalized confusion tensor:
        sum over axis=2 == 1.0
    """

    model = ModelCls(mongo_client=clean_mongo_client)
    model.drop()  # ensure model collections reset

    # Two small batches with overlapping and new class ids
    batch1 = {0: {0: 0}, 1: {1: 1}, 2: {2: 0}}
    batch2 = {0: {3: 1, 4: 1}, 3: {2: 1, 4: 0}, 4: {2: 1, 4: 2}}

    model.process_batch(batch1)
    model.process_batch(batch2)

    pi_tensor = model.build_full_pi_tensor()

    assert pi_tensor.ndim == 3, (
        f"{ModelCls.__name__} returned invalid tensor shape {pi_tensor.shape}"
    )

    row_sums = pi_tensor.sum(axis=2)

    # Some rows may be zero if class never observed.
    # Valid rows must sum to 1.
    valid_mask = row_sums > 0

    assert np.allclose(row_sums[valid_mask], 1.0), (
        f"{ModelCls.__name__}: Row sums not normalized\n{row_sums}"
    )
