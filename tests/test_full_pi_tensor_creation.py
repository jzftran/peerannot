import mongomock
import numpy as np
import pytest

from peerannot.models.aggregation.dawid_skene_online import (
    VectorizedDawidSkeneOnlineMongo,
)
from peerannot.models.aggregation.diagonal_multinomial_online import (
    VectorizedDiagonalMultinomialOnlineMongo,
)
from peerannot.models.aggregation.flat_single_binomial_online import (
    VectorizedFlatSingleBinomialOnlineMongo,
)
from peerannot.models.aggregation.multinomial_binary_online import (
    VectorizedMultinomialBinaryOnlineMongo,
)
from peerannot.models.aggregation.pooled_diagonal_multinomial_online import (
    VectorizedPooledDiagonalMultinomialOnlineMongo,
)
from peerannot.models.aggregation.pooled_flat_diagonal_online import (
    VectorizedPooledFlatDiagonalOnlineMongo,
)
from peerannot.models.aggregation.pooled_flat_single_binomial_online import (
    VectorizedPooledFlatSingleBinomialOnlineMongo,
)
from peerannot.models.aggregation.pooled_multinomial_binary_online import (
    VectorizedPooledMultinomialBinaryOnlineMongo,
)
from peerannot.models.aggregation.pooled_multinomial_online import (
    VectorizedPooledMultinomialOnlineMongo,
)

MODELS = [
    VectorizedDiagonalMultinomialOnlineMongo,
    VectorizedFlatSingleBinomialOnlineMongo,
    VectorizedMultinomialBinaryOnlineMongo,
    VectorizedPooledFlatDiagonalOnlineMongo,
    VectorizedPooledDiagonalMultinomialOnlineMongo,
    VectorizedPooledFlatSingleBinomialOnlineMongo,
    VectorizedPooledMultinomialBinaryOnlineMongo,
    VectorizedPooledMultinomialOnlineMongo,
    VectorizedDawidSkeneOnlineMongo,
]

# Save the original implementation
_orig_add_update = mongomock.collection.BulkOperationBuilder.add_update


def _patched_add_update(self, selector, update, multi, upsert, **kwargs):
    """
    Patch mongomock BulkOperationBuilder.add_update to ignore
    unsupported kwargs (sort, hint, collation, array_filters).
    """
    return _orig_add_update(self, selector, update, multi, upsert)


@pytest.fixture(autouse=True, scope="session")
def patch_mongomock_bulk_update():
    # Apply monkeypatch globally for all tests
    mongomock.collection.BulkOperationBuilder.add_update = _patched_add_update
    yield
    # Restore original after tests
    mongomock.collection.BulkOperationBuilder.add_update = _orig_add_update


@pytest.mark.parametrize("ModelCls", MODELS)
def test_build_full_pi_tensor_row_normalization(ModelCls):
    """
    For each model class, verify that build_full_pi_tensor()
    returns a row-normalized confusion tensor:
        sum over axis=2 == 1.0
    """
    model = ModelCls(mongo_client=mongomock.MongoClient())
    model.drop()  # reset DB and state

    # Simulate two small batches with overlapping and new class ids
    batch1 = {0: {0: 0}, 1: {1: 1}, 2: {2: 0}}
    batch2 = {0: {3: 1, 4: 1}, 3: {2: 1, 4: 0}, 4: {2: 1, 4: 2}}

    model.process_batch(batch1)
    model.process_batch(batch2)

    pi_tensor = model.build_full_pi_tensor()
    assert pi_tensor.ndim >= 2, (
        f"{ModelCls.__name__} returned invalid tensor shape"
    )

    if pi_tensor.ndim == 3:
        summed = pi_tensor.sum(axis=2)
    else:
        raise AssertionError(f"Unexpected tensor shape: {pi_tensor.shape}")

    # Check row sums are close to 1

    is_valid = np.all(np.isin(summed, [0.0, 1.0]))

    assert is_valid, f"{ModelCls.__name__}: Row sums not normalized\n{summed}"
