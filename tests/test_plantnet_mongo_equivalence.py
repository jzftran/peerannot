import numpy as np
import pytest

from peerannot.models.aggregation.plantnet import PlantNet, PlantNetMongo

pymongo = pytest.importorskip("pymongo")
MongoClient = pymongo.MongoClient

testcontainers_mongodb = pytest.importorskip("testcontainers.mongodb")
MongoDbContainer = testcontainers_mongodb.MongoDbContainer


@pytest.fixture(scope="module")
def mongo_client():
    """
    Provide a MongoDB client backed by a Testcontainers MongoDB instance.
    """
    container = MongoDbContainer("mongo:7.0")
    container.start()

    client = MongoClient(container.get_connection_url())

    try:
        yield client
    finally:
        client.close()
        container.stop()


@pytest.fixture
def clean_db(mongo_client):
    """
    Provide an isolated database for each test.
    """
    db_name = "test_plantnet_mongo_equivalence"
    mongo_client.drop_database(db_name)
    yield db_name
    mongo_client.drop_database(db_name)


def _normalize_labels(labels):
    """
    Convert labels to numpy array with consistent sentinel value.
    """
    return np.asarray(
        [int(label) if label is not None else -1 for label in labels],
        dtype=int,
    )


def test_plantnet_and_plantnet_mongo_same_output(mongo_client, clean_db):
    answers = {
        0: {0: 2, 1: 2, 2: 2},
        1: {0: 6, 1: 2, 3: 2},
        2: {1: 8, 2: 7, 3: 8},
        3: {0: 1, 1: 1, 2: 5},
        4: {2: 4},
        5: {0: 0, 1: 0, 2: 1, 3: 6},
        6: {1: 5, 3: 3},
        7: {0: 3, 2: 6, 3: 4},
        8: {1: 7, 3: 7},
        9: {0: 8, 2: 1, 3: 1},
        10: {0: 0, 1: 0, 2: 1},
        11: {2: 3},
        12: {0: 7, 2: 8, 3: 1},
        13: {1: 3},
        14: {0: 5, 2: 4, 3: 4},
        15: {0: 5, 1: 7},
        16: {0: 0, 1: 4, 3: 4},
        17: {1: 5, 2: 7, 3: 7},
        18: {0: 3},
        19: {1: 7, 2: 7},
    }
    maxiter = 5
    epsilon = 1e-5

    dense_model = PlantNet(
        answers=answers,
        n_classes=9,
        n_workers=4,
        alpha=0.5,
        beta=0.2,
        AI="ignored",
        authors=None,
    )

    dense_model.run(maxiter=maxiter, epsilon=epsilon)
    dense_answers = np.asarray(dense_model.get_answers(), dtype=int)

    mongo_model = PlantNetMongo(
        alpha=0.5,
        beta=0.2,
        mongo_client=mongo_client,
        db_name=clean_db,
    )

    mongo_model.process_batch(
        answers,
        maxiter=maxiter,
        epsilon=epsilon,
    )

    mongo_answers = _normalize_labels(mongo_model.get_answers())

    np.testing.assert_array_equal(mongo_answers, dense_answers)
