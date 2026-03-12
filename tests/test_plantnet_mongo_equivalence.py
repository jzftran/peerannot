import numpy as np
import pytest

from peerannot.models.aggregation.plantnet import PlantNet, PlantNetMongo

pymongo = pytest.importorskip("pymongo")
MongoClient = pymongo.MongoClient

testcontainers_mongodb = pytest.importorskip("testcontainers.mongodb")
MongoDbContainer = testcontainers_mongodb.MongoDbContainer


@pytest.fixture(scope="module")
def mongo_client():
    """Provide a MongoDB client backed by a Testcontainers MongoDB instance."""
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
    """Provide an isolated database for each test."""
    db_name = "test_plantnet_mongo_equivalence"
    mongo_client.drop_database(db_name)
    yield db_name
    mongo_client.drop_database(db_name)


@pytest.fixture
def answers():
    """Fixture for the standard answers dictionary."""
    return {
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


def _normalize_labels(labels):
    """Convert labels to numpy array with consistent sentinel value."""
    return np.asarray(
        [int(label) if label is not None else -1 for label in labels],
        dtype=int,
    )


@pytest.fixture
def authors():
    """Fixture for the authors list."""
    return [1, 1, 1, 1, 1, 2, 1, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 2, 0, 2]


@pytest.fixture
def authors_file(tmp_path, authors):
    """Fixture for the authors file."""
    authors_file = tmp_path / "authors.txt"
    np.savetxt(authors_file, authors, fmt="%i")
    return authors_file


@pytest.fixture
def authors_dict(authors):
    """Fixture for the authors dictionary."""
    return {str(i): str(author) for i, author in enumerate(authors)}


@pytest.fixture
def dense_model(answers):
    """Fixture for the dense PlantNet model."""
    return PlantNet(
        answers=answers,
        n_classes=9,
        n_workers=4,
        alpha=0.5,
        beta=0.2,
        AI="ignored",
        authors=None,
    )


@pytest.fixture
def mongo_model(mongo_client, clean_db):
    """Fixture for the MongoDB-backed PlantNet model."""
    return PlantNetMongo(
        alpha=0.5,
        beta=0.2,
        mongo_client=mongo_client,
        db_name=clean_db,
    )


@pytest.fixture
def dense_model_with_authors(answers, authors_file):
    """Fixture for the dense PlantNet model with authors."""
    return PlantNet(
        answers=answers,
        n_classes=9,
        n_workers=4,
        alpha=0.5,
        beta=0.2,
        AI="ignored",
        authors=authors_file,
    )


@pytest.fixture
def mongo_model_with_authors(mongo_client, clean_db, authors_dict):
    """Fixture for the MongoDB-backed PlantNet model with authors."""
    return PlantNetMongo(
        alpha=0.5,
        beta=0.2,
        mongo_client=mongo_client,
        db_name=clean_db,
        authors=authors_dict,
    )


def _run_models(dense_model, mongo_model, answers, maxiter=5, epsilon=1e-5):
    """Helper to run both models and return results."""
    dense_model.run(maxiter=maxiter, epsilon=epsilon)
    mongo_model.process_batch(answers, maxiter=maxiter, epsilon=epsilon)
    return dense_model, mongo_model


def test_plantnet_and_plantnet_mongo_same_output(
    dense_model,
    mongo_model,
    answers,
):
    """Test that both models produce the same answers."""
    dense_model, mongo_model = _run_models(dense_model, mongo_model, answers)
    dense_answers = np.asarray(dense_model.get_answers(), dtype=int)
    mongo_answers = _normalize_labels(mongo_model.get_answers())
    np.testing.assert_array_equal(mongo_answers, dense_answers)


def test_plantnet_and_plantnet_mongo_same_weights(
    dense_model,
    mongo_model,
    answers,
):
    """Test that both models produce the same worker weights."""
    dense_model, mongo_model = _run_models(dense_model, mongo_model, answers)
    mongo_weights, _ = _fetch_mongo_worker_data(mongo_model)
    for i, weight in enumerate(dense_model.weights):
        assert np.isclose(weight, mongo_weights[i], rtol=1e-5), (
            f"Worker {i} weights differ: dense={weight}, mongo={mongo_weights[i]}"
        )


def test_plantnet_and_plantnet_mongo_same_nj(
    dense_model,
    mongo_model,
    answers,
):
    """Test that both models produce the same n_j values."""
    dense_model, mongo_model = _run_models(dense_model, mongo_model, answers)
    _, mongo_nj = _fetch_mongo_worker_data(mongo_model)
    for i, nj in enumerate(dense_model.n_j):
        assert np.isclose(nj, mongo_nj[i], rtol=1e-5), (
            f"Worker {i} n_j differ: dense={nj}, mongo={mongo_nj[i]}"
        )


def test_plantnet_and_plantnet_mongo_with_authors(
    dense_model_with_authors,
    mongo_model_with_authors,
    answers,
):
    """Test that both models produce the same results with authors provided."""
    dense_model, mongo_model = _run_models(
        dense_model_with_authors,
        mongo_model_with_authors,
        answers,
    )
    dense_answers = np.asarray(dense_model.get_answers(), dtype=int)
    mongo_answers = _normalize_labels(mongo_model.get_answers())
    np.testing.assert_array_equal(mongo_answers, dense_answers)


def _fetch_mongo_worker_data(mongo_model):
    """Helper to fetch worker data from MongoDB."""
    mongo_weights = {}
    mongo_nj = {}
    for doc in mongo_model.db.worker_confusion_matrices.find({}):
        worker_id = int(doc["_id"])
        mongo_weights[worker_id] = doc["weight"]
        mongo_nj[worker_id] = doc["n_j"]
    return mongo_weights, mongo_nj
