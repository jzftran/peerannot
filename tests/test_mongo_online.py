from unittest.mock import MagicMock

import mongomock
import numpy as np
import numpy.testing as npt
import pytest
import sparse as sp
from pymongo import UpdateOne

from peerannot.models.aggregation.mongo_online_helpers import (
    MongoOnlineAlgorithm,
    SparseMongoOnlineAlgorithm,
)


class MongoOnlineTest(MongoOnlineAlgorithm):
    @property
    def pi(self):
        pass

    def _online_update_pi(self, worker_mapping, class_mapping, batch_pi):
        pass

    def _e_step(self, batch_matrix, batch_pi, batch_rho):
        pass

    def _m_step(self, batch_matrix, batch_T):
        pass


class SparseMongoOnlineTest(SparseMongoOnlineAlgorithm):
    """Minimal concrete implementation for testing."""

    def pi(self):
        pass

    def _online_update_pi(self, *args, **kwargs):
        pass

    def _e_step(self, *args, **kwargs):
        pass

    def _m_step(self, *args, **kwargs):
        pass


@pytest.fixture
def mock_model():
    class DummyModel:
        def __init__(self):
            self.task_mapping = {}
            self.worker_mapping = {}
            self.class_mapping = {}
            self.t = 42

            self.get_or_create_indices = MagicMock()
            self._em_loop_on_batch = MagicMock(
                return_value=[100.0, 95.0, 93.5],
            )
            self.log_batch_summary = MagicMock()

            self.process_batch_matrix = (
                MongoOnlineTest.process_batch_matrix.__get__(
                    self,
                )
            )

    return DummyModel()


@pytest.fixture
def mock_model_process_em():
    class DummyModel:
        def __init__(self):
            self.task_mapping = {}
            self.worker_mapping = {}
            self.class_mapping = {}
            self.get_or_create_indices = MagicMock()

            self._init_T = MagicMock(
                return_value=np.array([[0.5, 0.5], [0.5, 0.5]]),
            )
            self._m_step = MagicMock(
                return_value=(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]])),
            )
            self._e_step = MagicMock(
                return_value=(np.array([[0.5, 0.5]]), np.array([1.0, 1.0])),
            )
            self.log_em_iter = MagicMock()
            self._online_update = MagicMock()

            self._em_loop_on_batch = MongoOnlineTest._em_loop_on_batch.__get__(
                self,
            )
            self.get_or_create_indices = MagicMock()

        process_batch_matrix = MongoOnlineTest.process_batch_matrix

    return DummyModel()


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


@pytest.fixture
def repo():
    # in-memory mock mongo client
    client = mongomock.MongoClient()

    return MongoOnlineTest(mongo_client=client)


@pytest.fixture
def repo_sparse():
    client = mongomock.MongoClient()

    return SparseMongoOnlineTest(mongo_client=client)


def test_insert_batch_inserts_and_updates(repo):
    batch1 = {
        "task1": {"user1": 1, "user2": 7},
        "task2": {"user3": 1},
    }

    repo.insert_batch(batch1)

    doc1 = repo.db.user_votes.find_one({"_id": "task1"})
    doc2 = repo.db.user_votes.find_one({"_id": "task2"})

    assert doc1["votes"] == {"user1": 1, "user2": 7}
    assert doc2["votes"] == {"user3": 1}

    batch2 = {
        "task1": {"user1": 7},
        "task2": {"user4": 1},
    }

    repo.insert_batch(batch2)

    doc1 = repo.db.user_votes.find_one({"_id": "task1"})
    doc2 = repo.db.user_votes.find_one({"_id": "task2"})

    assert doc1["votes"] == {"user1": 7, "user2": 7}
    assert doc2["votes"] == {"user3": 1, "user4": 1}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("task.1", "task__DOT__1"),
        ("Passiflora spp.", "Passiflora spp__DOT__"),
        ("nodots", "nodots"),
        ("", ""),
    ],
)
def test_escape_replaces_dots(raw, expected):
    assert MongoOnlineAlgorithm._escape_id(raw) == expected


@pytest.mark.parametrize(
    "escaped,expected",
    [
        ("task__DOT__1", "task.1"),
        ("Passiflora spp__DOT__", "Passiflora spp."),
        ("nodots", "nodots"),
        ("", ""),
    ],
)
def test_unescape_replaces_marker(escaped, expected):
    assert MongoOnlineAlgorithm._unescape_id(escaped) == expected


def test_escape_and_unescape_are_inverse():
    ids = ["task.1", "Passiflora spp.", "nodots", ""]
    for i in ids:
        assert (
            MongoOnlineAlgorithm._unescape_id(
                MongoOnlineAlgorithm._escape_id(i),
            )
            == i
        )


@pytest.mark.parametrize("value", [123, 0, None, True])
def test_non_string_inptuts_are_stringified(value):
    # escape should cast non-str to str
    escaped = MongoOnlineAlgorithm._escape_id(value)
    assert isinstance(escaped, str)

    # unescape should also cast to str
    unescaped = MongoOnlineAlgorithm._unescape_id(value)
    assert isinstance(unescaped, str)


def test_counts_reflect_documents(repo):
    # Initially empty
    assert repo.n_classes == 0
    assert repo.n_workers == 0
    assert repo.n_task == 0

    # Insert some docs into collections
    repo.class_mapping.insert_many([{"_id": 1}, {"_id": 2}])
    repo.worker_mapping.insert_one({"_id": "w1"})
    repo.task_mapping.insert_many(
        [{"_id": "t1"}, {"_id": "t2"}, {"_id": "t3"}],
    )

    assert repo.n_classes == 2
    assert repo.n_workers == 1
    assert repo.n_task == 3


@pytest.mark.parametrize(
    "gamma0,t,decay,expected",
    [
        (1.0, 1, 0.5, 1.0),
        (2.0, 2, 1.0, 1.0),
        (10.0, 10, 0.5, 10 / (10**0.5)),
    ],
)
def test_gamma_computation(repo, gamma0, t, decay, expected):
    repo.gamma0 = gamma0
    repo.t = t
    repo.decay = decay
    assert pytest.approx(repo.gamma) == expected


def test_drop_database_removes_data(repo):
    repo.class_mapping.insert_one({"_id": 1})
    assert repo.n_classes == 1

    repo.drop()

    # database should be dropped -> collections empty
    assert repo.client.list_database_names() == []


def test_T_matrix_builds_correctly(repo):
    # Insert task->class probabilities
    repo.db.task_mapping.insert_many(
        [
            {"_id": "t1", "index": 0},
            {"_id": "t2", "index": 1},
        ],
    )
    repo.db.class_mapping.insert_many(
        [
            {"_id": "c1", "index": 0},
            {"_id": "c2", "index": 1},
        ],
    )

    repo.db.task_class_probs.insert_many(
        [
            {"_id": "t1", "probs": {"c1": 0.1, "c2": 0.9}},
            {"_id": "t2", "probs": {"c1": 0.5}},
        ],
    )

    T = repo.T

    assert T.shape == (2, 2)

    npt.assert_allclose(T[0], [0.1, 0.9])
    npt.assert_allclose(T[1], [0.5, 0.0])


def test_rho_array_builds_correctly(repo):
    repo.db.class_mapping.insert_many(
        [
            {"_id": "c1", "index": 0},
            {"_id": "c2", "index": 1},
        ],
    )
    repo.db.class_priors.insert_many(
        [
            {"_id": "c1", "prob": 0.2},
            {"_id": "c2", "prob": 0.8},
        ],
    )

    rho = repo.rho

    assert rho.shape == (2,)
    npt.assert_allclose(rho, [0.2, 0.8])


def test_get_or_create_indices_empty_collection(repo):
    coll = repo.class_mapping

    keys = ["a", "b", "c"]
    result = repo.get_or_create_indices(coll, keys)

    # All keys should be mapped to indices starting at 0
    assert result == {"a": 0, "b": 1, "c": 2}

    # Collection should now contain documents
    docs = list(coll.find({}))
    assert len(docs) == 3
    indices = [doc["index"] for doc in docs]
    assert sorted(indices) == [0, 1, 2]


def test_get_or_create_indices_existing_keys(repo):
    coll = repo.class_mapping

    coll.insert_many([{"_id": "a", "index": 0}, {"_id": "b", "index": 1}])

    keys = ["a", "b", "c"]
    result = repo.get_or_create_indices(coll, keys)

    assert result == {"a": 0, "b": 1, "c": 2}

    docs = list(coll.find({}))
    assert len(docs) == 3
    assert sorted(doc["_id"] for doc in docs) == ["a", "b", "c"]


def test_get_or_create_indices_only_new_keys(repo):
    coll = repo.class_mapping

    keys = ["x", "y"]
    result = repo.get_or_create_indices(coll, keys)
    assert result == {"x": 0, "y": 1}

    more_keys = ["z", "w"]
    result2 = repo.get_or_create_indices(coll, more_keys)
    assert result2 == {"z": 2, "w": 3}

    assert coll.estimated_document_count() == 4


def test_get_or_create_indices_mixed(repo):
    coll = repo.class_mapping

    coll.insert_many([{"_id": "a", "index": 0}])

    keys = ["a", "b", "c"]
    result = repo.get_or_create_indices(coll, keys)

    assert result == {"a": 0, "b": 1, "c": 2}

    docs = list(coll.find({}))
    assert len(docs) == 3
    indices = [doc["index"] for doc in docs]
    assert sorted(indices) == [0, 1, 2]


def test_prepare_mapping_initial(repo):
    batch = {
        "t1": {"w1": "c1", "w2": "c2"},
        "t2": {"w3": "c1"},
    }

    task_mapping = {}
    worker_mapping = {}
    class_mapping = {}

    repo._prepare_mapping(batch, task_mapping, worker_mapping, class_mapping)

    assert task_mapping == {"t1": 0, "t2": 1}
    assert worker_mapping == {"w1": 0, "w2": 1, "w3": 2}
    assert class_mapping == {"c1": 0, "c2": 1}

    assert repo._reverse_task_mapping == {0: "t1", 1: "t2"}
    assert repo._reverse_worker_mapping == {0: "w1", 1: "w2", 2: "w3"}
    assert repo._reverse_class_mapping == {0: "c1", 1: "c2"}


def test_prepare_mapping_with_existing(repo):
    batch1 = {"t1": {"w1": "c1"}}
    batch2 = {"t2": {"w2": "c2"}, "t1": {"w3": "c1"}}

    task_mapping = {}
    worker_mapping = {}
    class_mapping = {}

    repo._prepare_mapping(batch1, task_mapping, worker_mapping, class_mapping)
    repo._prepare_mapping(batch2, task_mapping, worker_mapping, class_mapping)

    assert task_mapping == {"t1": 0, "t2": 1}
    assert worker_mapping == {"w1": 0, "w2": 1, "w3": 2}
    assert class_mapping == {"c1": 0, "c2": 1}

    assert repo._reverse_task_mapping == {0: "t1", 1: "t2"}
    assert repo._reverse_worker_mapping == {0: "w1", 1: "w2", 2: "w3"}
    assert repo._reverse_class_mapping == {0: "c1", 1: "c2"}


def test_prepare_mapping_empty_batch(repo):
    task_mapping = {}
    worker_mapping = {}
    class_mapping = {}

    repo._prepare_mapping({}, task_mapping, worker_mapping, class_mapping)

    assert task_mapping == {}
    assert worker_mapping == {}
    assert class_mapping == {}

    assert repo._reverse_task_mapping == {}
    assert repo._reverse_worker_mapping == {}
    assert repo._reverse_class_mapping == {}


def test_process_batch_to_matrix_basic(repo):
    batch = {
        "t1": {"w1": "c1", "w2": "c2"},
        "t2": {"w3": "c1"},
    }

    task_mapping = {"t1": 0, "t2": 1}
    worker_mapping = {"w1": 0, "w2": 1, "w3": 2}
    class_mapping = {"c1": 0, "c2": 1}

    matrix = repo._process_batch_to_matrix(
        batch,
        task_mapping,
        worker_mapping,
        class_mapping,
    )

    assert matrix.shape == (2, 3, 2)

    assert matrix[0, 0, 0]  # t1, w1, c1
    assert matrix[0, 1, 1]  # t1, w2, c2
    assert matrix[1, 2, 0]  # t2, w3, c1

    total_true = np.sum(matrix)
    assert total_true == 3


def test_process_batch_to_matrix_empty_batch(repo):
    task_mapping = {"t1": 0, "t2": 1}
    worker_mapping = {"w1": 0, "w2": 1}
    class_mapping = {"c1": 0, "c2": 1}

    matrix = repo._process_batch_to_matrix(
        {},
        task_mapping,
        worker_mapping,
        class_mapping,
    )

    assert matrix.shape == (2, 2, 2)
    assert not np.any(matrix)


def test_get_probas_returns_T(repo):
    repo.db.task_mapping.insert_many(
        [
            {"_id": "t1", "index": 0},
            {"_id": "t2", "index": 1},
        ],
    )
    repo.db.class_mapping.insert_many(
        [
            {"_id": "c1", "index": 0},
            {"_id": "c2", "index": 1},
        ],
    )

    repo.db.task_class_probs.insert_many(
        [
            {"_id": "t1", "probs": {"c1": 0.1, "c2": 0.9}},
            {"_id": "t2", "probs": {"c1": 0.5}},  # only one prob
        ],
    )

    T = repo.T

    probas = repo.get_probas()
    npt.assert_allclose(probas, T)


def test_get_answer_returns_current_answer(repo):
    repo.db.task_class_probs.insert_one({"_id": "t1", "current_answer": "c2"})
    assert repo.get_answer("t1") == "c2"


def test_get_answer_missing_field_raises_keyerror(repo):
    repo.db.task_class_probs.insert_one({"_id": "t1"})
    with pytest.raises(KeyError):
        repo.get_answer("t1")


def test_process_batch_increments_t_and_calls_process_batch_matrix(
    mocker,
    repo,
):
    batch = {
        "task.1": {
            "worker1": 0,
            "worker2": 1,
        },  # has a dot in ID to test escaping
        "task2": {"worker3": 1},
    }

    # Patch insert_batch to avoid mongomock bulk_write issues
    mocker.patch.object(repo, "insert_batch", return_value=None)

    # Patch process_batch_matrix to check call behavior
    mocked_matrix = mocker.patch.object(
        repo,
        "process_batch_matrix",
        return_value=[0.1, 0.2, 0.3],
    )

    old_t = repo.t

    result = repo.process_batch(batch, maxiter=10, epsilon=1e-3)

    # Assert result
    assert result == [0.1, 0.2, 0.3]

    # t increments
    assert repo.t == old_t + 1

    # Check that process_batch_matrix was called
    args, kwargs = mocked_matrix.call_args
    batch_matrix, task_mapping, worker_mapping, class_mapping = args[:4]

    assert "task__DOT__1" in task_mapping  # Escaped ID
    assert "worker1" in worker_mapping
    assert "worker2" in worker_mapping
    assert "worker3" in worker_mapping
    assert "0" in class_mapping or 1 in class_mapping  # class IDs mapped


@pytest.fixture
def mock_model_process_em():
    client = mongomock.MongoClient()
    model = MongoOnlineTest(mongo_client=client)
    model.t = 42

    # Mock process_batch_matrix dependencies
    model.get_or_create_indices = MagicMock()
    model.log_batch_summary = MagicMock()

    # Mock _em_loop_on_batch dependencies
    model._em_loop_on_batch = MagicMock(return_value=[100.0, 95.0, 93.5])
    model.log_em_iter = MagicMock()
    model._online_update = MagicMock()

    # Mock _m_step and _e_step to return predictable values
    model._m_step = MagicMock(
        return_value=(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]])),
    )
    model._e_step = MagicMock(
        return_value=(np.array([[0.5, 0.5]]), np.array([2.0, 2.0])),
    )

    # Dummy mappings
    model.task_mapping = {}
    model.worker_mapping = {}
    model.class_mapping = {}

    return model


def test_process_batch_matrix_calls_dependencies(mock_model_process_em):
    batch_matrix = np.random.randint(0, 2, size=(3, 3, 3))  # dummy matrix
    task_mapping = {"t1": 0, "t2": 1}
    worker_mapping = {"w1": 0}
    class_mapping = {"c1": 0, "c2": 1}

    ll = mock_model_process_em.process_batch_matrix(
        batch_matrix,
        task_mapping,
        worker_mapping,
        class_mapping,
        maxiter=10,
        epsilon=1e-5,
    )

    assert ll == [100.0, 95.0, 93.5]

    mock_model_process_em.get_or_create_indices.assert_any_call(
        mock_model_process_em.task_mapping,
        list(task_mapping),
    )
    mock_model_process_em.get_or_create_indices.assert_any_call(
        mock_model_process_em.worker_mapping,
        list(worker_mapping),
    )
    mock_model_process_em.get_or_create_indices.assert_any_call(
        mock_model_process_em.class_mapping,
        list(class_mapping),
    )

    mock_model_process_em._em_loop_on_batch.assert_called_once_with(
        batch_matrix,
        task_mapping,
        worker_mapping,
        class_mapping,
        1e-5,
        10,
    )

    mock_model_process_em.log_batch_summary.assert_called_once()
    args, kwargs = mock_model_process_em.log_batch_summary.call_args
    assert args[0] == 42  # self.t
    assert args[1] == len(task_mapping)
    assert args[2] == len(worker_mapping)
    assert args[3] == len(class_mapping)
    assert args[4] == len(ll)
    assert args[6] == ll[-1]  # last log likelihood


def test_em_loop_on_batch_convergence(repo):
    batch_matrix = np.random.randint(0, 2, size=(3, 3, 3))  # dummy matrix
    task_mapping = {"t1": 0}
    worker_mapping = {"w1": 0}
    class_mapping = {"c1": 0}

    repo._m_step = MagicMock(
        return_value=(np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]])),
    )
    repo._e_step = MagicMock(
        return_value=(np.array([[0.5, 0.5]]), np.array([1.0, 1.0])),
    )
    repo.log_em_iter = MagicMock()
    repo._online_update = MagicMock()

    ll = repo._em_loop_on_batch(
        batch_matrix,
        task_mapping,
        worker_mapping,
        class_mapping,
        epsilon=1e-8,
        maxiter=3,
    )

    assert len(ll) == 2
    expected_ll = [np.log(2.0)] * 2
    np.testing.assert_allclose(ll, expected_ll)

    assert repo.log_em_iter.call_count == 2

    args, kwargs = repo._online_update.call_args

    assert args[0] == task_mapping
    assert args[1] == worker_mapping
    assert args[2] == class_mapping

    np.testing.assert_allclose(args[3], np.array([[0.5, 0.5]]))  # batch_T
    np.testing.assert_allclose(args[4], np.array([[0.5, 0.5]]))  # batch_rho
    np.testing.assert_allclose(args[5], np.array([[0.5, 0.5]]))  # batch_pi


def test_online_update_calls_subupdates(repo):
    task_mapping = {"t1": 0}
    worker_mapping = {"w1": 0}
    class_mapping = {"c1": 0, "c2": 1}

    batch_T = np.array([[0.7, 0.3]])
    batch_rho = np.array([[0.6, 0.4]])
    batch_pi = np.array([[0.5, 0.5]])
    repo.t = 12
    repo._online_update_T = MagicMock()
    repo._online_update_rho = MagicMock()
    repo._online_update_pi = MagicMock()

    repo._online_update(
        task_mapping,
        worker_mapping,
        class_mapping,
        batch_T,
        batch_rho,
        batch_pi,
    )

    repo._online_update_T.assert_called_once_with(
        task_mapping,
        class_mapping,
        batch_T,
        repo.top_k,
    )
    repo._online_update_rho.assert_called_once_with(class_mapping, batch_rho)
    repo._online_update_pi.assert_called_once_with(
        worker_mapping,
        class_mapping,
        batch_pi,
    )


def test_normalize_probs(repo):
    # Insert docs with unnormalized probs
    repo.db.task_class_probs.insert_many(
        [
            {"_id": "t1", "probs": {"c1": 2.0, "c2": 6.0}},  # sums to 8
            {"_id": "t2", "probs": {"c1": 0.0, "c2": 0.0}},  # sums to 0
        ],
    )

    repo._normalize_probs(["t1", "t2"])

    # Fetch back
    t1 = repo.db.task_class_probs.find_one({"_id": "t1"})
    t2 = repo.db.task_class_probs.find_one({"_id": "t2"})

    # Check normalization for t1
    assert pytest.approx(t1["probs"]["c1"]) == 0.25
    assert pytest.approx(t1["probs"]["c2"]) == 0.75
    assert t1["current_answer"] == "c2"  # argmax

    assert all(v == 0 for v in t2["probs"].values())
    assert t2["current_answer"] in t2["probs"]


def test_normalize_probs_bulk_write_called(repo, monkeypatch):
    # Insert doc
    repo.db.task_class_probs.insert_one(
        {"_id": "t3", "probs": {"c1": 1.0, "c2": 3.0}},
    )

    called = {}

    def fake_bulk_write(ops, ordered):
        called["ops"] = ops
        called["ordered"] = ordered

    monkeypatch.setattr(
        repo.db.task_class_probs,
        "bulk_write",
        fake_bulk_write,
    )

    repo._normalize_probs(["t3"])

    assert "ops" in called
    op = called["ops"][0]
    assert isinstance(op, UpdateOne)
    assert op._filter == {"_id": "t3"}
    assert op._doc["$set"]["probs"] == {"c1": 0.25, "c2": 0.75}
    assert op._doc["$set"]["current_answer"] == "c2"
    assert called["ordered"] is False


def test_online_update_T_builds_correct_update(repo, monkeypatch):
    task_mapping = {"t1": 0}
    class_mapping = {"c1": 0, "c2": 1}
    batch_T = np.array([[0.6, 0.4]])
    repo.t = 12
    captured = {}

    def fake_bulk_write(ops, ordered):
        captured["ops"] = ops
        captured["ordered"] = ordered

    monkeypatch.setattr(
        repo.db.task_class_probs,
        "bulk_write",
        fake_bulk_write,
    )

    repo._online_update_T(task_mapping, class_mapping, batch_T, top_k=None)

    # Ensure bulk_write called
    assert "ops" in captured
    assert captured["ordered"] is False
    op = captured["ops"][0]
    assert isinstance(op, UpdateOne)
    assert op._filter == {"_id": "t1"}
    # check the pipeline contains $set for both classes
    set_stage = op._doc[0]["$set"]
    assert "probs.c1" in set_stage
    assert "probs.c2" in set_stage


def test_online_update_T_writes_and_normalizes(repo):
    repo.db.task_class_probs.insert_one(
        {"_id": "t1", "probs": {"c1": 0.5, "c2": 0.5}},
    )
    repo.t = 12341
    repo._normalize_probs = MagicMock()
    task_mapping = {"t1": 0}
    class_mapping = {"c1": 0, "c2": 1}
    batch_T = np.array([[0.6, 0.4]])

    repo._online_update_T(task_mapping, class_mapping, batch_T)

    repo._normalize_probs.assert_called_once_with(["t1"])

    doc = repo.db.task_class_probs.find_one({"_id": "t1"})
    assert "probs" in doc


def test_online_update_T_handles_multiple_tasks(repo):
    task_mapping = {"t1": 0, "t2": 1}
    class_mapping = {"c1": 0}
    batch_T = np.array([[0.7], [0.3]])
    repo.t = 9
    repo._normalize_probs = MagicMock()
    repo._online_update_T(task_mapping, class_mapping, batch_T)

    repo._normalize_probs.assert_called_once_with(["t1", "t2"])

    docs = list(repo.db.task_class_probs.find({}, {"_id": 1}))
    ids = {d["_id"] for d in docs}
    assert ids == {"t1", "t2"}


def test_online_update_rho_scales_and_increments(repo):
    repo.db.class_priors.insert_many(
        [
            {"_id": "c1", "prob": 0.5},
            {"_id": "c2", "prob": 0.5},
        ],
    )

    repo.t = 9
    # repo.gamma = 0.1
    class_mapping = {"c1": 0, "c2": 1}
    batch_rho = np.array([0.6, 0.4])

    repo._online_update_rho(class_mapping, batch_rho)

    docs = {d["_id"]: d["prob"] for d in repo.db.class_priors.find()}
    expected_c1 = 0.5 * 0.9 + 0.6 * 0.1
    expected_c2 = 0.5 * 0.9 + 0.4 * 0.1
    assert np.isclose(docs["c1"], expected_c1)
    assert np.isclose(docs["c2"], expected_c2)


def test_online_update_rho_upserts_new_classes(repo):
    class_mapping = {"c3": 0}
    batch_rho = np.array([1.0])
    repo.t = 9
    repo._online_update_rho(class_mapping, batch_rho)

    doc = repo.db.class_priors.find_one({"_id": "c3"})

    assert np.isclose(doc["prob"], 1.0 * repo.gamma)


def test_online_update_rho_no_ops_when_empty(repo, monkeypatch):
    class_mapping = {}
    batch_rho = np.array([])
    repo.t = 9

    called = {}

    def fake_bulk_write(ops, ordered):
        called["yes"] = True

    monkeypatch.setattr(repo.db.class_priors, "bulk_write", fake_bulk_write)

    repo._online_update_rho(class_mapping, batch_rho)

    assert called == {}


def test_process_batch_to_matrix_basic(repo_sparse):
    batch = {
        "task1": {"worker1": "classA", "worker2": "classB"},
        "task2": {"worker1": "classB"},
    }
    task_mapping = {"task1": 0, "task2": 1}
    worker_mapping = {"worker1": 0, "worker2": 1}
    class_mapping = {"classA": 0, "classB": 1}

    result = repo_sparse._process_batch_to_matrix(
        batch,
        task_mapping,
        worker_mapping,
        class_mapping,
    )

    assert isinstance(result, sp.COO)
    assert result.shape == (2, 2, 2)

    # check coordinates contain expected triples
    coords = set(zip(*result.coords))
    expected = {
        (0, 0, 0),  # task1, worker1, classA
        (0, 1, 1),  # task1, worker2, classB
        (1, 0, 1),  # task2, worker1, classB
    }
    assert coords == expected


def test_process_batch_to_matrix_empty(repo_sparse):
    batch = {}
    task_mapping = {"task1": 0}
    worker_mapping = {"worker1": 0}
    class_mapping = {"classA": 0}

    result = repo_sparse._process_batch_to_matrix(
        batch,
        task_mapping,
        worker_mapping,
        class_mapping,
    )

    assert isinstance(result, sp.COO)
    assert result.nnz == 0
    assert result.shape == (1, 1, 1)


def test_process_batch_to_matrix_single_entry(repo_sparse):
    batch = {"taskX": {"workerY": "classZ"}}
    task_mapping = {"taskX": 0}
    worker_mapping = {"workerY": 0}
    class_mapping = {"classZ": 0}

    result = repo_sparse._process_batch_to_matrix(
        batch,
        task_mapping,
        worker_mapping,
        class_mapping,
    )

    assert result.shape == (1, 1, 1)
    coords = list(zip(*result.coords))
    assert coords == [(0, 0, 0)]
    assert np.all(result.data)


def test_process_batch_to_matrix_another_single_entry(repo_sparse):
    batch = {
        "taskX": {"workerY": "classZ"},
        "taskA": {"workerA": "classA", "workerB": "classC"},
    }
    task_mapping = {"taskX": 0, "taskA": 1}
    worker_mapping = {"workerY": 0, "workerA": 1, "workerB": 2}
    class_mapping = {"classZ": 0, "classA": 1, "classC": 2}

    result = repo_sparse._process_batch_to_matrix(
        batch,
        task_mapping,
        worker_mapping,
        class_mapping,
    )

    assert result.shape == (2, 3, 3)
    coords = list(zip(*result.coords))
    assert coords == [(0, 0, 0), (1, 1, 1), (1, 2, 2)]
    assert np.all(result.data)


def test_init_T_with_empty_db(repo_sparse):
    # COO batch matrix: 2 tasks x 2 workers x 2 classes
    coords = np.array([[0, 1], [0, 1], [0, 1]])
    data = np.array([1, 1])
    batch_matrix = sp.COO(coords, data, shape=(2, 2, 2))

    task_mapping = {"t1": 0, "t2": 1}
    class_mapping = {"c1": 0, "c2": 1}

    # no prior DB state
    result = repo_sparse._init_T(batch_matrix, task_mapping, class_mapping)

    assert isinstance(result, sp.COO)
    dense = result.todense()
    assert dense.shape == (2, 2)
    assert np.allclose(dense.sum(axis=1), 1.0)


def test_init_T_merges_existing_probs(repo_sparse):
    repo_sparse.t = 1
    # 1 task, 1 worker, 2 classes
    coords = np.array([[0, 0], [0, 0], [0, 1]])
    data = np.array([1, 1])
    batch_matrix = sp.COO(coords, data, shape=(1, 1, 2))

    task_mapping = {"tX": 0}
    class_mapping = {"cA": 0, "cB": 1}

    # Insert existing probs in DB
    repo_sparse.db.task_class_probs.insert_one(
        {
            "_id": "tX",
            "probs": {"cA": 0.7, "cB": 0.3},
        },
    )

    result = repo_sparse._init_T(batch_matrix, task_mapping, class_mapping)

    dense = result.todense()
    expected = (1 - repo_sparse.gamma) * np.array(
        [0.5, 0.5],
    ) + repo_sparse.gamma * np.array([0.7, 0.3])

    assert np.allclose(dense[0], expected)


def test_online_update_T_merges_and_top_k(repo_sparse):
    repo_sparse.t = 1

    coords = np.array(
        [[0, 0, 0], [0, 1, 2]],
    )
    data = np.array([0.5, 0.3, 0.2])
    batch_T = sp.COO(coords, data, shape=(1, 3))  # 1 task, 3 classes

    task_mapping = {"task0": 0}
    class_mapping = {"classA": 0, "classB": 1, "classC": 2}

    repo_sparse._reverse_task_mapping = {0: "task0"}
    repo_sparse._reverse_class_mapping = {
        0: "classA",
        1: "classB",
        2: "classC",
    }

    repo_sparse.db.task_class_probs.insert_one(
        {
            "_id": "task0",
            "probs": {"classA": 0.1, "classB": 0.1, "classC": 0.1},
        },
    )

    # Mock _normalize_probs to avoid side effects
    repo_sparse._normalize_probs = MagicMock()

    repo_sparse._online_update_T(task_mapping, class_mapping, batch_T)

    updated = repo_sparse.db.task_class_probs.find_one({"_id": "task0"})
    assert updated is not None

    scale = 1 - repo_sparse.gamma
    expected = {
        "classA": scale * 0.1 + 0.5 * repo_sparse.gamma,
        "classB": scale * 0.1 + 0.3 * repo_sparse.gamma,
        "classC": scale * 0.1 + 0.2 * repo_sparse.gamma,
    }

    for cls, val in expected.items():
        assert abs(updated["probs"][cls] - val) < 1e-8

    # _normalize_probs should be called
    repo_sparse._normalize_probs.assert_called_once_with(["task0"])
