from itertools import chain

import numpy as np
import pytest

from peerannot.models.aggregation.dawid_skene_online import (
    DawidSkeneMongo,
    DawidSkeneOnline,
)
from peerannot.models.aggregation.online_helpers import (
    batch_generator_by_task,
    batch_generator_by_user,
    batch_generator_by_vote,
)

batch1 = {
    0: {0: 0},
    1: {1: 1},
    2: {2: 0},
}

batch2 = {
    0: {3: 1, 4: 1},
    3: {2: 1, 4: 0},
    4: {2: 1, 4: 2},
}


expected_confusion_matrices = [
    {
        "_id": "0",
        "confusion_matrix": [
            {"from_class_id": 0, "to_class_id": 0, "prob": 1.0},
        ],
    },
    {
        "_id": "1",
        "confusion_matrix": [
            {"from_class_id": 1, "to_class_id": 1, "prob": 1.0},
        ],
    },
    {
        "_id": "2",
        "confusion_matrix": [
            {"from_class_id": 0, "to_class_id": 0, "prob": 0.3402460446135528},
            {"from_class_id": 1, "to_class_id": 1, "prob": 1.0},
            {"from_class_id": 0, "to_class_id": 1, "prob": 0.6597539553864472},
            {"from_class_id": 2, "to_class_id": 1, "prob": 1.0},
        ],
    },
    {
        "_id": "3",
        "confusion_matrix": [
            {"from_class_id": 1, "to_class_id": 1, "prob": 1.0},
            {"from_class_id": 0, "to_class_id": 1, "prob": 1.0},
        ],
    },
    {
        "_id": "4",
        "confusion_matrix": [
            {"from_class_id": 1, "to_class_id": 1, "prob": 0.2538683445334543},
            {"from_class_id": 1, "to_class_id": 0, "prob": 0.3730658277332728},
            {"from_class_id": 1, "to_class_id": 2, "prob": 0.3730658277332728},
            {"from_class_id": 0, "to_class_id": 1, "prob": 0.568874072230784},
            {"from_class_id": 0, "to_class_id": 0, "prob": 0.431125927769216},
            {"from_class_id": 2, "to_class_id": 2, "prob": 1.0},
        ],
    },
]

expected_T = np.array(
    [
        [0.77552133, 0.22447867, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ],
)


def sort_conf_matrix(matrix):
    """Sorts matrix entries to allow consistent comparison"""
    return sorted(matrix, key=lambda x: (x["from_class_id"], x["to_class_id"]))


def test_dawid_skene_confusion_matrices():
    dsm = DawidSkeneMongo()
    dsm.drop()
    try:
        dsm.process_batch(batch1)
        dsm.process_batch(batch2)

        actual = list(dsm.db.worker_confusion_matrices.find({}))
        actual = sorted(actual, key=lambda x: x["_id"])

        assert len(actual) == len(expected_confusion_matrices)

        for expected, actual_doc in zip(actual, expected_confusion_matrices):
            assert expected["_id"] == actual_doc["_id"]

            expected_matrix = sort_conf_matrix(
                expected["confusion_matrix"],
            )
            actual_matrix = sort_conf_matrix(
                actual_doc["confusion_matrix"],
            )

            assert len(expected_matrix) == len(actual_matrix)

            for exp, act in zip(expected_matrix, actual_matrix):
                assert exp["from_class_id"] == act["from_class_id"]
                assert exp["to_class_id"] == act["to_class_id"]
                np.testing.assert_allclose(
                    exp["prob"],
                    act["prob"],
                    rtol=1e-6,
                    atol=1e-8,
                )

    finally:
        dsm.drop()


def test_dawid_skene_online_process_batch():
    dso = DawidSkeneOnline()
    dso.process_batch(batch1)
    dso.process_batch(batch2)

    np.testing.assert_allclose(dso.T, expected_T, rtol=1e-6, atol=1e-8)


def test_basic_batching():
    answers = {
        "obs1": {"u1": 10},
        "obs2": {"u2": 20},
        "obs3": {"u3": 30},
        "obs4": {"u4": 40},
    }
    result = list(batch_generator_by_task(answers, 2))
    expected = [
        {"obs1": {"u1": 10}, "obs2": {"u2": 20}},
        {"obs3": {"u3": 30}, "obs4": {"u4": 40}},
    ]
    assert result == expected


def test_leftover_batch():
    answers = {
        "obs1": {"u1": 10},
        "obs2": {"u2": 20},
        "obs3": {"u3": 30},
    }
    result = list(batch_generator_by_task(answers, 2))
    expected = [
        {"obs1": {"u1": 10}, "obs2": {"u2": 20}},
        {"obs3": {"u3": 30}},
    ]
    assert result == expected


def test_single_element_batches():
    answers = {
        "obs1": {"u1": 10},
        "obs2": {"u2": 20},
    }
    result = list(batch_generator_by_task(answers, 1))
    expected = [
        {"obs1": {"u1": 10}},
        {"obs2": {"u2": 20}},
    ]
    assert result == expected


def test_batch_size_larger_than_input():
    answers = {
        "obs1": {"u1": 10},
        "obs2": {"u2": 20},
    }
    result = list(batch_generator_by_task(answers, 10))
    expected = [answers]
    assert result == expected


def test_empty_input():
    answers = {}
    result = list(batch_generator_by_task(answers, 3))
    assert result == []


@pytest.mark.parametrize("bad_size", [0, -1, -5])
def test_invalid_batch_size(bad_size):
    answers = {"obs1": {"u1": 10}}
    with pytest.raises(ValueError):
        list(list(batch_generator_by_task(answers, bad_size)))


def test_single_user_per_batch():
    answers = {
        "obs1": {"u1": 10},
        "obs2": {"u2": 20},
        "obs3": {"u1": 30},
    }
    result = list(batch_generator_by_user(answers, 1))

    # Expected: one batch per user
    # order may vary since sets are unordered
    possible_batches = [
        [
            {"obs1": {"u1": 10}, "obs3": {"u1": 30}},  # user u1
            {"obs2": {"u2": 20}},  # user u2
        ],
        [
            {"obs2": {"u2": 20}},
            {"obs1": {"u1": 10}, "obs3": {"u1": 30}},
        ],
    ]
    assert result in possible_batches


def test_multiple_users_in_batch():
    answers = {
        "obs1": {"u1": 10, "u2": 20},
        "obs2": {"u3": 30},
    }
    result = list(batch_generator_by_user(answers, 2))

    # Up to 2 users per batch
    all_users = set(
        chain.from_iterable(obs.keys() for obs in answers.values()),
    )
    # Flatten batches to verify coverage
    batched_users = set().union(
        *(
            batch_user
            for b in result
            for batch_user in chain.from_iterable(b.values())
        ),
    )

    assert batched_users == set(answers["obs1"].values()) | set(
        answers["obs2"].values(),
    )
    assert all(len(set(chain.from_iterable(b.values()))) <= 2 for b in result)


def test_batch_size_larger_than_users():
    answers = {
        "obs1": {"u1": 10, "u2": 20},
        "obs2": {"u3": 30},
    }
    result = list(batch_generator_by_user(answers, 10))

    # All users should appear in a single batch
    assert len(result) == 1
    assert result[0] == answers


def test_empty_answers():
    answers = {}
    result = list(batch_generator_by_user(answers, 2))
    assert result == []


@pytest.mark.parametrize("bad_size", [0, -1, -10])
def test_invalid_batch_size(bad_size):
    answers = {"obs1": {"u1": 10}}
    with pytest.raises(
        ValueError,
        match="batch_size must be a positive integer",
    ):
        list(list(batch_generator_by_user(answers, bad_size)))


def test_overlapping_users():
    answers = {
        "obs1": {"u1": 10, "u2": 20},
        "obs2": {"u2": 30, "u3": 40},
    }
    result = list(batch_generator_by_user(answers, 1))

    # Each batch corresponds to one user
    users_seen = [
        set(batch_user for obs in batch.values() for batch_user in obs)
        for batch in result
    ]
    all_users = {u for obs in answers.values() for u in obs}
    assert set().union(*users_seen) == all_users
    # Ensure each batch contains only one user's answers
    assert all(len(u) == 1 for u in users_seen)


def test_single_vote_per_batch():
    answers = {
        "obs1": {"u1": 10},
        "obs2": {"u2": 20},
        "obs3": {"u3": 30},
    }
    result = list(batch_generator_by_vote(answers, 1))
    expected = [
        {"obs1": {"u1": 10}},
        {"obs2": {"u2": 20}},
        {"obs3": {"u3": 30}},
    ]
    assert result == expected


def test_two_votes_per_batch():
    answers = {
        "obs1": {"u1": 10, "u2": 20},
        "obs2": {"u3": 30},
    }
    result = list(batch_generator_by_vote(answers, 2))

    expected = [
        {"obs1": {"u1": 10, "u2": 20}},  # first two votes
        {"obs2": {"u3": 30}},  # last one
    ]
    assert result == expected


def test_batch_size_larger_than_total_votes():
    answers = {
        "obs1": {"u1": 10, "u2": 20},
        "obs2": {"u3": 30},
    }
    result = list(batch_generator_by_vote(answers, 10))

    # Everything should fit into one batch
    assert result == [answers]


def test_empty_input():
    answers = {}
    result = list(batch_generator_by_vote((answers, 2)))
    assert result == []


@pytest.mark.parametrize("bad_size", [0, -1, -5])
def test_invalid_batch_size(bad_size):
    answers = {"obs1": {"u1": 10}}
    with pytest.raises(
        ValueError,
        match="batch_size must be a positive integer",
    ):
        list(batch_generator_by_vote(answers, bad_size))


def test_split_votes_within_observation():
    answers = {
        "obs1": {"u1": 10, "u2": 20, "u3": 30},
    }
    result = list(batch_generator_by_vote(answers, 2))

    expected = [
        {"obs1": {"u1": 10, "u2": 20}},  # first 2 votes
        {"obs1": {"u3": 30}},  # leftover
    ]
    assert result == expected
