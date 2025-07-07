import numpy as np

from peerannot.models.aggregation.dawid_skene_online import DawidSkeneMongo

expected_confusion_matrices = [
    {
        "_id": 0,
        "confusion_matrix": [
            {"from_class_id": 0, "to_class_id": 0, "prob": 1.0},
        ],
    },
    {
        "_id": 1,
        "confusion_matrix": [
            {"from_class_id": 1, "to_class_id": 1, "prob": 1.0},
        ],
    },
    {
        "_id": 2,
        "confusion_matrix": [
            {"from_class_id": 0, "to_class_id": 0, "prob": 0.3402460446135528},
            {"from_class_id": 1, "to_class_id": 1, "prob": 1.0},
            {"from_class_id": 0, "to_class_id": 1, "prob": 0.6597539553864472},
            {"from_class_id": 2, "to_class_id": 1, "prob": 1.0},
        ],
    },
    {
        "_id": 3,
        "confusion_matrix": [
            {"from_class_id": 1, "to_class_id": 1, "prob": 1.0},
        ],
    },
    {
        "_id": 4,
        "confusion_matrix": [
            {"from_class_id": 1, "to_class_id": 1, "prob": 0.5},
            {"from_class_id": 1, "to_class_id": 0, "prob": 0.25},
            {"from_class_id": 1, "to_class_id": 2, "prob": 0.25},
            {"from_class_id": 0, "to_class_id": 0, "prob": 1.0},
            {"from_class_id": 2, "to_class_id": 2, "prob": 1.0},
        ],
    },
]


def sort_conf_matrix(matrix):
    """Sorts matrix entries to allow consistent comparison"""
    return sorted(matrix, key=lambda x: (x["from_class_id"], x["to_class_id"]))


def test_dawid_skene_confusion_matrices():
    dsm = DawidSkeneMongo()

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
