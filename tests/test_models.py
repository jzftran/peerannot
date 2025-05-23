import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from peerannot.models import DawidSkene

dir_toydata = Path(__file__).parents[1] / "datasets" / "toy-data"


def load_json(path):
    with open(path) as f:
        return json.load(f)


ANSWERS = load_json(dir_toydata / "answers.json")


def test_mv():
    sparse = [True, False]
    from peerannot.models import MajorityVoting

    for sparse_ in sparse:
        mv = MajorityVoting(ANSWERS, n_classes=2, sparse=sparse_)
        y = mv.get_answers()
        expected = np.array([1, 0, 1])
        assert all([e == y_ for e, y_ in zip(expected, y)])


def test_ns():
    from peerannot.models import NaiveSoft as NS

    ns = NS(ANSWERS, n_classes=2)
    y = ns.get_probas()
    expected = np.array([[0.25, 0.75], [2 / 3, 1 / 3], [0, 1]])
    assert y.shape == (3, 2)
    assert np.isclose((expected - y).sum(), 0)


def test_ds():
    from peerannot.models import DawidSkene as DS

    ds = DS(answers=ANSWERS, n_classes=2, n_workers=4)
    ds.run(maxiter=10)
    y = ds.get_probas()
    assert y.shape == (3, 2)
    y = ds.get_answers()
    expected = np.array([1, 0, 1])
    assert all([e == y_ for e, y_ in zip(expected, y)])

    expected = np.array([[0.25, 0.75], [2 / 3, 1 / 3], [0, 1]])
    assert np.isclose((expected - ds.get_probas()).sum(), 0)


def test_ds_wrong_parameters():
    with pytest.raises(ValidationError) as exc_info:
        ds = DawidSkene(answers=ANSWERS, n_classes=-2, n_workers=4)

    assert "Input should be greater than or equal to 1" in str(exc_info.value)


def test_wawa():
    from peerannot.models import Wawa

    sparse = [True, False]
    for sparse_ in sparse:
        wawa = Wawa(ANSWERS, n_classes=2, n_workers=4, sparse=sparse_)
        wawa.run()
        y = wawa.get_answers()
        expected = np.array([1, 0, 1])
        assert np.isclose(
            (wawa.worker_score - np.array([1 / 2, 1, 1, 1 / 2])).sum(),
            0,
        )
        assert all([e == y_ for e, y_ in zip(expected, y)])
        # assert True == False


def test_IWMV():
    from peerannot.models import IWMV

    sparse = [True, False]
    for sparse_ in sparse:
        wawa_it = IWMV(ANSWERS, n_classes=2, n_workers=4, sparse=sparse_)
        wawa_it.run(10)
        y = wawa_it.get_answers()
        expected = np.array([1, 0, 1])
        assert np.isclose(
            (wawa_it.worker_score - np.array([1 / 2, 1, 1, 1 / 2])).sum(),
            0,
        )
        assert all([e == y_ for e, y_ in zip(expected, y)])


def test_twothird():
    from peerannot.models import TwoThird

    sparse = [True, False]
    for sparse_ in sparse:
        two = TwoThird(ANSWERS, n_classes=2, n_workers=4, sparse=sparse_)
        y = two.get_answers()
        expected = np.array([1, 0, 1])
        assert all([e == y_ for e, y_ in zip(expected, y)])


def test_glad():
    from peerannot.models import GLAD

    glad = GLAD(
        ANSWERS,
        n_classes=2,
        n_workers=4,
        dataset=dir_toydata / "temp",
    )
    glad.run()
    y = glad.get_answers()
    expected = np.array([1, 0, 1])
    assert y.shape == (3,)
    assert all([e == y_ for e, y_ in zip(expected, y)])
    assert glad.beta.shape == (3,)
    assert glad.alpha.shape == (4,)
    assert (
        len(
            list(
                (dir_toydata / "temp" / "identification" / "glad").glob(
                    "*.npy",
                ),
            ),
        )
        == 2
    )
    shutil.rmtree(dir_toydata / "temp")  # cleanup


def test_plantnet():
    from peerannot.models import PlantNet

    pn = PlantNet(
        ANSWERS,
        n_classes=2,
        n_workers=4,
        alpha=0.5,
        beta=0.2,
        AI="ignored",
        authors=None,
    )
    pn.run(maxiter=2)
    y = pn.get_answers()
    expected = np.array([1, 0, 1])
    assert y.shape == (3,)
    assert all([e == y_ for e, y_ in zip(expected, y)])
