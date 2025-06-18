.. _api_aggregate:

API aggregate
===============

.. automodule:: peerannot.models.aggregation
   :members:
   :no-inherited-members:

.. currentmodule:: peerannot.models.aggregation

The ``aggregate`` strategies are used to combine labels from multiple users into a single label.
This aggregated label can be either a probability distribution (soft label) or a single class label (hard label).

All strategies are located at `this direction on GitHub <https://github.com/peerannot/peerannot/tree/main/peerannot/models/aggregation>`.

All aggregation-based strategies are available running:

.. prompt:: bash

    peerannot agginfo

.. autosummary::
   :recursive:
   :toctree: generated/
   :nosignatures:

    majority_voting.MajorityVoting
    NaiveSoft.NaiveSoft
    Wawa.Wawa
    IWMV.IWMV
    dawid_skene.DawidSkene
    dawid_skene_online.DawidSkeneOnline
    dawid_skene_retroactive.DawidSkeneRetroactive
    multinomial_binary.MultinomialBinary
    pooled_diagonal_multinomial.PooledDiagonalMultinomial
    diagonal_multinomial.DiagonalMultinomial
    diagonal_multinomial.VectorizedDiagonalMultinomial
    GLAD.GLAD
    plantnet.PlantNet
    twothird.TwoThird
    DS_clust.DawidSkeneClust
    WDS.WDS
