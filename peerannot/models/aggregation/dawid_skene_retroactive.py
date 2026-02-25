from __future__ import annotations

from peerannot.models.aggregation.online_helpers import RetroactiveAlgorithm


class DawidSkeneRetroactive(DawidSkeneBatch, RetroactiveAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
