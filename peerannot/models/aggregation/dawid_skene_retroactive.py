from __future__ import annotations

from peerannot.models.aggregation.dawid_skene_online import DawidSkeneOnline
from peerannot.models.aggregation.online_helpers import RetroactiveAlgorithm


class DawidSkeneRetroactive(DawidSkeneOnline, RetroactiveAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
