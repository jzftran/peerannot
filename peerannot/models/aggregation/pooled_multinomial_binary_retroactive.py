from peerannot.models.aggregation.online_helpers import RetroactiveAlgorithm
from peerannot.models.aggregation.pooled_multinomial_binary_online import (
    PoooledMultinomialBinaryOnline,
)


class PoooledMultinomialBinaryRetroactive(
    PoooledMultinomialBinaryOnline,
    RetroactiveAlgorithm,
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
