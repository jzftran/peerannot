class DidNotConverge(RuntimeWarning):
    def __init__(
        self,
        algorithm_name: str,
        eps: float,
        epsilon: float,
        *args,
    ) -> None:
        super().__init__(
            f"{algorithm_name} did not converge: err={eps}, {epsilon=}.",
            *args,
        )


class NotInitialized(RuntimeWarning):
    def __init__(
        self,
        algorithm_name: str,
        *args,
    ) -> None:
        super().__init__(
            f"{algorithm_name} not initialized - process at least one batch first.",
            *args,
        )
