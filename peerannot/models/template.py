"""
=================================
Parent template to all strategies
=================================
"""

from typing import Any
from peerannot.helpers.converters import AnswersDict, Converter


class CrowdModel:
    def __init__(self, answers: AnswersDict) -> None:
        self.converter = Converter(answers)
        transformed_answers = self.converter.transform()
        self.answers: dict[int, Any] = {
            int(k): v
            for k, v in sorted(
                transformed_answers.items(),
                key=lambda item: int(item[0]),
            )
        }
