import numpy as np

from ..template import CrowdModel


class NaiveSoft(CrowdModel):
    """
    ===================================
    Naive soft: Frequency distribution
    ===================================
    """

    def __init__(self, answers, n_classes=2, **kwargs):
        """Naive soft: Frequency distribution of labels

        .. math::

            \\mathrm{NaiveSoft}(i, \\mathcal{D}) = \\left(\\sum_{j\\in\\mathcal{A}(x_i)}\\mathbf{1}(y_i^{(j)} = k)\\right)_{k\\in[K]}


        :param answers: Dictionary of workers answers with format

         .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict
        :param n_classes: Number of possible classes, defaults to 2
        :type n_classes: int, optional
        """
        super().__init__(answers)
        self.n_classes = n_classes

    def get_probas(self):
        """Get soft labels distribution for each task

        :return: Label frequency for each task
        :rtype: numpy.ndarray(n_task, n_classes)
        """
        baseline = np.zeros((len(self.answers), self.n_classes))
        for task_id in list(self.answers.keys()):
            task = self.answers[task_id]
            for vote in list(task.values()):
                baseline[task_id, vote] += 1
        self.baseline = baseline
        return baseline / baseline.sum(axis=1).reshape(-1, 1)

    def get_answers(self):
        """Argmax of soft labels, in this case corresponds to a majority vote

        :return: Hard labels (majority vote)
        :rtype: numpy.ndarray
        """
        return np.vectorize(self.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1),
        )
