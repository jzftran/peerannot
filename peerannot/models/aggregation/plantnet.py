import json
import warnings
from collections import defaultdict
from collections.abc import Hashable
from pathlib import Path
from typing import Optional

import numpy as np
from pymongo import UpdateOne
from tqdm.auto import tqdm

from peerannot.models.aggregation.mongo_online_helpers import (
    SparseMongoBatchAlgorithm,
)
from peerannot.models.aggregation.warnings_errors import NotInitialized
from peerannot.models.template import CrowdModel

THETACONF = 2
THETAACC = 0.7


class PlantNet(CrowdModel):
    """
    ===================================
    PlantNet aggregation strategy
    ===================================

    Weighted majority vote based on the number of identified
    classes (species) per worker. Each task if either valid
    (:math:`s_i=1` or not) if the confidence and accuracy in the estimated
    label are above the set thresholds.
    """

    def __init__(
        self,
        answers,
        n_classes,
        AI="ignored",
        parrots="ignored",
        alpha=1,
        beta=1,
        AIweight=1,  # if AI is fixed or invalidating
        authors=None,  # path to txt file containing authors id for each task
        scores=None,  # path to txt file containing scores for each task
        threshold_scores=None,  # threshold for scores
        **kwargs,
    ):
        r"""Compute a weighted majority vote based on the number of identified
        classes (species) per worker

        :param answers: dictionary of workers answers with format

         .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict
        :param n_classes: Number of possible classes (should be high)
        :type n_classes: int
        :param AI: How to consider entries with `worker=AI` in the dictionnary of answers, defaults to "ignored". Several options are available:

            - ignored: ignore the AI labels
            - worker: consider the AI as a worker
            - fixed: consider the AI as a worker with a fixed weight=`AIweight`
            - invalidating: consider the AI as a worker with a weight=`AIweight` that can only invalidate the tasks
            - confident: consider the AI as a worker with a weight=`AIweight` if the predicted score is above the threshold `threshold_scores`

        :type AI: str, optional
        :param parrots: How to deal with parrot answers, defaults to "ignored" (not implemented yet)
        :type parrots: str, optional
        :param alpha: Value of :math:`\\alpha` parameter in weight function, defaults to 1
        :type alpha: float, optional
        :param beta: Value of :math:`\\beta` parameter in weight function, defaults to 1
        :type beta: float, optional
        :param AIweight: Weight of the AI if not ignored, defaults to 1
        :type AIweight: float, optional
        :param authors: Path to txt file containing authors id for each task
        :type authors: str, optional
        :param scores: Path to json file containing AI prediction scores for each task
        :type scores: str, optional
        :param threshold_scores: Threshold for AI prediction scores if AI strategy is set to `confident`
        :type threshold_scores: float between 0 and 1, optional
        """
        self.AI = AI
        super().__init__(answers)
        self.n_workers = kwargs["n_workers"]
        self.parrots = parrots
        self.alpha = alpha
        self.beta = beta
        if self.AI == "ignored":
            for task in self.answers:
                self.answers[task] = {
                    k: v for k, v in self.answers[task].items() if k != "AI"
                }
            self.weight_AI = -1
        elif self.AI == "worker":
            self.n_workers += 1
            for task in self.answers:
                for worker, label in self.answers[task].items():
                    if worker == "AI":
                        self.answers[task][self.n_workers] = self.answers[
                            task
                        ]["AI"].pop(worker)
            self.weight_AI = -1
        elif self.AI == "fixed" or self.AI == "invalidating":
            self.weight_AI = AIweight
            ans_ai = -np.ones(len(self.answers), dtype=int)
            for i, task in enumerate(self.answers):
                for worker, label in self.answers[task].items():
                    if worker == "AI":
                        ans_ai[i] = int(label)
                self.answers[task] = {
                    k: v for k, v in self.answers[task].items() if k != "AI"
                }
            self.ans_ai = ans_ai
        elif self.AI == "confident":
            self.weight_AI = AIweight
            ans_ai = -np.ones(len(self.answers), dtype=int)
            for i, task in enumerate(self.answers):
                for worker, label in self.answers[task].items():
                    if worker == "AI":
                        ans_ai[i] = int(label)
                self.answers[task] = {
                    k: v for k, v in self.answers[task].items() if k != "AI"
                }
            self.ans_ai = ans_ai
            with open(scores) as f:
                self.scores = json.load(f)
            self.scores = np.array(list(self.scores.values()))
            self.scores_threshold = threshold_scores
        else:
            raise ValueError(
                f"Option {self.AI} should be one of worker, fixed, invalidating, confident or ignored",
            )
        self.n_classes = n_classes
        self.authors = authors
        if self.authors is None:
            self.authors = -np.ones(len(self.answers), dtype=int)
        else:
            self.authors = np.loadtxt(self.authors, dtype=int)
        if kwargs.get("dataset"):
            self.path_save = (
                Path(kwargs["dataset"]) / "identification" / "plantnet"
            )
        else:
            self.path_save = None

    def get_wmv(self, weights):
        """Compute weighted majority vote

        :param weights: Weights of each worker
        :type weights: np.ndarray of size n_workers
        :return: Most weighted labels
        :rtype: np.ndarray of size n_task
        """

        def calculate_init():
            """WMV by task"""
            init = np.zeros(self.n_classes)
            for worker, label in self.answers[i].items():
                init[label] += weights[int(worker)]
            return init

        def calculate_yhat(i):
            """Run WMV by task and add AI vote depending on the strategy

            :param i: Task index
            :type i: int
            :return: Most weighted label
            :rtype: int
            """
            init = calculate_init()
            if (
                self.AI == "fixed"
                or (
                    self.AI == "confident"
                    and self.scores[i] >= self.scores_threshold
                )
            ) and self.ans_ai[i] != -1:
                init[self.ans_ai[i]] += self.weight_AI
            return np.argmax(init)

        yhat = np.zeros(self.n_task)
        for i in range(self.n_task):
            yhat[i] = calculate_yhat(i)
        return yhat

    def get_conf_acc(self, yhat, weights):
        """Compute confidence and accuracy scores for each task

        :param yhat: Estimated labels
        :type yhat: np.ndarray of size n_task
        :param weights: Weights of each worker
        :type weights: np.ndarray of size n_workers
        """

        def calculate_conf_acc(i):
            """Compute confidence and accuracy scores

            .. math::

                \\mathrm{conf}_i(\\hat y_i) = \\sum_{j\\in \\mathcal{A}(x_i)} w_j \\mathbf{1}(y_i^{(j)}=\\hat y_i)

            .. math::

                \\mathrm{acc}_i(\\hat y_i) = \\mathrm{conf}_i(\\hat y_i) / \\sum_{k\\in [K]} \\mathrm{conf}_i(k)

            :param i: task index
            :type i: int
            :return: (acc, conf) scores
            :rtype: tuple of float
            """
            sum_weights = 0
            conf = 0
            for worker, label in self.answers[i].items():
                if worker != "AI":
                    sum_weights += weights[int(worker)]
                    conf += weights[int(worker)] * (label == yhat[i])
                if self.AI == "fixed":
                    sum_weights += self.weight_AI
                    conf += self.weight_AI * (self.ans_ai[i] == yhat[i])
                if self.AI == "invalidating":
                    if conf / (sum_weights + self.weight_AI) < THETAACC:
                        sum_weights += self.weight_AI
                if (
                    self.AI == "confident"
                    and self.scores[i] >= self.scores_threshold
                ):
                    sum_weights += self.weight_AI
                    conf += self.weight_AI * (self.ans_ai[i] == yhat[i])
            acc = conf / sum_weights
            return acc, conf

        acc = np.zeros(self.n_task)
        conf = np.zeros(self.n_task)
        for i in range(self.n_task):
            acc[i], conf[i] = calculate_conf_acc(i)
        return acc, conf

    def get_valid_tasks(self, acc, conf):
        """Compute mask for valid observations (:math:`s_i=1`):

        .. math::

            s_i=1 \\text{ if } \\mathrm{conf}_i > \\theta_{\\text{conf}} \\text{ and } \\mathrm{acc}_i > \\theta_{\\text{acc}}

        """
        valid = np.zeros(self.n_task)
        mask = np.where((conf > THETACONF) & (acc > THETAACC), True, False)
        valid[mask] = 1
        return valid

    def get_weights(self):
        """Compute weight transformation

        :return: Weight of each worker:

         .. math::

            w_j = \\alpha^{n_j} - \\beta^{n_j} + \\log(2.1)

        :rtype: np.ndarray of size n_workers
        """
        return self.n_j**self.alpha - self.n_j**self.beta + np.log(2.1)

    def get_n(self, valid, yhat):
        """Compute the number of identified classes

        :param valid: Indicator of valid tasks
        :type valid: np.ndarray of size n_task
        :param yhat: Estimated labels
        :type yhat: np.ndarray of size n_task
        """
        taxa_obs = np.zeros(self.n_workers)
        taxa_votes = np.zeros(self.n_workers)
        dico_labs_workers = {k: {} for k in range(self.n_workers)}
        for task_id, label_task in zip(self.answers.keys(), yhat):
            for worker, lab_worker in self.answers[task_id].items():
                if worker != "AI":
                    if lab_worker == label_task:
                        if self.authors[int(task_id)] == int(worker):
                            if valid[int(task_id)] == 1:
                                if (
                                    dico_labs_workers[int(worker)].get(
                                        lab_worker,
                                        None,
                                    )
                                    is None
                                ):
                                    taxa_obs[int(worker)] += 1
                                    dico_labs_workers[int(worker)][
                                        lab_worker
                                    ] = 1
        for task_id, label_task in zip(self.answers.keys(), yhat):
            for worker, lab_worker in self.answers[task_id].items():
                if worker != "AI":
                    if lab_worker == label_task:
                        if (
                            dico_labs_workers[int(worker)].get(
                                lab_worker,
                                None,
                            )
                            is None
                        ):
                            taxa_votes[int(worker)] += 1 / 10
                            dico_labs_workers[int(worker)][lab_worker] = 1
        self.n_j = np.array(
            [
                taxa_obs[w] + np.round(taxa_votes[w])
                for w in range(self.n_workers)
            ],
        )

    def run(self, maxiter=100, epsilon=1e-5):  # epsilon = diff in weights
        """Run the PlantNet aggregation algorithm

        :param maxiter: Maximum number of iterations in the EM, defaults to 100 (at least 5)
        :type maxiter: int, optional
        :param epsilon: Stopping criterion if weights are not updated anymore, defaults to 1e-5
        :type epsilon: float, optional
        """
        self.n_task = len(self.answers)
        valid = np.ones(self.n_task)
        weights = np.log(2.1) * np.ones(self.n_workers)
        # print("Begin WMV init")
        init_yhat = self.get_wmv(weights)
        # print("Begin acc, conf init")
        acc, conf = self.get_conf_acc(init_yhat, weights)
        valid = self.get_valid_tasks(acc, conf)
        self.get_n(valid, init_yhat)
        for step in tqdm(range(maxiter)):
            n_j = self.n_j
            weights = self.get_weights()
            yhat = self.get_wmv(weights)
            acc, conf = self.get_conf_acc(init_yhat, weights)
            valid = self.get_valid_tasks(acc, conf)
            self.get_n(valid, init_yhat)
            if (
                np.sum(np.abs(self.n_j - n_j)) / self.n_task <= epsilon
                and step > 5
            ):
                break
        self.labels_hat = yhat if maxiter > 1 else init_yhat
        self.valid = valid
        self.weights = weights
        self.conf = conf
        self.acc = acc

    def get_answers(self):
        """:return: Hard labels and None when no consensus is reached
        :rtype: numpy.ndarray
        """
        ans = self.labels_hat
        if self.path_save:
            noconsensus = np.where(np.array(self.valid) == 0)[0]
            tab = np.ones((noconsensus.shape[0], 2))
            tab[:, 1] = noconsensus
            tab[:, 0] = -1
            if not self.path_save.exists():
                self.path_save.mkdir(parents=True, exist_ok=True)
            np.savetxt(self.path_save / "too_hard.txt", tab, fmt="%1i")
        return np.vectorize(self.inv_labels.get)(np.array(ans))

    def get_probas(self):
        """Not available for this strategy, default to `get_answers()`"""
        warnings.warn(
            """
            PlantNet agreement only returns hard labels.
            Defaulting to `get_answers()`.
            """,
        )
        return self.get_answers()


class PlantNetMongo(SparseMongoBatchAlgorithm):
    """MongoDB-backed PlantNet weighted agreement."""

    def __init__(
        self,
        alpha: float = 1,
        beta: float = 1,
        theta_conf: float = THETACONF,
        theta_acc: float = THETAACC,
        authors: Optional[dict[Hashable, Hashable]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.theta_conf = theta_conf
        self.theta_acc = theta_acc
        self.authors = authors or {}

    def _get_weights(self, n_j: np.ndarray) -> np.ndarray:
        return n_j**self.alpha - n_j**self.beta + np.log(2.1)

    def _fetch_votes(
        self,
    ) -> tuple[
        list[Hashable],
        dict[Hashable, int],
        dict[int, Hashable],
        list[tuple[int, int, int]],
    ]:
        task_to_idx = self._batch_task_to_idx.copy()
        worker_to_idx = self._batch_worker_to_idx.copy()

        class_names = list(self._batch_class_to_idx.keys())
        int_labels = []
        all_int = True
        for cls in class_names:
            try:
                int_labels.append(int(self._unescape_id(cls)))
            except (TypeError, ValueError):
                all_int = False
                break

        if all_int and int_labels:
            max_label = max(int_labels)
            ordered_classes = [
                self._escape_id(i) for i in range(max_label + 1)
            ]
        else:
            ordered_classes = sorted(
                class_names,
                key=lambda cls: str(self._unescape_id(cls)),
            )

        class_to_idx = {
            class_name: idx for idx, class_name in enumerate(ordered_classes)
        }
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        n_task = len(task_to_idx)
        votes_by_task: list[list[tuple[int, int]]] = [
            [] for _ in range(n_task)
        ]

        for doc in self.db.user_votes.find(
            {"_id": {"$in": list(task_to_idx.keys())}},
            {"_id": 1, "votes": 1},
        ):
            task_id = doc["_id"]
            task_idx = task_to_idx.get(task_id)
            if task_idx is None:
                continue
            for worker_id, label in doc.get("votes", {}).items():
                worker_idx = worker_to_idx.get(worker_id)
                class_idx = class_to_idx.get(label)
                if worker_idx is None or class_idx is None:
                    continue
                votes_by_task[task_idx].append((worker_idx, class_idx))

        task_ids = [self._batch_idx_to_task[i] for i in range(n_task)]

        all_votes: list[tuple[int, int, int]] = [
            (task_idx, worker_idx, class_idx)
            for task_idx, votes in enumerate(votes_by_task)
            for worker_idx, class_idx in votes
        ]

        return (
            task_ids,
            worker_to_idx,
            idx_to_class,
            all_votes,
        )

    def _get_wmv(
        self,
        all_votes: list[tuple[int, int, int]],
        n_task: int,
        n_classes: int,
        weights: np.ndarray,
    ) -> np.ndarray:
        weighted_votes = np.zeros((n_task, n_classes), dtype=np.float64)

        for task_idx, worker_idx, class_idx in all_votes:
            weighted_votes[task_idx, class_idx] += weights[worker_idx]

        return np.argmax(weighted_votes, axis=1)

    def _get_conf_acc(
        self,
        all_votes: list[tuple[int, int, int]],
        n_task: int,
        yhat: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sum_weights = np.zeros(n_task, dtype=np.float64)
        conf = np.zeros(n_task, dtype=np.float64)

        for task_idx, worker_idx, class_idx in all_votes:
            w = weights[worker_idx]
            sum_weights[task_idx] += w
            if class_idx == yhat[task_idx]:
                conf[task_idx] += w

        acc = np.divide(
            conf,
            sum_weights,
            out=np.zeros_like(conf),
            where=sum_weights > 0,
        )
        return acc, conf

    def _get_valid_tasks(
        self,
        acc: np.ndarray,
        conf: np.ndarray,
    ) -> np.ndarray:
        valid = np.zeros_like(acc, dtype=np.int8)
        valid[(conf > self.theta_conf) & (acc > self.theta_acc)] = 1
        return valid

    def _get_n(
        self,
        task_ids: list[Hashable],
        all_votes: list[tuple[int, int, int]],
        idx_to_worker: dict[int, Hashable],
        n_workers: int,
        valid: np.ndarray,
        yhat: np.ndarray,
    ) -> np.ndarray:
        task_to_votes = defaultdict(list)
        for task_idx, worker_idx, class_idx in all_votes:
            task_to_votes[task_idx].append((worker_idx, class_idx))

        taxa_obs = np.zeros(n_workers, dtype=np.float64)
        taxa_votes = np.zeros(n_workers, dtype=np.float64)
        dico_labs_workers: list[dict[int, int]] = [
            {} for _ in range(n_workers)
        ]

        for task_idx, task_id in enumerate(task_ids):
            task_unescaped = self._unescape_id(task_id)
            author = self.authors.get(
                task_unescaped,
                self.authors.get(task_id),
            )
            for worker_idx, lab_worker in task_to_votes.get(task_idx, []):
                if lab_worker == yhat[task_idx]:
                    worker_name = idx_to_worker[worker_idx]
                    if author == worker_idx or author == worker_name:
                        if valid[task_idx] == 1:
                            if (
                                dico_labs_workers[worker_idx].get(lab_worker)
                                is None
                            ):
                                taxa_obs[worker_idx] += 1
                                dico_labs_workers[worker_idx][lab_worker] = 1

        for task_idx in range(len(task_ids)):
            for worker_idx, lab_worker in task_to_votes.get(task_idx, []):
                if lab_worker == yhat[task_idx]:
                    if dico_labs_workers[worker_idx].get(lab_worker) is None:
                        taxa_votes[worker_idx] += 1 / 10
                        dico_labs_workers[worker_idx][lab_worker] = 1

        return np.array(
            [taxa_obs[w] + np.round(taxa_votes[w]) for w in range(n_workers)],
            dtype=np.float64,
        )

    def process_batch(
        self,
        batch,
        maxiter: int = 100,
        epsilon: float = 1e-5,
    ) -> list[float]:
        self._batch_size = len(batch)
        self.t += 1
        batch = {
            self._escape_id(task_id): {
                self._escape_id(worker_id): self._escape_id(label)
                for worker_id, label in votes.items()
            }
            for task_id, votes in batch.items()
        }

        self.insert_batch(batch)
        self._prepare_mapping(batch)

        self.get_or_create_indices(
            self.task_mapping,
            list(self._batch_task_to_idx),
        )
        self.get_or_create_indices(
            self.worker_mapping,
            list(self._batch_worker_to_idx),
        )
        self.get_or_create_indices(
            self.class_mapping,
            list(self._batch_class_to_idx),
        )

        (
            task_ids,
            worker_to_idx,
            idx_to_class,
            all_votes,
        ) = self._fetch_votes()

        n_task = len(task_ids)
        n_workers = len(worker_to_idx)
        n_classes = len(idx_to_class)
        idx_to_worker = {v: k for k, v in worker_to_idx.items()}

        if n_task == 0 or n_workers == 0 or n_classes == 0:
            self._drop_batch_mappings()
            return []

        valid = np.ones(n_task, dtype=np.int8)
        weights = np.log(2.1) * np.ones(n_workers, dtype=np.float64)

        init_yhat = self._get_wmv(all_votes, n_task, n_classes, weights)
        acc, conf = self._get_conf_acc(all_votes, n_task, init_yhat, weights)
        valid = self._get_valid_tasks(acc, conf)
        n_j = self._get_n(
            task_ids,
            all_votes,
            idx_to_worker,
            n_workers,
            valid,
            init_yhat,
        )

        ll: list[float] = []
        yhat = init_yhat.copy()
        for step in tqdm(range(maxiter), desc=self.__class__.__name__):
            prev_n_j = n_j.copy()
            weights = self._get_weights(prev_n_j)
            yhat = self._get_wmv(all_votes, n_task, n_classes, weights)
            acc, conf = self._get_conf_acc(
                all_votes,
                n_task,
                init_yhat,
                weights,
            )
            valid = self._get_valid_tasks(acc, conf)
            n_j = self._get_n(
                task_ids,
                all_votes,
                idx_to_worker,
                n_workers,
                valid,
                init_yhat,
            )

            delta = np.sum(np.abs(n_j - prev_n_j)) / max(n_task, 1)
            ll.append(float(delta))

            if delta <= epsilon and step > 5:
                break

        labels = yhat if maxiter > 1 else init_yhat

        updates = []
        for task_idx, task_name in enumerate(task_ids):
            label_class = idx_to_class[int(labels[task_idx])]
            updates.append(
                UpdateOne(
                    {"_id": task_name},
                    {
                        "$set": {
                            "probs": {label_class: 1.0},
                            "current_answer": label_class,
                            "valid": int(valid[task_idx]),
                            "conf": float(conf[task_idx]),
                            "acc": float(acc[task_idx]),
                        },
                    },
                    upsert=True,
                ),
            )

        if updates:
            self.db.task_class_probs.bulk_write(updates, ordered=False)

        worker_updates = []
        for worker_idx, worker_name in idx_to_worker.items():
            worker_updates.append(
                UpdateOne(
                    {"_id": worker_name},
                    {
                        "$set": {
                            "n_j": float(n_j[worker_idx]),
                            "weight": float(weights[worker_idx]),
                        },
                    },
                    upsert=True,
                ),
            )
        if worker_updates:
            self.db.worker_confusion_matrices.bulk_write(
                worker_updates,
                ordered=False,
            )

        self._drop_batch_mappings()
        return ll

    def get_answers(self) -> np.ndarray:
        if self.n_task == 0:
            raise NotInitialized(self.__class__.__name__)

        idx_to_task = {
            doc["index"]: doc["_id"]
            for doc in self.task_mapping.find({}, {"_id": 1, "index": 1})
        }
        answers = np.empty(self.n_task, dtype=object)
        for idx in range(self.n_task):
            task_name = idx_to_task[idx]
            doc = self.db.task_class_probs.find_one(
                {"_id": task_name},
                {"current_answer": 1},
            )
            answers[idx] = (
                self._unescape_id(doc["current_answer"])
                if doc and doc.get("current_answer") is not None
                else None
            )
        return answers

    def get_probas(self) -> np.ndarray:
        warnings.warn(
            """
            PlantNet agreement only returns hard labels.
            Defaulting to `get_answers()`.
            """,
        )
        return self.get_answers()

    @property
    def pi(self) -> np.ndarray:
        return np.zeros((self.n_workers, self.n_classes, self.n_classes))

    def _online_update_pi(self, batch_pi):
        raise NotImplementedError(
            "PlantNetMongo does not use EM confusion-matrix updates.",
        )

    def _e_step(self, batch_matrix, batch_pi, batch_rho):
        raise NotImplementedError("PlantNetMongo does not use EM E-step.")

    def _m_step(self, batch_matrix, batch_T):
        raise NotImplementedError("PlantNetMongo does not use EM M-step.")
