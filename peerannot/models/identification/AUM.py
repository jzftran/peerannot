from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
sample_identifier = Union[int, str]


class AUM:
    """
    =============================
    AUM (Pleiss et. al, 2020)
    =============================

    Measures the AUM per task given the ground truth label.

    Using:

    - Margin estimation

    - Trust score per task
    """

    def __init__(
        self,
        tasks,
        n_classes,
        model,
        criterion,
        optimizer,
        n_epoch,
        topk=False,
        verbose=False,
        use_pleiss=False,
        **kwargs,
    ):
        """Compute the AUM score for each task. Given a classifier :math:`\\mathcal{C}` and tasks :math:`x_i` with hard labels from an aggregation :math:`y_i^\\text{agg}`, the AUM writes

        .. math::

            \\mathrm{AUM}(x_i)=\\frac{1}{T}\\sum_{t=1}^T \\left(\\sigma(\\mathcal{C}(x_i)){y_i^{\\text{agg}}} - \\sigma(\\mathcal{C}(x_i))_{[2]}\\right)

        :param tasks: Dataset of tasks as :math:`(x_i, _, y_i^{\\text{agg}}, i)_{(i,j)}`
        :type tasks: torch Dataset
        :param n_classes: Number of possible classes, defaults to 2
        :type n_classes: int
        :param model: Neural network to use
        :type model: torch Module
        :param criterion: loss to minimize for the network
        :type criterion: torch loss
        :param optimizer: Optimization strategy for the minimization
        :type optimizer: torch optimizer
        :param n_epoch: Number of epochs (should be the first learning rate scheduler step drop or lower than half the training epochs)
        :type n_epoch: int
        :param verbose: Print details in log, defaults to False
        :type verbose: bool, optional
        :param use_pleiss: Use Pleiss margin instead of Yang, defaults to False
        :type use_pleiss: bool, optional
        """

        self.n_classes = n_classes
        self.model = model
        self.DEVICE = kwargs.get("DEVICE", DEVICE)
        self.optimizer = optimizer
        self.tasks = tasks
        self.criterion = criterion
        self.verbose = verbose
        self.use_pleiss = use_pleiss
        self.n_epoch = n_epoch
        self.initial_lr = self.optimizer.param_groups[0]["lr"]
        self.checkpoint = {
            "epoch": n_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if topk is False:
            self.topk = 1
        else:
            self.topk = int(topk)

        self.filenames = np.array(
            [Path(samp[0]).name for samp in self.tasks.dataset.dataset.samples],
        )
        self.path = Path("./temp/").mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, "./temp/checkpoint_aum.pth")

    def make_step(self, batch):
        """One optimization step

        :param batch: Batch of tasks

            Batch:
                - index 0: tasks :math:`(x_i)_i`
                - index 1: placeholder
                - index 2: labels
                - index 3: tasks index :math:`(i)_i`

        :type batch: batch
        :return: Tuple with length, logits, targets, ground turths and index
        :rtype: tuple
        """
        xi = batch[0]
        labels = batch[2]
        idx = batch[3].tolist()
        self.optimizer.zero_grad()
        xi, labels = xi.to(self.DEVICE), labels.to(self.DEVICE)
        out = self.model(xi)
        loss = self.criterion(out, labels)
        loss.backward()
        self.optimizer.step()
        return out, labels, idx

    def get_aum(self):
        """Records prediction scores of interest for the AUM during n_epoch training epochs"""
        AUM_recorder = {
            "index": [],
            "task": [],
            "label": [],
            "epoch": [],
            "label_logit": [],
            "label_prob": [],
            "other_max_logit": [],
            "other_max_prob": [],
            "secondlogit": [],
            "secondprob": [],
        }
        for i in range(self.n_classes):
            AUM_recorder[f"logits_{i}"] = []
        self.model.to(self.DEVICE)
        self.model.train()
        for epoch in (
            tqdm(range(self.n_epoch), desc="Epoch", leave=False)
            if self.verbose
            else range(self.n_epoch)
        ):
            for batch in self.tasks:
                out, labels, idx = self.make_step(batch)
                len_ = len(idx)
                AUM_recorder["task"].extend(self.filenames[idx])
                AUM_recorder["index"].extend(idx)
                AUM_recorder["label"].extend(labels.tolist())
                AUM_recorder["epoch"].extend([epoch] * len_)
                # AUM_recorder["all_logits"].extend(out.tolist())
                # s_y and P_y
                AUM_recorder["label_logit"].extend(
                    out.gather(1, labels.view(-1, 1)).squeeze().tolist(),
                )
                probs = out.softmax(dim=1)
                AUM_recorder["label_prob"].extend(
                    probs.gather(1, labels.view(-1, 1)).squeeze().tolist(),
                )
                # (s\y)[1] and (P\y)[1]
                masked_logits = torch.scatter(out, 1, labels.view(-1, 1), float("-inf"))
                masked_probs = torch.scatter(
                    probs, 1, labels.view(-1, 1), float("-inf"),
                )
                (
                    other_logit_values,
                    other_logit_index,
                ) = masked_logits.max(1)
                (
                    other_prob_values,
                    other_prob_index,
                ) = masked_probs.max(1)
                if len(other_logit_values) > 1:
                    other_logit_values = other_logit_values.squeeze()
                    other_prob_values = other_prob_values.squeeze()
                AUM_recorder["other_max_logit"].extend(other_logit_values.tolist())
                AUM_recorder["other_max_prob"].extend(other_prob_values.tolist())

                # s[2] ans P[2]
                second_logit = torch.sort(out, axis=1)[0][:, -(self.topk + 1)]
                second_prob = torch.sort(probs, axis=1)[0][:, -(self.topk + 1)]
                AUM_recorder["secondlogit"].extend(second_logit.tolist())
                AUM_recorder["secondprob"].extend(second_prob.tolist())
                for cl in range(self.n_classes):
                    AUM_recorder[f"logits_{cl}"].extend(out[:, cl].tolist())
        self.AUM_recorder = pd.DataFrame(AUM_recorder)

    def compute_aum(self):
        """Compute the AUM for each task"""
        data = self.AUM_recorder
        tasks = {
            "sample_id": [],
            "filename": [],
            "AUM_yang": [],
            "AUM_pleiss": [],
        }
        burn = 0

        for index in data["index"].unique():
            tmp = data[data["index"] == index]
            y = tmp.label.iloc[0]
            target_values = tmp.label_logit.values[burn:]
            logits = tmp.values[burn:, -self.n_classes :]
            llogits = np.copy(logits)
            _ = np.put_along_axis(
                logits, logits.argmax(1).reshape(-1, 1), float("-inf"), 1,
            )
            masked_logits = logits
            other_logit_values, other_logit_index = masked_logits.max(
                1,
            ), masked_logits.argmax(1)
            other_logit_values = other_logit_values.squeeze()
            other_logit_index = other_logit_index.squeeze()
            margin_values_yang = (target_values - other_logit_values).tolist()
            _ = np.put_along_axis(
                llogits,
                np.repeat(y, len(tmp)).reshape(-1, 1),
                float("-inf"),
                1,
            )
            masked_logits = llogits
            other_logit_values, other_logit_index = masked_logits.max(
                1,
            ), masked_logits.argmax(1)
            other_logit_values = other_logit_values.squeeze()
            other_logit_index = other_logit_index.squeeze()
            margin_values_pleiss = (target_values - other_logit_values).mean()
            tasks["sample_id"].append(index)
            tasks["filename"].append(self.filenames[index])
            tasks["AUM_yang"].append(np.mean(margin_values_yang))
            tasks["AUM_pleiss"].append(np.mean(margin_values_pleiss))
        self.aums = pd.DataFrame(tasks)

    def cut_lowests(self, alpha=0.01):
        """Computes the tasks with the lowest AUM scores.
        The index of such tasks are stored in the `.too_hard` attribute.

        :param alpha: quantile order to identify as ambiguous, defaults to 0.01
        :type alpha: float, optional
        """
        quantile = np.nanquantile(list(self.aums["AUM_pleiss"].to_numpy()), alpha)
        if self.use_pleiss:
            too_hard = self.aums[self.aums["AUM_pleiss"] <= quantile]
        else:
            too_hard = self.aums[self.aums["AUM_yang"] <= quantile]
        self.index_too_hard = too_hard["sample_id"].to_numpy()
        self.tasks_too_hard = [
            int(filename.split("-")[-1].split(".")[0])
            for filename in too_hard["filename"].to_numpy()
        ]
        self.quantile = quantile
        self.too_hard = np.column_stack(
            (self.index_too_hard, self.tasks_too_hard),
        ).astype(int)

    def run(self, alpha=0.01):
        """Run AUM identification"""
        self.get_aum()
        self.compute_aum()
        self.cut_lowests(alpha)
