from typing import Any, Callable, Iterator, Literal, cast
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from torch import Tensor, nn

from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Recall, Precision


class ClassificationModelSystem(LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        

        task = "multiclass"

        # Initialize metrics if not provided
        metrics = [
                Accuracy(num_classes=num_classes, task=task),
                F1Score(num_classes=num_classes, task=task),
                Recall(num_classes=num_classes, task=task),
                Precision(num_classes=num_classes, task=task),
            ]

        self.criteron = nn.CrossEntropyLoss()
        self.num_classes = num_classes

        _metrics = MetricCollection(metrics)
        self.train_metrics = _metrics.clone(prefix="train/")
        self.val_metrics = _metrics.clone(prefix="val/")
        self.test_metrics = _metrics.clone(prefix="test/")



    def training_step(self, item) -> STEP_OUTPUT:
        question, trajectory, y = item
        y_hat = self(trajectory, question)

        loss = self.criteron(y_hat, y)

        batch_size = trajectory.shape[0]

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            # batch_size=batch_size,
            on_epoch=True,
            on_step=True,
        )

        total_loss = loss 
        self.log("train/total_loss", total_loss, prog_bar=True, batch_size=batch_size)

        self.log_dict(
            self.train_metrics(y_hat, y),
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
        )

        return {"loss": total_loss, "y_hat": y_hat}

    def validation_step(self, item, index: int) -> STEP_OUTPUT:
        return self._non_train_step(item, "val")

    def test_step(self, item, index: int) -> STEP_OUTPUT:
        return self._non_train_step(item, "test")

    def _non_train_step(self, item, step: Literal["val", "test"]) -> STEP_OUTPUT:
        question,trajectory, y = item
        y_hat = self(trajectory, question)
        loss = self.criteron(y_hat, y)
        self.log_dict(
            self.val_metrics(y_hat, y) if step == "val" else self.test_metrics(y_hat, y),
            on_epoch=True,
            on_step=True,
            batch_size=trajectory.shape[0],
        )
        self.log(
            f"{step}/loss",
            loss,
            prog_bar=True,
            batch_size=trajectory.shape[0],
            on_epoch=True,
            on_step=False,
        )
        return {"loss": loss, "y_hat": y_hat}
