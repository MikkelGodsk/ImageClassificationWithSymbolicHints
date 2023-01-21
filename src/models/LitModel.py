import sys
from abc import ABC, abstractmethod
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision

from src.third_party_files.reliability_diagrams import (
    reliability_diagram,
)  # From GitHub. Download https://github.com/hollance/reliability-diagrams and save as reliability.py

gpu = torch.device("cuda")


class LitModel(pl.LightningModule, ABC):
    """
    Base class for my lightning modules.

    Note: pl.LightningModule inherits from torch.nn.Module

    Modified from: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    `self.save_hyperparameters()` should be called after calling `super().__init__()` in child classes.

    The models must individually put the data on the gpu at some point, e.g. images might go directly to `.cuda()`, while text might have to go through preprocessing and then on `.cuda()`.
    """

    def __init__(self, n_bins=25, **kwargs):
        super().__init__()  # **kwargs)

        self.net = None
        self.softmax = torch.nn.Softmax(dim=-1)

        self.train_top_1_acc = torchmetrics.Accuracy(top_k=1)
        self.train_top_5_acc = torchmetrics.Accuracy(top_k=5)
        self.val_top_1_acc = torchmetrics.Accuracy(top_k=1)
        self.val_top_5_acc = torchmetrics.Accuracy(top_k=5)
        self.test_top_1_acc = torchmetrics.Accuracy(top_k=1)
        self.test_top_5_acc = torchmetrics.Accuracy(top_k=5)
        # self.test_conf_matrix = torchmetrics.ConfusionMatrix(num_classes=205)
        self.expected_calib_error = torchmetrics.CalibrationError(
            n_bins=n_bins, norm="l1", compute_on_cpu=True
        )
        self.maximum_calib_error = torchmetrics.CalibrationError(
            n_bins=n_bins, norm="max", compute_on_cpu=True
        )

    @abstractmethod
    def forward_no_softmax(self, x):
        raise NotImplemented

    @abstractmethod
    def forward_no_top(self, x):
        raise NotImplemented

    @property
    @abstractmethod
    def top(self):
        raise NotImplemented

    def forward(self, x):
        z = self.forward_no_softmax(x)
        return self.softmax(z)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = F.cross_entropy(z, y)
        self.train_top_1_acc(z, y)
        self.train_top_5_acc(z, y)
        self.log("hp_train_loss", loss.item(), on_step=True, on_epoch=False)
        self.log(
            "hp_train_top_1_acc", self.train_top_1_acc, on_epoch=True, on_step=False
        )
        self.log(
            "hp_train_top_5_acc", self.train_top_5_acc, on_epoch=True, on_step=False
        )
        return loss

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp_metric": 0})

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = F.cross_entropy(z, y)
        self.val_top_1_acc(z, y)
        self.val_top_5_acc(z, y)
        self.expected_calib_error(z, y)
        self.maximum_calib_error(z, y)
        self.log("hp_validation_loss", loss.item(), on_step=True, on_epoch=False)
        self.log(
            "hp_validation_top_1_acc", self.val_top_1_acc, on_step=False, on_epoch=True
        )
        self.log(
            "hp_validation_top_5_acc", self.val_top_5_acc, on_step=False, on_epoch=True
        )
        self.log("hp_val_ece", self.expected_calib_error, on_step=False, on_epoch=True)
        self.log("hp_val_mce", self.maximum_calib_error, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = F.cross_entropy(z, y)
        self.test_top_1_acc(z, y)
        self.test_top_5_acc(z, y)
        self.expected_calib_error(z, y)
        self.maximum_calib_error(z, y)
        self.log("hp_test_loss", loss.item(), on_step=True, on_epoch=False)
        self.log("hp_test_top_1_acc", self.test_top_1_acc, on_step=False, on_epoch=True)
        self.log("hp_test_top_5_acc", self.test_top_5_acc, on_step=False, on_epoch=True)
        self.log("hp_metric", self.test_top_1_acc, on_step=False, on_epoch=True)
        self.log("hp_test_ece", self.expected_calib_error, on_step=False, on_epoch=True)
        self.log("hp_test_mce", self.maximum_calib_error, on_step=False, on_epoch=True)

    @abstractmethod
    def _configure_optim_train(self):
        pass

    def configure_optimizers(self):
        return self._configure_optim_train()


class CalibratedLitModel(LitModel):
    """
    A wrapper to a LitModel classifier which enables calibration using temperature scaling.

    Sets the classifier in eval-mode by default.
    """

    def __init__(
        self,
        clf: LitModel,
        init_temperature=1.0,
        n_bins=15,
        learning_rate=0.01,
        max_iter=50,
    ):
        super().__init__(n_bins=n_bins)
        self.clf = clf
        self.temperature = torch.nn.Parameter(
            torch.ones(1) * init_temperature, requires_grad=True
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.clf.eval()

        # For BFGS:
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def forward_no_softmax(self, x):
        z = self.clf.forward_no_softmax(x)
        return z / self.temperature

    def forward_no_top(self, x):
        return self.clf.forward_no_top(x)

    def top(self):
        return self.clf.top

    def forward(self, x):
        z_scaled = self.forward_no_softmax(x)
        return self.softmax(z_scaled)

    def pred_labels_and_logits(self, ds: torch.utils.data.DataLoader):
        """
        Useful for adjusting temperature.
        """
        labels = []
        logits = []
        with torch.no_grad():
            self.clf.eval()
            for x, y in ds:
                z = self.clf.forward_no_softmax(x)
                logits.append(z.cpu())
                labels.append(y.cpu())

            logits = torch.vstack(logits).to(self.temperature.device)
            labels = torch.hstack(labels).to(self.temperature.device)
        return labels, logits

    def calibrate(self, ds: torch.utils.data.DataLoader):
        """
        One of two approaches. This one should be the better one, as we are dealing with a convex optimization problem.

        Calibrates the model on the dataset outside of PyTorch Lightning.
        As https://gist.github.com/huberl/56ac1a18a95a7a87730e36df781afb5f
        And https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
        """
        optimizer = self.configure_optimizers()
        criterion = torch.nn.CrossEntropyLoss()
        labels, logits = self.pred_labels_and_logits(ds)

        def eval():
            # Needs to be called multiple times to compute new gradients.
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        return self

    def _configure_optim_train(self):
        return torch.optim.LBFGS(
            [self.temperature], lr=self.learning_rate, max_iter=self.max_iter
        )

    def cuda(self):
        """
        PyTorch complained about the BERT model not being all on GPU.
        The temperature parameter of this class turned out to be on the cpu and would not move to GPU.
        This turned out to solve the problem.
        After `trainer.fit`, call this method.
        """
        super().cuda()
        self.temperature = self.temperature.cuda()
        self.clf = self.clf.cuda()
        return self

    def on_train_start(self):
        self.clf.eval()

    def on_train_end(self):
        self.clf.train()

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward_no_softmax(x)
        loss = F.cross_entropy(z, y)
        self.expected_calib_error(z, y)
        self.maximum_calib_error(z, y)
        self.log("hp_train_loss", loss.item(), on_step=True, on_epoch=False)
        self.log(
            "hp_train_ece", self.expected_calib_error, on_step=False, on_epoch=True
        )
        self.log("hp_train_mce", self.maximum_calib_error, on_step=False, on_epoch=True)
        self.log(
            "hp_temperature", self.temperature.item(), on_step=True, on_epoch=False
        )
        return loss

    """def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward_no_softmax(x)
        loss = F.cross_entropy(z, y)
        self.expected_calib_error(z, y)
        self.maximum_calib_error(z, y)
        self.log("hp_validation_loss", loss.item(), on_step=True, on_epoch=False)
        self.log("hp_val_ece", self.expected_calib_error, on_step=False, on_epoch=True)
        self.log("hp_val_mce", self.maximum_calib_error, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward_no_softmax(x)
        loss = F.cross_entropy(z, y)
        self.expected_calib_error(z, y)
        self.maximum_calib_error(z, y)
        self.log("hp_test_loss", loss.item(), on_step=True, on_epoch=False)
        self.log("hp_test_ece", self.expected_calib_error, on_step=False, on_epoch=True)
        self.log("hp_test_mce", self.maximum_calib_error, on_step=False, on_epoch=True)"""


def evaluate_model(clf: pl.LightningModule, dataloader: torch.utils.data.DataLoader):
    """
    Example:  Experiments.evaluate_model(fusion_model, bimodal_dataloader(bimodal_val_ds))
    """
    trainer = pl.Trainer(
        gpus=min(1, torch.cuda.device_count()),
        max_epochs=5,
        checkpoint_callback=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    v = trainer.test(clf, dataloaders=dataloader)
    print(v)
    return v


def reliability_values(
    clf: Union[LitModel, CalibratedLitModel], ds: torch.utils.data.DataLoader
):
    clf.eval()

    y_probs = []
    y_trues = []
    with torch.no_grad():  # No computation of gradients. Fixes the memory issue.
        for x, y in ds:
            probs = clf(x)
            y_probs.append(probs.cpu())
            y_trues.append(y.cpu())

    y_probs = torch.vstack(y_probs)
    y_trues = torch.hstack(y_trues)
    y_preds = torch.argmax(y_probs, axis=1)
    y_confs = y_probs[
        torch.arange(len(y_preds)), y_preds
    ]  # Estimated probability of true class.

    return {
        "true_labels": y_trues,
        "pred_labels": y_preds,
        "confidences": y_confs,
    }, y_probs


def reliability_plot(
    clf: Union[LitModel, CalibratedLitModel],
    ds: torch.utils.data.DataLoader,
    fig_name: str,
):
    """
    Modified from  https://github.com/hollance/reliability-diagrams
    """
    import os

    import matplotlib.pyplot as plt

    d, _ = reliability_values(clf, ds)
    d = {k: v.detach().numpy() for k, v in d.items()}

    # Override matplotlib default styling.
    plt.style.use("seaborn")

    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)

    fig = reliability_diagram(**d, num_bins=10, return_fig=True)

    # Save as file
    if not os.path.isdir("reliability_plots"):
        os.mkdir("reliability_plots")
    fig.savefig(os.path.join("reliability_plots", fig_name))


def calibrate(
    clf: Union[LitModel, CalibratedLitModel],
    val_ds: torch.utils.data.DataLoader,
    model_name: str,
    n_bins: int = 15,
    save_dir="~/bscproj/CMPlaces",
    learning_rate=0.01,
    max_iter=50,
) -> CalibratedLitModel:
    """
    Example:  Experiments.calibrate(bert_clf, val_ds, "BERT")
    Choose a relatively large batch size, e.g. 250 !!

    Progress can be loaded into TensorBoard.
    """

    # First reliability plot:
    reliability_plot(clf, val_ds, "reliability_" + model_name + ".svg")

    # Calibrate model
    if isinstance(clf, CalibratedLitModel):
        calibrated_clf = clf.cuda()
    else:
        calibrated_clf = CalibratedLitModel(
            clf, n_bins=n_bins, learning_rate=learning_rate, max_iter=max_iter
        ).cuda()
    calibrated_clf.calibrate(val_ds)

    # Second reliability plot:
    reliability_plot(
        calibrated_clf, val_ds, "reliability_calibrated_" + model_name + ".svg"
    )

    print(
        "\n\nFinal temperature of calibrated {:s}: {:.16f}\n\n".format(
            model_name, calibrated_clf.temperature.detach().cpu().numpy()[0]
        )
    )

    return calibrated_clf


def manual_temperature_search_plot(
    clf: LitModel,
    ds: torch.utils.data.DataLoader,
    temperatures=None,
    model_name=None,
    n_bins=15,
    calibration_lr=0.01,
    calibration_max_iter=50,
):
    import numpy as np

    temperatures = (
        temperatures if temperatures is not None else np.arange(0.5, 2.5, 0.005)
    )

    calibrated_clf = CalibratedLitModel(
        clf, n_bins=n_bins, learning_rate=calibration_lr, max_iter=calibration_max_iter
    )
    calibrated_clf.calibrate(ds)
    optim_t = calibrated_clf.temperature.detach().cpu().numpy()
    labels, logits = calibrated_clf.pred_labels_and_logits(ds)

    nll = []
    ece = []
    mce = []
    avg_conf = []

    for t in temperatures:
        nll.append(F.cross_entropy(logits / t, labels))
        avg_conf.append(torch.mean(torch.max(F.softmax(logits / t), dim=1).values))
        ece.append(
            torchmetrics.functional.calibration_error(
                logits / t, labels, n_bins=n_bins, norm="l1"
            )
        )
        mce.append(
            torchmetrics.functional.calibration_error(
                logits / t, labels, n_bins=n_bins, norm="max"
            )
        )

    nll = torch.hstack(nll).detach().cpu().numpy()
    avg_conf = torch.hstack(avg_conf).detach().cpu().numpy()
    ece = torch.hstack(ece).detach().cpu().numpy()
    mce = torch.hstack(mce).detach().cpu().numpy()

    if model_name is not None:
        import matplotlib.pyplot as plt

        plt.plot(
            temperatures,
            nll,
            temperatures,
            avg_conf,
            temperatures,
            ece,
            temperatures,
            mce,
        )
        plt.plot(
            np.hstack((optim_t, optim_t)), np.hstack((0, np.max(nll))), "--", alpha=0.7
        )
        plt.title("Calibration metrics vs temperatures for {:s}".format(model_name))
        plt.legend(["NLL", "Avg. conf", "ECE", "MCE", "Estimated optimal T"])
        if not os.path.isdir("reliability_plots"):
            os.mkdir("reliability_plots")
        plt.savefig(
            os.path.join(
                "reliability_plots", "Temperature search - {:s}".format(model_name)
            )
        )
        plt.clf()

    return temperatures, nll, avg_conf, ece, mce
