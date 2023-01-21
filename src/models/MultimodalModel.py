from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from src.models.LitModel import LitModel, torch


def estimate_prior_class_probs(
    ds: torch.utils.data.Dataset, n_classes: int
) -> torch.Tensor:
    """
    Estimates the log-prior probabilities of each class.
    """
    counts = torch.zeros((n_classes,))
    for _, c in ds:
        counts[c] += 1.0
    log_prior_probs = torch.log(counts / len(ds))

    return log_prior_probs


class NaiveBayesFusion(LitModel):
    """
    Before using this model, `.log_prior_probs` must be set manually!
    E.g. before training, run `.estimate_prior_class_probs`. If the prior distributions of classes are different between two
    datasets/splits, it must be re-estimated.
    """

    def __init__(
        self,
        *classifiers: List[LitModel],
        learning_rate=0.001,
        adam_epsilon=1e-8,
        weight_decay: List[float] = [],
        **kwargs
    ):
        """
        kwargs can be used to store additional hyperparameters that needs to be logged.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._classifiers: torch.nn.ModuleList[LitModel] = torch.nn.ModuleList(
            classifiers
        )
        self._N = len(classifiers)  # Should always be the number of classifiers.
        self.log_prior_probs = (
            None  # The pi-vector. Is not a parameter and should be set manually
        )
        self.mask = None  # If P(c)=0, then we define the prediction to zero. Otherwise it results in NaNs
        self.freeze_net = False

    def forward_no_top(self, x):
        raise NotImplemented("Not used in naive Bayes fusion")

    def forward_no_softmax(self, x):
        z_stack = torch.stack(
            [clf.forward_no_softmax(x_i) for x_i, clf in zip(x, self._classifiers)]
        )
        return torch.sum(z_stack, dim=0) - (self._N - 1) * self.log_prior_probs

    def forward(self, x):
        z = self.forward_no_softmax(x)
        z = z.mul(self.mask)
        return self.softmax(z)

    @property
    def top(self):
        raise NotImplemented(
            "Not defined for naive Bayes fusion"
        )  # Should probably inherit from elsewhere as it violates the Liskov substitution principle.

    def estimate_prior_class_probs(
        self,
        ds: torch.utils.data.Dataset,
        n_classes: Optional[int] = None,
        precomputed_log_probs: Optional[torch.Tensor] = None,
    ):
        """
        Either takes a subclass of torch.utils.data.Dataset containing a `.classes` property,
        or needs a supplied n_classes argument.

        Can take a precomputed log-probs vector since this is a very expensive vector to compute. This vector has to be the same shape as `estimate_prior_class_probs` returns.
        """
        if n_classes is not None:
            self.n_classes = n_classes
        elif "classes" in dir(ds):
            self.n_classes = len(ds.classes)
        else:
            raise ValueError(
                "n_classes must be supplied if dataset does not contain a `.classes` property"
            )
        if precomputed_log_probs is not None:
            self.log_prior_probs = precomputed_log_probs
        else:
            self.log_prior_probs = estimate_prior_class_probs(ds, self.n_classes)
        self.mask = ~torch.isinf(self.log_prior_probs).view(
            1, -1
        )  # We keep everything finite.
        self.log_prior_probs[~self.mask.squeeze(0)] = 0.0
        return self

    def cuda(self):
        super().cuda()  # Registers the submodules as well.
        self.log_prior_probs = self.log_prior_probs.cuda()  # Is not a parameter.
        self.mask = self.mask.cuda()
        return self

    @property
    def classifiers(self):
        return self._classifiers

    @classifiers.setter
    def classifiers(self, *classifiers: List[LitModel]):
        self._classifiers = classifiers
        self._N = len(classifiers)

    def _configure_optim_train(self):
        return [
            torch.optim.AdamW(
                clf.parameters() if not self.freeze_net else clf.top.parameters(),
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
                weight_decay=l2_rate,
            )
            for clf, l2_rate in zip(self.classifiers, self.hparams.weight_decay)
        ]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # Wrapping the training step in case we give multiple optimizers.
        return super().training_step(batch, batch_idx)  # Could be an issue?

    @property
    def learning_rate(self):
        return self.hparams.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.hparams.learning_rate = learning_rate

    @property
    def weight_decay(self):
        return self.hparams.weight_decay

    @weight_decay.setter
    def weight_decay(self, weight_decay):
        self.hparams.weight_decay = weight_decay


class LogisticRegressionBasedFusion(LitModel):
    def __init__(
        self,
        *classifiers: List[LitModel],
        out_features,
        learning_rate=0.001,
        adam_epsilon=1e-8,
        weight_decay: float = 1e-6,
        **kwargs
    ):
        """
        kwargs can be used to store additional hyperparameters that needs to be logged.
        If given a list of weight_decays, the list must be 1 element longer than the number of classifiers!
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._classifiers: torch.nn.ModuleList[LitModel] = torch.nn.ModuleList(
            classifiers
        )
        self._top = torch.nn.LazyLinear(out_features)

    def forward_no_top(self, x):
        with torch.no_grad():
            z_stack = torch.cat(
                [clf.forward_no_top(x_i) for x_i, clf in zip(x, self._classifiers)],
                dim=1,
            )
        return z_stack

    def forward_no_softmax(self, x):
        z_stack = self.forward_no_top(x)
        return self._top(z_stack)

    @property
    def top(self):
        return self._top

    @property
    def classifiers(self):
        return self._classifiers

    @classifiers.setter
    def classifiers(self, *classifiers: List[LitModel]):
        self._classifiers = classifiers

    def _configure_optim_train(
        self,
    ):  # If we would like to freeze parts of the network, we should probably just return optimizers for only the non-frozen parameters here.
        if isinstance(self.hparams.weight_decay, float):
            return [
                torch.optim.AdamW(
                    self._top.parameters(),
                    lr=self.hparams.learning_rate,
                    eps=self.hparams.adam_epsilon,
                    weight_decay=self.hparams.weight_decay,
                )
            ]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # Wrapping the training step in case we give multiple optimizers.
        return super().training_step(batch, batch_idx)

    @property
    def learning_rate(self):
        return self.hparams.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.hparams.learning_rate = learning_rate

    @property
    def weight_decay(self):
        return self.hparams.weight_decay

    @weight_decay.setter
    def weight_decay(self, weight_decay):
        self.hparams.weight_decay = weight_decay


class SVMBasedFusion(LitModel):
    def __init__(
        self,
        *classifiers: List[LitModel],
        regularization_strength=0.003,
        max_iter=1000,
        n_jobs=1
    ):
        super().__init__()
        self._classifiers: List[LitModel] = torch.nn.ModuleList(classifiers)
        self.max_iter = max_iter
        self.svm = SGDClassifier(
            loss="hinge",  # Makes it an SVM
            penalty="l2",
            max_iter=self.max_iter,
            early_stopping=False,
            validation_fraction=0.1,
            n_jobs=n_jobs,
            learning_rate="optimal",
            verbose=False,
            alpha=regularization_strength,
        )
        self.calib_svm = CalibratedClassifierCV(  # SVM needs to be calibrated
            base_estimator=self.svm, n_jobs=1, cv="prefit"
        )
        self._cuda = False

    def fit_svm(
        self, ds: torch.utils.data.DataLoader, epochs=1, n_classes=205, verbose=True
    ):
        classes = np.arange(0, n_classes)
        i = 0
        for epoch in tqdm(range(epochs), desc="Epochs", disable=not verbose):
            for x, y in tqdm(ds, desc="Dataset", disable=not verbose):
                z_stack = self.forward_no_top(x)
                self.svm.partial_fit(z_stack, y, classes=classes)
        return self

    def top(self):
        raise NotImplemented("Not used in SVM-based fusion")

    def cuda(self):
        super().cuda()
        self._cuda = True

    def calibrate_svm(self, ds: torch.utils.data.DataLoader, verbose=True):
        z_stack = []
        labels = []
        for x, y in tqdm(ds, desc="Calibration", disable=not verbose):
            z_stack.append(self.forward_no_top(x))
            labels.append(y)

        z_stack = np.vstack(z_stack)
        labels = np.hstack(labels)
        self.calib_svm.fit(z_stack, labels)
        return self

    def forward_no_top(self, x):
        with torch.no_grad():
            z_stack = (
                torch.hstack(
                    [clf.forward_no_top(x_i) for x_i, clf in zip(x, self._classifiers)]
                )
                .detach()
                .cpu()
                .numpy()
            )
        return z_stack

    def forward_no_softmax(self, x):
        raise NotImplemented("Not used in SVM-based fusion")

    def forward(self, x):
        z_stack = self.forward_no_top(x)
        probs = torch.tensor(self.calib_svm.predict_proba(z_stack))
        if self._cuda:
            probs = probs.cuda()
        return probs

    def _configure_optim_train(self):
        raise NotImplemented("Not used in SVM-based fusion")


#############
### Tests ###
#############
if __name__ == "__main__":
    import BERT
    import dataset
    import VGG16

    # Instantiate models
    bert_params = {
        "learning_rate": 0.0006870443398072322,
        "adam_epsilon": 1e-08,
        "weight_decay": 0.0,
        "dropout_rate": 0.31547838646677684,
        "top_dense_layer_units": [],
    }  # Found using Optuna (see function BERT.tune_hparams())
    bert_clf: BERT.LitBERTModel = BERT.LitBERTModel(**bert_params)
    bert_clf.load_state_dict(torch.load("BERT_model_aaa"))
    vgg16_clf: VGG16.LitVGG16Model = VGG16.LitVGG16Model()
    fusion_model: NaiveBayesFusion = NaiveBayesFusion(vgg16_clf, bert_clf)

    # Load bimodal validation set
    val_text_ds: dataset.TextModalityDS = dataset.load_text_ds("val_text.json")
    val_img_ds: dataset.ImageModalityDS = dataset.load_img_ds("val")
    bimodal_val_ds: dataset.BimodalDS = dataset.BimodalDS(
        image_ds=val_img_ds, text_ds=val_text_ds
    )

    # Estimate prior class probabilities
    fusion_model.estimate_prior_class_probs(bimodal_val_ds)
    print("Estimated class log-probs:")
    print(fusion_model.log_prior_probs)

    import pytorch_lightning as pl

    # Run through validation set
    fusion_model.cuda()
    trainer = pl.Trainer(
        gpus=min(1, torch.cuda.device_count()),
        max_epochs=5,
        checkpoint_callback=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    print(
        trainer.test(
            fusion_model, dataloaders=dataset.bimodal_dataloader(bimodal_val_ds)
        )
    )
