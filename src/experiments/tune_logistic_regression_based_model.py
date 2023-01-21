import typing

import numpy as np
import optuna
import pytorch_lightning as pl
import torch

import src.models.BERT as BERT
import src.models.VGG16 as VGG16
from src.experiments.main import evaluate_model
from src.features.dataset import *
from src.models.MultimodalModel import *


def objective(
    trial: optuna.trial.Trial,
    train_ds: torch.utils.data.DataLoader,
    val_split: torch.utils.data.DataLoader,
    make_params: typing.Callable[
        [optuna.trial.Trial],
        typing.Dict[str, typing.Union[float, int, typing.List[int]]],
    ],
    max_epochs: int,
    make_logger: typing.Callable[[], pl.loggers.base.LightningLoggerBase],
) -> float:
    """
    From example: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
    """
    params = make_params(trial)
    logger = make_logger()

    bert_params = {
        "learning_rate": 0.0,
        "adam_epsilon": 1e-08,
        "weight_decay": 0.0,
        "dropout_rate": 0.0,
        "top_dense_layer_units": [],
    }  # These are by-passed in the fusion model
    bert_clf: BERT.LitBERTModel = BERT.LitBERTModel(**bert_params)
    bert_clf.load_state_dict(torch.load("new_BERT_model"))
    vgg16_clf: VGG16.LitVGG16Model = VGG16.LitVGG16Model()
    fusion_model: LogisticRegressionBasedFusion = LogisticRegressionBasedFusion(
        vgg16_clf,
        bert_clf,
        out_features=205,
        learning_rate=params["lr"],
        adam_epsilon=1e-08,
        weight_decay=params["top_l2_rate"],
    )

    trainer = pl.Trainer(
        gpus=min(1, torch.cuda.device_count()),
        max_epochs=max_epochs,
        logger=logger,
        max_time="00:23:00:00",
        checkpoint_callback=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="hp_validation_top_1_acc"
            )
        ],
    )
    fusion_model.cuda()
    trainer.logger.log_hyperparams(params)
    trainer.fit(fusion_model, train_ds, val_split)
    return trainer.callback_metrics["hp_validation_top_1_acc"].item()


def hyperparam_tuning(
    train_ds: torch.utils.data.DataLoader,
    val_split: torch.utils.data.DataLoader,
    make_params: typing.Callable[
        [optuna.trial.Trial],
        typing.Dict[str, typing.Union[float, int, typing.List[int]]],
    ],
    n_trials: int,
    timeout: int,
    max_epochs: int,
):
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    logger_fn = lambda: pl.loggers.TensorBoardLogger(
        save_dir="~/bscproj/CMPlaces/lightning_logs",
        name="Logistic_regression_model_hparam_search",
    )
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: objective(
            trial, train_ds, val_split, make_params, max_epochs, logger_fn
        ),
        n_trials=n_trials,
        timeout=timeout,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params


def tune_hparams(
    make_params: typing.Callable[
        [optuna.trial.Trial],
        typing.Dict[str, typing.Union[float, int, typing.List[int]]],
    ],
    n_trials: int = 50,
    timeout: int = 86100,
):

    text_ds: TextModalityDS = load_text_ds("train_text.json")
    img_ds: ImageModalityDS = load_img_ds("train")  # This call will take a long time...
    # text_ds: TextModalityDS = load_text_ds('val_text.json')  # For debugging only
    # img_ds: ImageModalityDS = load_img_ds('val')

    train_val_split = 0.8
    train_text_split, val_text_split = split_text_ds(text_ds, train_val_split)
    train_img_split, val_img_split = split_img_ds(img_ds, train_val_split)
    train_split = BimodalDS(image_ds=train_img_split, text_ds=train_text_split)
    val_split = BimodalDS(image_ds=val_img_split, text_ds=val_text_split)

    return hyperparam_tuning(
        bimodal_dataloader(train_split),
        bimodal_dataloader(val_split),
        make_params,
        n_trials=n_trials,
        timeout=timeout,
        max_epochs=1,
    )


if __name__ == "__main__":

    def make_params(trial):
        return {
            "lr": trial.suggest_loguniform("BERT_lr", 5e-6, 1),
            "top_l2_rate": trial.suggest_loguniform("top_l2_rate", 1e-10, 1e-1),
        }

    tune_hparams(make_params)
