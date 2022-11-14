from BERT import *
from main import *


dataset='ImageNet'


num_classes = 1000 if dataset.lower() == "imagenet" else 205
bert_embedding_dim = 1024 if dataset.lower() == "imagenet" else 768
bert_model_name = "bert-large-cased" if dataset.lower() == "imagenet" else "bert-base-cased"
    
    
def objective(
    trial: optuna.trial.Trial, 
    train_ds: torch.utils.data.DataLoader, 
    val_split: torch.utils.data.DataLoader, 
    make_params: typing.Callable[[optuna.trial.Trial], typing.Dict[str, typing.Union[float, int, typing.List[int]]]], 
    max_epochs: int, 
    make_logger: typing.Callable[[], pl.loggers.base.LightningLoggerBase]
) -> float:
    """
        From example: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
    """
    params = make_params(trial)
    logger = make_logger()
    model = LitBERTModel(**params, num_classes=num_classes, bert_embedding_dim=bert_embedding_dim, bert_model_name=bert_model_name)
    model.freeze_net = True
    trainer = pl.Trainer(
        logger=logger,
        #checkpoint_callback=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        max_epochs=max_epochs,
        gpus=1 if torch.cuda.is_available() else None
    )
    
    trainer.logger.log_hyperparams(params)
    #trainer.fit(model, train_ds, val_split)
    #return trainer.callback_metrics["hp_validation_top_1_acc"].item()
    return evaluate_model(model, val_split, "BERT-1024-based-model")

    
def hyperparam_tuning(
    train_ds: torch.utils.data.DataLoader, 
    val_split: torch.utils.data.DataLoader, 
    make_params: typing.Callable[[optuna.trial.Trial], typing.Dict[str, typing.Union[float, int, typing.List[int]]]], 
    n_trials: int, 
    timeout: int, 
    max_epochs: int
): 
    pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner())
    logger_fn = lambda: pl.loggers.TensorBoardLogger(save_dir="~/bscproj/CMPlaces", name="BERT_hparam_"+dataset)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, train_ds, val_split, make_params, max_epochs, logger_fn), 
        n_trials=n_trials, 
        timeout=timeout,
        gc_after_trial=True
    )
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    return trial.params
        
    
def tune_hparams(make_params: typing.Callable[[optuna.trial.Trial], typing.Dict[str, typing.Union[float, int, typing.List[int]]]], n_trials: int = 50, timeout: int = 86100):
    from dataset import load_text_ds, text_dataloader, set_dataset
    set_dataset(dataset)
    train_val_split=0.8
    full_train_ds = load_text_ds('train_text.json')
    t = int(len(full_train_ds)*train_val_split)
    train_ds, val_split = torch.utils.data.random_split(full_train_ds, [t, len(full_train_ds)-t], generator=torch.Generator().manual_seed(42))
    #train_ds = text_dataloader(train_ds)
    val_split = text_dataloader(val_split)
    
    return hyperparam_tuning(train_ds, val_split, make_params, n_trials=n_trials, timeout=timeout, max_epochs=5)



def make_params_v1(trial):
    n_layers = trial.suggest_int("n_layers", 0, 2)
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 5e-4, 1e-2),
        'adam_epsilon': 1e-8,
        'weight_decay': trial.suggest_loguniform("weight_decay", 1e-10, 1e-1),
        'dropout_rate': trial.suggest_float("dropout_rate", 0, 0.5),
        'top_dense_layer_units': [
            trial.suggest_int("n_units_l{}".format(i), 128, 256+128, log=True) for i in range(n_layers)
        ]
    }
    return params


def make_params_v2(trial):
    n_layers = trial.suggest_int("n_layers", 0, 2)
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 5e-4, 1e-2),
        'adam_epsilon': 1e-8,
        'weight_decay': 0, #trial.suggest_loguniform("weight_decay", 1e-10, 1e-1),
        'dropout_rate': trial.suggest_float("dropout_rate", 0, 0.5),
        'top_dense_layer_units': [
            trial.suggest_int("n_units_l{}".format(i), 128, 256+128, log=True) for i in range(n_layers)
        ]
    }
    return params


def make_params_v3(trial):
    params = {
        'learning_rate': trial.suggest_loguniform("learning_rate", 5e-7, 1e-1), #trial.suggest_loguniform("learning_rate", 5e-4, 1e-2),
        'adam_epsilon': 1e-8,
        'weight_decay': 0, #trial.suggest_loguniform("weight_decay", 1e-10, 1e-1),
        'dropout_rate': trial.suggest_float("dropout_rate", 0, 0.5),
        'top_dense_layer_units': []
    }
    return params


if __name__ == '__main__':
    tune_hparams(make_params_v1)
    tune_hparams(make_params_v2)
    tune_hparams(make_params_v3)
