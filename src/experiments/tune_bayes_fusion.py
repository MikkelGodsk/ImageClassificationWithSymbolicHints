import optuna
import typing
import numpy as np

from main import *

#TODO: Perhaps give callbacks in experiments?


# Parameters
dataset="CMPlaces"#"ImageNet" #"CMPlaces"#"ImageNet"      # Overwrite in main


# Other stuff related to the dataset
num_classes = 1000 if dataset.lower() == "imagenet" else 205
bert_embedding_dim = 1024 if dataset.lower() == "imagenet" else 768
bert_model_name = "bert-large-cased" if dataset.lower() == "imagenet" else "bert-base-cased"

log_prior_class_probs = None
try:
    if dataset.lower() == 'imagenet':
        log_prior_class_probs = torch.Tensor(np.loadtxt('imagenet_prior.txt'))
    else:
        log_prior_class_probs = torch.Tensor(np.loadtxt('cmplaces_train_prior.txt'))
except FileNotFoundError:
    print("Class priors must be computed beforehand. Run the file `compute_class_priors.py`.")


# Overwrite functions from main for convenience
def get_img_classifier() -> img_clf_t:
    if dataset.lower() == "imagenet":
        resnet50_clf: ResNet50.LitResNet50Model = ResNet50.LitResNet50Model()
        return resnet50_clf
        
    elif dataset.lower() == "cmplaces":
        # Load VGG16
        vgg16_clf: VGG16.LitVGG16Model = VGG16.LitVGG16Model()
        return vgg16_clf
    else:
        raise ValueError
        
    
def get_text_classifier(load_bert: bool, train_text_ds=None):
    if dataset.lower() == "imagenet":
        bert_params = {
            'learning_rate': 0.0005008982647821122,
            'adam_epsilon': 1e-08,
            'weight_decay': 0.0,
            'dropout_rate': 0.005137990668310669,
            'top_dense_layer_units': [],
            'num_classes': num_classes,
            'bert_embedding_dim': bert_embedding_dim,
            'bert_model_name': bert_model_name
        }
        
    elif dataset.lower() == "cmplaces":
        # Train BERT
        bert_params = {
            'learning_rate': 0.0006870443398072322,
            'adam_epsilon': 1e-08,
            'weight_decay': 0.0,
            'dropout_rate': 0.31547838646677684,
            'top_dense_layer_units': [],
            'num_classes': num_classes
        }  # Found using Optuna (see file tune_BERT_hparams.py)
    else:
        raise ValueError
        
    if not load_bert:
        text_clf: BERT.LitBERTModel = BERT.train_and_evaluate(
            bert_params, 
            train_ds = text_dataloader(train_text_ds),
            n_epochs=5, 
            model_version="Train_BERT"
        )[0]
        torch.save(text_clf.state_dict(), "BERT_model_"+dataset.lower())
    else:
        text_clf: BERT.LitBERTModel = BERT.LitBERTModel(**bert_params)
        text_clf.load_state_dict(torch.load("BERT_model_"+dataset.lower()))
    return text_clf
    
    
# Optimization stuff
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
    
    text_clf: text_clf_t = get_text_classifier(load_bert=True)
    img_clf: img_clf_t = get_img_classifier()
    fusion_model: MultimodalModel.NaiveBayesFusion = MultimodalModel.NaiveBayesFusion(
        img_clf, 
        text_clf, 
        learning_rate=params['lr'],
        adam_epsilon=1e-08, 
        weight_decay=[params['img_l2_rate'], params['text_l2_rate']]
    )
    return experiment5(
        fusion_model,
        train_ds,
        val_split,
        n_epochs=max_epochs,
        make_plots=False,
        train_log_prior_probs=log_prior_class_probs,
        val_log_prior_probs=log_prior_class_probs,   # The dataset is assumed to be big enough for these probabilities to be roughly the same
        params=params
    )[0]
    
    
def hyperparam_tuning(
    train_ds: torch.utils.data.DataLoader, 
    val_split: torch.utils.data.DataLoader, 
    make_params: typing.Callable[[optuna.trial.Trial], typing.Dict[str, typing.Union[float, int, typing.List[int]]]], 
    n_trials: int, 
    timeout: int, 
    max_epochs: int
): 
    pruner: optuna.pruners.BasePruner = (optuna.pruners.MedianPruner())
    logger_fn = lambda: pl.loggers.TensorBoardLogger(save_dir="~/bscproj/CMPlaces/lightning_logs")
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, train_ds, val_split, make_params, max_epochs, logger_fn), 
        n_trials=n_trials, 
        timeout=timeout
    )
    
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    return trial.params
        
    
def tune_hparams(make_params: typing.Callable[[optuna.trial.Trial], typing.Dict[str, typing.Union[float, int, typing.List[int]]]], n_trials: int = 2, timeout: int = 86100):
    global log_prior_class_probs
    set_dataset(dataset)
    
    text_ds: TextModalityDS = load_text_ds('train_text.json')
    img_ds: ImageModalityDS = load_img_ds('train')  # This call will take a long time...
    #text_ds: TextModalityDS = load_text_ds('val_text.json')  # For debugging only
    #img_ds: ImageModalityDS = load_img_ds('val')
    
    train_val_split=0.8
    train_text_split, val_text_split = split_text_ds(text_ds, train_val_split)
    train_img_split, val_img_split = split_img_ds(img_ds, train_val_split)
    train_split = BimodalDS(image_ds=train_img_split, text_ds=train_text_split)
    val_split = BimodalDS(image_ds=val_img_split, text_ds=val_text_split)
        
    return hyperparam_tuning(train_split, val_split, make_params, n_trials=n_trials, timeout=timeout, max_epochs=1)



if __name__ == '__main__':
    def make_params(trial):
        return {
            'img_l2_rate': trial.suggest_loguniform("img_l2_rate", 1e-15, 1),
            'lr': trial.suggest_loguniform("lr", 5e-7, 1),
            'text_l2_rate': trial.suggest_loguniform("text_l2_rate", 1e-15, 1)
        }
    print("Using dataset {:s}".format(dataset), flush=True)
    tune_hparams(make_params)
    #raise ValueError("Debugging on validation set done (10 trials)")
    
