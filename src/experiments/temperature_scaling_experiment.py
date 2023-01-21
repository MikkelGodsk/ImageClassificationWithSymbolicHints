import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src.experiments.main import *

# For test: train_val_split is changed and validation set is split.

dataset = "CMPlaces" #"CMPlaces"#'ImageNet'
val_split = True  # Whether to use a small partition of the training set, or the validation set
train_val_split=0.98  # Used if val_split is True

# Setup
print("Obtaining dataset", flush=True)
set_dataset(dataset)
if not val_split:
    val_text_ds: TextModalityDS = load_text_ds('val_text.json')
    val_img_ds: ImageModalityDS = load_img_ds('val')
    bimodal_val_ds: torch.utils.data.DataLoader = bimodal_dataloader(BimodalDS(image_ds=val_img_ds, text_ds=val_text_ds), batch_size=1)
else:
    val_text_ds: TextModalityDS = load_text_ds('train_text.json')
    val_img_ds: ImageModalityDS = load_img_ds('train')
    _, val_text_split = split_text_ds(val_text_ds, train_val_split)
    _, val_img_split = split_img_ds(val_img_ds, train_val_split)
    bimodal_val_ds: torch.utils.data.DataLoader = bimodal_dataloader(BimodalDS(image_ds=val_img_ds, text_ds=val_text_ds), batch_size=1, num_workers=0)

# log priors
log_prior_class_probs = None
try:
    if dataset.lower() == 'imagenet':
        log_prior_class_probs = torch.Tensor(np.loadtxt('imagenet_prior.txt')).unsqueeze(0)
    else:
        log_prior_class_probs = torch.Tensor(np.loadtxt('cmplaces_val_prior.txt')).unsqueeze(0)
except FileNotFoundError:
    print("Class priors must be computed beforehand. Run the file `compute_class_priors.py`.")
    
mask = ~torch.isinf(log_prior_class_probs).view(1,-1)  # We keep everything finite.
log_prior_class_probs[~mask] = 0.0
    
    
# Get classifiers (functions should've been put elsewhere for reusability across experiments.)
num_classes = 1000 if dataset.lower() == "imagenet" else 205
bert_embedding_dim = 1024 if dataset.lower() == "imagenet" else 768
bert_model_name = "bert-large-cased" if dataset.lower() == "imagenet" else "bert-base-cased"
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

print("Obtaining models", flush=True)
img_clf = get_img_classifier()
text_clf = get_text_classifier(True)

# Inference
print("Performing inference", flush=True)
logits_img = []
logits_txt = []
labels = []
with torch.no_grad():
    img_clf.eval()
    text_clf.eval()
    i = 0
    for x, y in bimodal_val_ds:
        img, txt = x
        logits_img.append(img_clf.forward_no_softmax(img).detach().cpu())   # Seems there could be a memory leak...
        logits_txt.append(text_clf.forward_no_softmax(txt).detach().cpu())
        labels.append(y)
        #i+=1
        if i>10:
            break

logits_img = torch.stack(logits_img)
logits_txt = torch.stack(logits_txt)
labels = torch.Tensor(labels).type(torch.int32)

# Matrix of accuracies
print("Computing accuracies", flush=True)
print('Image clf accuracy: {:.2f}'.format(torchmetrics.functional.accuracy(logits_img, labels)), flush=True)
print('Text clf accuracy: {:.2f}'.format(torchmetrics.functional.accuracy(logits_txt, labels)), flush=True)
temperatures = np.logspace(np.log10(1/8), np.log10(8), 15)
accuracies = np.zeros((len(temperatures), len(temperatures)))
for i, img_temp in enumerate(temperatures):   # Image classifier temperatures along rows
    for j, txt_temp in enumerate(temperatures):    # Text classifier temperatures along columns
        z = logits_img*img_temp + logits_txt*txt_temp - log_prior_class_probs  # Log prior is the culprit.
        z.mul(mask)
        accuracies[i,j] = torchmetrics.functional.accuracy(z, labels)

np.savetxt("Temperature_scaling_accuracies_{:s}".format(dataset) + '_val_split.txt' if val_split else '.txt', accuracies)
