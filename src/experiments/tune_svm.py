import numpy as np

from src.experiments.main import *

trial_no = 1  # of 4  (one-based indexing)

reg_strengths = [
    np.array([1e-7, 3e-7, 1e-6, 3e-6]),
    np.array([0.0003, 0.001, 0.003]),
    np.array([0.00001, 0.00003, 0.0001, 0.0003]),
    np.array([0.0001, 0.001, 0.01, 0.1]),
][trial_no - 1]
print(reg_strengths)

dataset_name = "ImageNet"  # 'CMPlaces'

print("Using dataset {:s}".format(dataset_name), flush=True)

set_dataset(dataset_name)
text_ds: TextModalityDS = load_text_ds("train_text.json")
img_ds: ImageModalityDS = load_img_ds("train")  # This call will take a long time...
# text_ds: TextModalityDS = load_text_ds('val_text.json')   # For debugging only
# img_ds: ImageModalityDS = load_img_ds('val')

train_text_ds, val_text_ds = split_text_ds(text_ds, train_val_split=0.8)
train_img_ds, val_img_ds = split_img_ds(img_ds, train_val_split=0.8)
train_split = BimodalDS(image_ds=train_img_ds, text_ds=train_text_ds)
val_split = BimodalDS(image_ds=val_img_ds, text_ds=val_text_ds)

print("\n\033[1m\033[34mExperiment 1\033[0m")
models: Tuple[text_clf_t, img_clf_t] = experiment1(
    bimodal_val_ds=val_split,
    train_text_ds=train_text_ds,
    load_bert=False,
    dataset_name=dataset_name,
)
bert_clf: text_clf_t = models[0]
vgg16_clf: img_clf_t = models[1]


for reg_strength in reg_strengths:
    print("Regularization strength: {:e}".format(reg_strength))
    experiment4a(
        bert_clf,
        vgg16_clf,
        train_split,
        val_split,
        regularization_strength=reg_strength,
        n_classes=205 if dataset_name.lower() == "cmplaces" else 1000,
    )
