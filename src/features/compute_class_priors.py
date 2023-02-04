import importlib

import numpy as np

from src.features.dataset import set_dataset, load_text_ds, load_img_ds, TextModalityDS, ImageModalityDS, BimodalDS
from src.models.MultimodalModel import estimate_prior_class_probs


for ds in ["ImageNet", "CMPlaces"]:
    set_dataset(ds)

    text_ds: TextModalityDS = load_text_ds("val_text.json")
    img_ds: ImageModalityDS = load_img_ds("val")

    log_prior_class_probs = estimate_prior_class_probs(
        BimodalDS(image_ds=img_ds, text_ds=text_ds), 
        n_classes=205 if ds == 'CMPlaces' else 1000 # Global variable from src.features.dataset
    ).numpy()
    np.savetxt(f"{ds.lower()}_prior.txt", log_prior_class_probs)
