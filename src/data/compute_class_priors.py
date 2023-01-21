import importlib
import numpy as np
from main import *
"""
set_dataset('CMPlaces')

text_ds: TextModalityDS = load_text_ds('val_text.json')  # For debugging only
img_ds: ImageModalityDS = load_img_ds('val')

log_prior_class_probs = MultimodalModel.estimate_prior_class_probs(BimodalDS(image_ds=img_ds, text_ds=text_ds), n_classes=205).numpy()
np.savetxt('cmplaces_val_prior.txt', log_prior_class_probs)


text_ds: TextModalityDS = load_text_ds('train_text.json')
img_ds: ImageModalityDS = load_img_ds('train')  # This call will take a long time...


log_prior_class_probs = MultimodalModel.estimate_prior_class_probs(BimodalDS(image_ds=img_ds, text_ds=text_ds), n_classes=205).numpy()
np.savetxt('cmplaces_train_prior.txt', log_prior_class_probs)
"""
#importlib.reload(dataset)
set_dataset('ImageNet')

text_ds: TextModalityDS = load_text_ds('val_text.json')
img_ds: ImageModalityDS = load_img_ds('val')

log_prior_class_probs = MultimodalModel.estimate_prior_class_probs(BimodalDS(image_ds=img_ds, text_ds=text_ds), n_classes=1000).numpy()
np.savetxt('imagenet_prior.txt', log_prior_class_probs)